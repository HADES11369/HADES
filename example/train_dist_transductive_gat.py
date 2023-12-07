import argparse
import time

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
from models import DistGAT, compute_acc

import EmbCache

import torch

torch.manual_seed(25)


def presampling(dataloader, num_nodes, num_epochs=1):
    presampling_heat = torch.zeros((num_nodes, ), dtype=torch.float32)
    sampling_times = 0
    for epoch in range(num_epochs):
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            torch.cuda.synchronize()
            # for block in blocks:
            #     frontier = block.ndata[dgl.NID]["_N"][block.srcnodes()]
            #     presampling_heat[frontier] += 1
            presampling_heat[input_nodes] += 1
            sampling_times += 1
    if torch.distributed.get_backend() == "nccl":
        presampling_heat = presampling_heat.cuda()
        torch.distributed.all_reduce(presampling_heat,
                                     torch.distributed.ReduceOp.SUM)
        sampling_times = torch.tensor([sampling_times], device="cuda")
        torch.distributed.all_reduce(sampling_times,
                                     torch.distributed.ReduceOp.SUM)
        presampling_heat = presampling_heat / sampling_times
        presampling_heat = presampling_heat.cpu()
    else:
        torch.distributed.all_gather(presampling_heat,
                                     torch.distributed.ReduceOp.SUM)
        sampling_times = torch.tensor([sampling_times])
        torch.distributed.all_reduce(sampling_times,
                                     torch.distributed.ReduceOp.SUM)
        presampling_heat = presampling_heat / sampling_times
    presampling_heat_accessed = presampling_heat[presampling_heat > 0]
    print("Presampling done, max: {:.3f} min: {:.3f} avg: {:.3f}".format(
        torch.max(presampling_heat_accessed).item(),
        torch.min(presampling_heat_accessed).item(),
        torch.mean(presampling_heat_accessed).item()))

    return presampling_heat


def load_embs(standalone, emb_layer, g):
    nodes = dgl.distributed.node_split(np.arange(g.num_nodes()),
                                       g.get_partition_book(),
                                       force_even=True)
    x = dgl.distributed.DistTensor(
        (
            g.num_nodes(),
            emb_layer.emb_dim,
        ),
        th.float32,
        "eval_embs",
        persistent=True,
    )
    num_nodes = nodes.shape[0]
    for i in range((num_nodes + 1023) // 1024):
        idx = nodes[i * 1024:(i + 1) * 1024 if (i + 1) *
                    1024 < num_nodes else num_nodes]
        embeds = emb_layer(idx).cpu()
        x[idx] = embeds

    if not standalone:
        g.barrier()

    return x


def evaluate(
    standalone,
    model,
    emb_layer,
    g,
    labels,
    val_nid,
    test_nid,
    batch_size,
    device,
):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    if not standalone:
        model = model.module
    model.eval()
    emb_layer.eval()
    with th.no_grad():
        inputs = load_embs(standalone, emb_layer, g)
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    emb_layer.train()
    return compute_acc(pred[val_nid],
                       labels[val_nid]), compute_acc(pred[test_nid],
                                                     labels[test_nid])


def run(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, n_classes, g = data
    fan_out = [int(fanout) for fanout in args.fan_out.split(",")]
    sampler = dgl.dataloading.NeighborSampler(fan_out)
    dataloader = dgl.dataloading.DistNodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    if args.presampling:
        hotness_list = presampling(dataloader, g.num_nodes())
    else:
        hotness_list = g.out_degrees()
    hot_threshold = args.hot_threshold
    print("#hot_nodes = {}".format((hotness_list
                                    >= hot_threshold).nonzero().numel()))

    print("Create distemb")
    emb_layer = EmbCache.DistEmb(g.num_nodes(),
                                 args.num_hidden,
                                 emb_dtype=torch.float32,
                                 name="emb_layer",
                                 cache_rate=args.emb_cache_rate,
                                 idx_dtype=train_nid.dtype,
                                 local_rank=th.distributed.get_rank() %
                                 args.num_gpus,
                                 local_trainer_num=args.num_gpus,
                                 hotness_list=hotness_list,
                                 hot_threshold=hot_threshold,
                                 part_book=g.get_partition_book(),
                                 cache_server_addr=args.cache_server_addr,
                                 cache_server_port=args.cache_server_port)

    print("Create model")
    gat_heads = [int(head) for head in args.heads.split(",")]
    model = DistGAT(args.num_hidden,
                    args.model_num_hidden,
                    n_classes,
                    len(fan_out),
                    gat_heads,
                    activation=F.relu,
                    feat_dropout=args.feat_dropout,
                    attn_dropout=args.attn_dropout)
    model = model.to(device)
    if not args.standalone:
        if args.num_gpus == -1:
            model = th.nn.parallel.DistributedDataParallel(model)
        else:
            dev_id = th.distributed.get_rank() % args.num_gpus
            model = th.nn.parallel.DistributedDataParallel(
                model, device_ids=[dev_id], output_device=dev_id)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)

    print("Create model optimizer")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Create emb optimizer")
    emb_optimizer = EmbCache.SparseAdam(
        [emb_layer.sparse_emb], args.sparse_lr,
        g.get_partition_book().num_partitions(), args.local_group,
        args.partition_group)

    print("Start train")
    # Training loop
    iter_tput = []
    epoch_time_log = []
    epoch = 0
    for epoch in range(args.num_epochs):
        tic = time.time()

        sample_time = 0
        load_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        emb_update_time = 0
        num_seeds = 0
        num_inputs = 0

        with model.join():
            # Loop over the dataloader to sample the computation dependency
            # graph as a list of blocks.
            step_time = []
            tic = time.time()
            tic_step = time.time()
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                torch.cuda.synchronize()
                sample_time += time.time() - tic_step

                load_begin = time.time()
                num_seeds += len(blocks[-1].dstdata[dgl.NID])
                num_inputs += len(blocks[0].srcdata[dgl.NID])
                blocks = [block.to(device) for block in blocks]
                batch_labels = g.ndata["labels"][seeds].long().to(device)
                batch_inputs = emb_layer(input_nodes)
                torch.cuda.synchronize()
                load_time += time.time() - load_begin

                forward_start = time.time()
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, batch_labels)
                torch.cuda.synchronize()
                forward_time += time.time() - forward_start

                backward_begin = time.time()
                emb_optimizer.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                torch.cuda.synchronize()
                backward_time += time.time() - backward_begin

                update_start = time.time()
                optimizer.step()
                torch.cuda.synchronize()
                update_time += time.time() - update_start

                emb_update_start = time.time()
                emb_optimizer.step()
                torch.cuda.synchronize()
                emb_update_time += time.time() - emb_update_start

                step_t = time.time() - tic_step
                step_time.append(step_t)
                iter_tput.append(len(blocks[-1].dstdata[dgl.NID]) / step_t)
                if step % args.log_every == 0:
                    acc = compute_acc(batch_pred, batch_labels)
                    gpu_mem_alloc = (th.cuda.max_memory_allocated() /
                                     1000000 if th.cuda.is_available() else 0)
                    print(
                        "Part {} | Epoch {:05d} | Step {:05d} | Loss {:.4f} | "
                        "Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU "
                        "{:.1f} MB | time {:.3f} s".format(
                            th.distributed.get_rank(),
                            epoch,
                            step,
                            loss.item(),
                            acc.item(),
                            np.mean(iter_tput[3:]),
                            gpu_mem_alloc,
                            np.sum(step_time[-args.log_every:]),
                        ))
                    train_acc_tensor = torch.tensor([acc.item()]).cuda()
                    torch.distributed.all_reduce(
                        train_acc_tensor, torch.distributed.ReduceOp.SUM)
                    train_acc_tensor /= torch.distributed.get_world_size()
                    if torch.distributed.get_rank() == 0:
                        print("Avg train acc {:.4f}".format(
                            train_acc_tensor[0].item()))
                tic_step = time.time()

        toc = time.time()
        epoch += 1

        th.distributed.barrier()
        timetable = ("=====================\n"
                     "Part {}, Epoch Time(s): {:.4f}\n"
                     "Sampling Time(s): {:.4f}\n"
                     "Loading Time(s): {:.4f}\n"
                     "Forward Time(s): {:.4f}\n"
                     "Backward Time(s): {:.4f}\n"
                     "Update Time(s): {:.4f}\n"
                     "Emb Update Time(s): {:.4f}\n"
                     "#seeds: {}\n"
                     "#inputs: {}\n"
                     "=====================".format(
                         th.distributed.get_rank(),
                         toc - tic,
                         sample_time,
                         load_time,
                         forward_time,
                         backward_time,
                         update_time,
                         emb_update_time,
                         num_seeds,
                         num_inputs,
                     ))
        print(timetable)
        epoch_time_log.append(toc - tic)

        if epoch % args.eval_every == 0 and epoch != 0:
            start = time.time()
            val_acc, test_acc = evaluate(
                args.standalone,
                model,
                emb_layer,
                g,
                g.ndata["labels"],
                val_nid,
                test_nid,
                args.batch_size_eval,
                device,
            )
            print("Part {}, Val Acc {:.4f}, Test Acc {:.4f}, time: {:.4f}".
                  format(th.distributed.get_rank(), val_acc, test_acc,
                         time.time() - start))
            acc_tensor = torch.tensor([val_acc, test_acc]).cuda()
            torch.distributed.all_reduce(acc_tensor,
                                         torch.distributed.ReduceOp.SUM)
            acc_tensor /= torch.distributed.get_world_size()
            if th.distributed.get_rank() == 0:
                print("All parts avg val acc {:.4f}, test acc {:.4f}".format(
                    acc_tensor[0].item(), acc_tensor[1].item()))

    print("Rank {}, Avg Epoch Time(s): {:.4f}".format(
        th.distributed.get_rank(), np.mean(epoch_time_log[2:])))
    epoch_time_tensor = torch.tensor([np.mean(epoch_time_log[2:])]).cuda()
    torch.distributed.all_reduce(epoch_time_tensor,
                                 torch.distributed.ReduceOp.SUM)
    epoch_time_tensor /= torch.distributed.get_world_size()
    train_nids_tensor = torch.tensor([train_nid.shape[0]]).cuda()
    torch.distributed.all_reduce(train_nids_tensor)
    if th.distributed.get_rank() == 0:
        all_parts_avg_epoch_time = epoch_time_tensor[0].item()
        total_train_nids_num = train_nids_tensor[0].item()
        print(
            "All parts avg epoch time {:.4f} sec, throughput {:.4f} seeds/sec".
            format(all_parts_avg_epoch_time,
                   total_train_nids_num / all_parts_avg_epoch_time))


def main(args):
    dgl.distributed.initialize(args.ip_config)
    if not args.standalone:
        th.distributed.init_process_group(backend="nccl")
    local_group, groups = th.distributed.new_subgroups(args.num_gpus)
    args.local_group = local_group

    # partition group
    ranks_list = []
    for i in range(args.num_gpus):
        rank = [i + v * args.num_gpus for v in range(len(groups))]
        ranks_list.append(rank)
    partition_group, groups = th.distributed.new_subgroups_by_enumeration(
        ranks_list)
    args.partition_group = partition_group

    g = dgl.distributed.DistGraph(args.graph_name,
                                  part_config=args.part_config)
    # since g.rank() != th.distributed.get_rank(), so only use one of them
    print("rank:", th.distributed.get_rank())

    pb = g.get_partition_book()
    train_nid = dgl.distributed.node_split(g.ndata["train_mask"],
                                           pb,
                                           force_even=True)
    if args.graph_name == "mag240m":
        val_nid = dgl.distributed.node_split(g.ndata["valid_mask"],
                                             pb,
                                             force_even=True)
    else:
        val_nid = dgl.distributed.node_split(g.ndata["val_mask"],
                                             pb,
                                             force_even=True)
    test_nid = dgl.distributed.node_split(g.ndata["test_mask"],
                                          pb,
                                          force_even=True)
    local_nid = pb.partid2nids(pb.partid).detach().numpy()
    print("part {}, train: {} (local: {}), val: {} (local: {}), test: {} "
          "(local: {})".format(
              th.distributed.get_rank(),
              len(train_nid),
              len(np.intersect1d(train_nid.numpy(), local_nid)),
              len(val_nid),
              len(np.intersect1d(val_nid.numpy(), local_nid)),
              len(test_nid),
              len(np.intersect1d(test_nid.numpy(), local_nid)),
          ))
    if args.num_gpus == -1:
        device = th.device("cpu")
    else:
        dev_id = th.distributed.get_rank() % args.num_gpus
        device = th.device("cuda:" + str(dev_id))
        th.cuda.set_device(device)
    labels = g.ndata["labels"][np.arange(g.num_nodes())]
    n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
    print("#labels:", n_classes)

    # Pack data
    data = train_nid, val_nid, test_nid, n_classes, g
    run(args, device, data)

    for group in groups:
        th.distributed.destroy_process_group(group)
    print("parent ends")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument("--cache_server_addr", type=str, default="127.0.0.1")
    parser.add_argument("--cache_server_port", type=int, default=32451)
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument("--id", type=int, help="the partition id")
    parser.add_argument("--ip_config",
                        type=str,
                        help="The file for IP configuration")
    parser.add_argument("--part_config",
                        type=str,
                        help="The path to the partition config file")
    parser.add_argument("--n_classes", type=int, help="the number of classes")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=-1,
        help="the number of GPU device. Use -1 for CPU training",
    )
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_hidden", type=int, default=256)
    parser.add_argument("--fan_out", type=str, default="5,10,15")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--feat_dropout", type=float, default=0.1)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--model_num_hidden", type=float, default=32)
    parser.add_argument("--heads", type=str, default="8,8,1")
    parser.add_argument("--local_rank",
                        type=int,
                        help="get rank of the process")
    parser.add_argument("--standalone",
                        action="store_true",
                        help="run in the standalone mode")
    parser.add_argument("--sparse_lr",
                        type=float,
                        default=0.01,
                        help="sparse lr rate")
    parser.add_argument("--presampling",
                        action="store_true",
                        help="use presampling heat")
    parser.add_argument("--hot-threshold", type=float, default=150.0)
    parser.add_argument("--emb-cache-rate", type=float, default=0.2)
    args = parser.parse_args()

    print(args)
    main(args)
