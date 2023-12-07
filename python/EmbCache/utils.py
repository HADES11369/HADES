import torch
import torch.distributed as dist


def alltoall_cpu(
    rank,
    world_size,
    output_tensor_list,
    input_tensor_list,
    group=None,
):
    """Each process scatters list of input tensors to all processes in a cluster
    and return gathered list of tensors in output list. The tensors should have the same shape.

    Parameters
    ----------
    rank : int
        The rank of current worker
    world_size : int
        The size of the entire communicator
    output_tensor_list : List of tensor
        The received tensors
    input_tensor_list : List of tensor
        The tensors to exchange
    """
    input_tensor_list = [
        tensor.to(torch.device("cpu")) for tensor in input_tensor_list
    ]
    for i in range(world_size):
        dist.scatter(
            output_tensor_list[i],
            input_tensor_list if i == rank else [],
            src=i,
            group=group,
        )


def alltoallv_cpu(
    rank,
    world_size,
    output_tensor_list,
    input_tensor_list,
    group=None,
):
    """Each process scatters list of input tensors to all processes in a cluster
    and return gathered list of tensors in output list.

    Parameters
    ----------
    rank : int
        The rank of current worker
    world_size : int
        The size of the entire communicator
    output_tensor_list : List of tensor
        The received tensors
    input_tensor_list : List of tensor
        The tensors to exchange
    """
    # send tensor to each target trainer using torch.distributed.isend
    # isend is async
    senders = []
    for i in range(world_size):
        if i == rank:
            output_tensor_list[i] = input_tensor_list[i].to(
                torch.device("cpu"))
        else:
            sender = dist.isend(
                input_tensor_list[i].to(torch.device("cpu")),
                dst=i,
                group=group,
            )
            senders.append(sender)

    for i in range(world_size):
        if i != rank:
            dist.recv(output_tensor_list[i], src=i, group=group)

    dist.barrier()


def alltoall(
    rank,
    world_size,
    output_tensor_list,
    input_tensor_list,
    device,
    group=None,
):
    """Each process scatters list of input tensors to all processes in a cluster
    and return gathered list of tensors in output list. The tensors should have the same shape.

    Parameters
    ----------
    rank : int
        The rank of current worker
    world_size : int
        The size of the entire communicator
    output_tensor_list : List of tensor
        The received tensors
    input_tensor_list : List of tensor
        The tensors to exchange
    device: torch.device
        Device of the tensors
    """
    if dist.get_backend() == "nccl":
        input_tensor_list = [tensor.cuda() for tensor in input_tensor_list]
        output_tensor_list = [tensor.cuda() for tensor in output_tensor_list]
        dist.all_to_all(output_tensor_list, input_tensor_list, group=group)
    else:
        input_tensor_list = [tensor.cpu() for tensor in input_tensor_list]
        output_tensor_list = [tensor.cpu() for tensor in output_tensor_list]
        alltoall_cpu(
            rank,
            world_size,
            output_tensor_list,
            input_tensor_list,
            group=group,
        )
    output_tensor_list = [tensor.to(device) for tensor in output_tensor_list]


def alltoallv(
    rank,
    world_size,
    output_tensor_list,
    input_tensor_list,
    device,
    group=None,
):
    """Each process scatters list of input tensors to all processes in a cluster
    and return gathered list of tensors in output list.

    Parameters
    ----------
    rank : int
        The rank of current worker
    world_size : int
        The size of the entire communicator
    output_tensor_list : List of tensor
        The received tensors
    input_tensor_list : List of tensor
        The tensors to exchange
    device: torch.device
        Device of the tensors
    """
    if dist.get_backend() == "nccl":
        input_tensor_list = [tensor.cuda() for tensor in input_tensor_list]
        output_tensor_list = [tensor.cuda() for tensor in output_tensor_list]
        dist.all_to_all(output_tensor_list, input_tensor_list, group=group)
    else:
        input_tensor_list = [tensor.cpu() for tensor in input_tensor_list]
        output_tensor_list = [tensor.cpu() for tensor in output_tensor_list]
        alltoallv_cpu(
            rank,
            world_size,
            output_tensor_list,
            input_tensor_list,
            group=group,
        )
    output_tensor_list = [tensor.to(device) for tensor in output_tensor_list]
