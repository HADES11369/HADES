# HADES
`HADES` is a system for distributed graph neural network (GNN) training with learnable embeddings.

To eliminate the severe data movement bottleneck for globally synchronizing node embeddings and their gradients in the training process, `HADES` exploits node hotness and coldness and introduces the following optimations:

* `Local aggregation` for hot nodes from multiple trainers on the same machine along the conventional immediate synchronization path
* `Cold data cache` for accumulating changes made to cold nodes for delayed, more sparse synchronization

`HADES` is implemented atop `DGL`, a widely adopted GNN training system. Compared to `DGL`, `HADES` introduces a substantial reduction of 18.94-46.28% in data movement overhead and maintains comparable training accuracy, achieving 2.11-2.84x speedups of the end-to-end training performance and an average speedup of 2.53x.

# Install
## Software Version
* Ubuntu 20.04
* CUDA v11.8
* PyTorch v2.0.1
* DGL v1.1.2

## Install HADES
We use `pip` to manage our python environment.

1. Install PyTorch and DGL
2. Download HADES source code
   ```shell
   git clone https://github.com/HADES11369/HADES.git
   ```
3. Install HADES
   ```shell
   cd HADES
   bash install.sh
   ```

## Usage

1. Partition graph:

   See [example/utils/partition_graph.py](./example/utils/partition_graph.py) and follow this [instruction](https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage/dist#step-2-partition-the-graph).

   For `Mag240M`, you can reference this [instruction](https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb_lsc/MAG240M#running-preprocessing-script) to process it into a homogenous graph.

2. Configure IP address:

   Edit [example/ip_config.txt](example/ip_config.txt), and add the IPs of all the machines that will participate in the training (they can access each other by SSH without a password). For example:

   ```
   192.168.1.51
   192.168.1.52
   ```

3. Launch cache server process (run on one machine):

   ```shell
   bash run_server.sh
   ```

   Where `-n` indicates the number of machines and `--num_clients` should equal the number of trainers on each machine.

4. Launch trainers (run on one machine):

   ```shell
   bash run_trainer.sh
   ```

   Where `--num_trainers` and `--num_gpus` indicate the number of trainers on each machine.
   The number of lines in `ip_config` should equal the number of machines and graph partitions.
