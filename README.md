# How to run

Install:

```shell
bash install.sh
```

Run:

1. Partition graph:

   See [example/utils/partition_graph.py](./example/utils/partition_graph.py).

2. Configure IP address:

   Edit [example/ip_config.txt](example/ip_config.txt), add the IPs of all the machines that will participate in the training (they can access each other by SSH without password). For example:

   ```
   192.168.1.51
   192.168.1.52
   ```

3. Launch EmbCacheServer:

   ```shell
   bash run_server.sh
   ```

   Where `-n` indicates the number of machines.

4. Launch trainers:

   ```shell
   bash run_trainer.sh
   ```

   Where `--num_trainers` indicates the number of trainers per machines.
   The number of lines in `ip_config` should be equal to the number of machines as well as graph partitions.