python3 example/launch_train.py --workspace ~/HADES/example/ \
   --num_trainers 2 \
   --num_samplers 1 \
   --num_servers 1 \
   --part_config /data/ogbn-products-2part/ogbn-products.json \
   --ip_config ip_config.txt \
   "~/venv/bin/python3 train_dist_transductive.py --presampling --hot-threshold 0.1  --num_hidden 256 --cache_server_addr 127.0.0.1 --cache_server_port 32451 --graph_name ogbn-products --ip_config ip_config_local.txt --num_gpus 2 --num_epochs 20 --eval_every 1"