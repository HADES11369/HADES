import argparse
import torch
import EmbCache
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=32451)
    parser.add_argument("--num_clients", type=int)
    parser.add_argument("--check_iters", type=int, default=50)
    parser.add_argument("--num-threads", type=int, default=16)
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
    torch.set_num_threads(args.num_threads)
    assert args.num_clients is not None
    server = EmbCache.EmbCacheServer(args.addr, args.port, args.num_clients,
                                     args.check_iters)
