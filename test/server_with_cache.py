import sys
import torch
import EmbCache
import time

assert len(sys.argv) == 3
server_name = sys.argv[1]
client_num = int(sys.argv[2])

# test init
server = EmbCache.EmbCacheServer(server_name, client_num)
