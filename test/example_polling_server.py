import EmbCache
import sys
import time
from queue import Queue

assert len(sys.argv) == 3
server_name = sys.argv[1]
client_num = int(sys.argv[2])

ipc_ctx = EmbCache.capi.ServerIPCContext()
ipc_ctx.build_connection(server_name, client_num)

polling_queue = Queue(client_num)
for i in range(client_num):
    polling_queue.put(i)

results = []
while not polling_queue.empty():
    client = polling_queue.get()
    result = ipc_ctx.try_recv_one(client)
    if (len(result) > 0):
        results.append((client, result))
        print(results)
    else:
        polling_queue.put(client)
    time.sleep(0.5)

