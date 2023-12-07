import EmbCache
import sys
import time

assert len(sys.argv) == 4
server_name = sys.argv[1]
client_name = sys.argv[2]
sleep_time = int(sys.argv[3])


ipc_ctx = EmbCache.capi.ClientIPCContext()
client_id = ipc_ctx.build_connection(server_name, client_name)
time.sleep(sleep_time)
ipc_ctx.send("msg from client {}".format(client_name))
