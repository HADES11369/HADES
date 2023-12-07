import sys
import torch
import EmbCache
import time

server_name = 'server'
num_clients = 1

shape = (10, )

server = EmbCache.EmbCacheServer(server_name, num_clients)
print("init done")

# Test 0: Create Tensor
begin = time.time()
my_tensor = server._create_shm_tensor(server_name, "shared" + server_name,
                                      "float32", shape)
end = time.time()
print("Create time: ", end - begin)
my_tensor[:] = 0

# Test 1: measure client
begin = time.time()
recvice_tensors = server._wait_for_uncache_embedding()
print(recvice_tensors)
size = min(recvice_tensors[0][1].shape[0], my_tensor.shape[0])
my_tensor[:size] = recvice_tensors[0][1][:size] + 100
server._send_staled_gradients([server_name])
end = time.time()
print("Client Measure time: ", end - begin)

begin = time.time()
recvice_tensors = server._wait_for_uncache_embedding()
server._send_staled_gradients([server_name])
end = time.time()
print("Client Measure time: ", end - begin)

begin = time.time()
recvice_tensors = server._wait_for_uncache_embedding()
server._send_staled_gradients([server_name])
end = time.time()
print("Client Measure time: ", end - begin)

# Test 2: measure client
## Warm up
begin = time.time()
my_tensor = server._create_shm_tensor(server_name, "shared" + server_name,
                                      "float32", shape)
server._send_staled_gradients([server_name])
recvice_tensors = server._wait_for_uncache_embedding()
end = time.time()
print("Server Measure time: ", end - begin)

## Resue
begin = time.time()
my_tensor = server._create_shm_tensor(server_name, "shared" + server_name,
                                      "float32", shape)
server._send_staled_gradients([server_name])
recvice_tensors = server._wait_for_uncache_embedding()
end = time.time()
print("Server Measure time: ", end - begin)

## Realloc
new_shape = (shape[0] // 2, )
begin = time.time()
my_tensor = server._create_shm_tensor(server_name, "shared" + server_name,
                                      "float32", new_shape)
server._send_staled_gradients([server_name])
recvice_tensors = server._wait_for_uncache_embedding()
end = time.time()
print("Server Measure time: ", end - begin)

# Test 3: Release
begin = time.time()
server._release_shm_tensor(server_name)
end = time.time()
print("Release time: ", end - begin)
