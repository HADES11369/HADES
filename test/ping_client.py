import sys
import torch
import EmbCache
import time

server_name = 'server'
client_name = 'client'

shape = (10, )

# test init
client = EmbCache.EmbCacheClient(server_name, client_name, [100], 0)
print("init done")

# Test 0: Create Tensor
begin = time.time()
my_tensor = client._create_shm_tensor(client_name, "shared" + client_name,
                                      "float32", shape)
end = time.time()
print("Create time: ", end - begin)
my_tensor[:] = 0

# Test 1: measure client
## Warm up
begin = time.time()
my_tensor = client._create_shm_tensor(client_name, "shared" + client_name,
                                      "float32", shape)
my_tensor[:] = 0
client.send_uncache_embedding(client_name)
name, tensor = client.get_staled_gradients()
print(tensor)
end = time.time()
print("Client Measure time: ", end - begin)

## Resue
begin = time.time()
my_tensor = client._create_shm_tensor(client_name, "shared" + client_name,
                                      "float32", shape)
client.send_uncache_embedding(client_name)
name, tensor = client.get_staled_gradients()
end = time.time()
print("Client Measure time: ", end - begin)

## Realloc
new_shape = (shape[0] // 2, )
begin = time.time()
my_tensor = client._create_shm_tensor(client_name, "shared" + client_name,
                                      "float32", new_shape)
client.send_uncache_embedding(client_name)
name, tensor = client.get_staled_gradients()
end = time.time()
print("Client Measure time: ", end - begin)

# Test 2: measure server
begin = time.time()
name, tensor = client.get_staled_gradients()
client.send_uncache_embedding(client_name)
end = time.time()
print("Server Measure time: ", end - begin)

begin = time.time()
name, tensor = client.get_staled_gradients()
client.send_uncache_embedding(client_name)
end = time.time()
print("Server Measure time: ", end - begin)

begin = time.time()
name, tensor = client.get_staled_gradients()
client.send_uncache_embedding(client_name)
end = time.time()
print("Server Measure time: ", end - begin)

# Test 3: Release
begin = time.time()
client._release_shm_tensor(client_name)
end = time.time()
print("Release time: ", end - begin)
