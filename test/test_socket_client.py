from multiprocessing.connection import Client
import time

NUM_CLIENTS = 8

address = ('localhost', 6000)
conns = []
for i in range(NUM_CLIENTS):
    conn = Client(address)
    conns.append(conn)

begin = time.time()
for i in range(NUM_CLIENTS):
    out = conns[i].recv()
    # print(out)

for i in range(NUM_CLIENTS):
    conns[i].send((100, 200))

end = time.time()
print((end - begin) * 1000)

for i in range(NUM_CLIENTS):
    conns[i].close()
