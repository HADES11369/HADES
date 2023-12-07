from multiprocessing.connection import Listener
import time

NUM_CLIENTS = 8

address = ('localhost', 6000)  # family is deduced to be 'AF_INET'
listener = Listener(address)

conns = []
for i in range(NUM_CLIENTS):
    conn = listener.accept()
    print('connection accepted from', listener.last_accepted)
    conns.append(conn)

print(conns)

begin = time.time()
for i in range(NUM_CLIENTS):
    conns[i].send((100, 200))

for i in range(NUM_CLIENTS):
    out = conns[i].recv()
    # print(out)
end = time.time()
print((end - begin) * 1000)

for i in range(NUM_CLIENTS):
    conns[i].close()

listener.close()
