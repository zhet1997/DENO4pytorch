# 2021-8-18
import os
import socketserver
import time
from threading import Lock, BoundedSemaphore
from set_predictor import build_predicter, run_predicter, s2n
import numpy as np

class MyServer(socketserver.StreamRequestHandler):

    def handle(self):

        print("conn is :", self.request)  # conn
        print("addr is :", self.client_address)  # addr
        model = build_predicter()
        while True:
            try:
                time.sleep(0.1)
                data = self.rfile.readline().decode()
                if not data:
                    break

                source.acquire()
                mutex.acquire()
                data = run_predicter(np.array(s2n(data)), model=model)
                time.sleep(0.1)
                mutex.release()

                data = [str(x) for x in data.tolist()]
                data = '\t'.join(data)
                data = data.encode()
                print(data)
                source.release()
                self.wfile.write(data)

            except Exception as e:
                print(e)
                break

if __name__ == "__main__":
    HOST = ''
    PORT = 21568
    ADDR = (HOST, PORT)
    mutex = Lock()
    MAX = 60
    source = BoundedSemaphore(MAX)
    s = socketserver.ThreadingTCPServer(ADDR, MyServer)
    print('the EngServer is start!')
    s.serve_forever()
