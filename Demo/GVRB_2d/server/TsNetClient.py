#! /usr/bin/env python
# -*- coding:utf-8 -*-
# version : Python 2.7.13
import pyDOE
import os, sys, time
import socket
from set_predictor import n2s, s2n

def doConnect(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
    except:
        pass
    return sock

def EngCli(sample):
    # print(sample)
    HOST = 'localhost'
    PORT = 21568
    BUFSIZ = 1024 * 10
    tcpCliSock = doConnect(HOST, PORT)
    finished = 0
    while finished == 0:
        try:
            msg = str.encode(sample + '\n')
            tcpCliSock.send(msg)
            data = tcpCliSock.recv(BUFSIZ).decode()
            finished = 1
        except socket.error:
            print("\r\nsocket error,do reconnect ")
            time.sleep(5)
            tcpCliSock = doConnect(HOST, PORT)
        except:
            print('\r\nother error occur ')
            time.sleep(5)

        time.sleep(5)

    tcpCliSock.close()
    return data







if __name__ == "__main__":
    X = pyDOE.lhs(93, samples=32, criterion='maximin')
    for ii in range(32):
        Y = EngCli(n2s(X[ii]))
        print(Y)


