# Import the required modules
from IPython.display import clear_output
import socket
import sys
import cv2
import matplotlib.pyplot as plt
import pickle
import numpy as np
import struct ## new
import zlib
from PIL import Image, ImageOps

HOST='192.168.20.21'
PORT=8485
VIDEO_NAME='WebCAM_Ouput.mp4'
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST,PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

conn,addr=s.accept()

#video record
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_NAME, fourcc, 30.0, (640,  480))

data = b""
payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))
while True:
    try:
        while len(data) < payload_size:
            data += conn.recv(4096)
            if not data:
                cv2.destroyAllWindows()
                conn,addr=s.accept()
                continue
        # receive image row data form client socket
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        # unpack image using pickle 
        frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        cv2.imshow('server',frame)
        cv2.waitKey(1)
        out.write(frame)
    except Exception as err:
        print(err)
        out.release()
        cv2.destroyAllWindows()
        break
s.close()