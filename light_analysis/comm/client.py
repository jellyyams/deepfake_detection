import socket # for socket
import sys
import os
import time
from enum import Enum

class LogLevel(Enum):
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4


LOG_LEVEL  = LogLevel.DEBUG
    
def log(message, log_level):
    if log_level >= LOG_LEVEL:
        print("CLIENT {}: {}\n".format(log_level, message))

def display_message(message):
    print("Client got following message from SLM server: {}\n".format(message))

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    log("Socket successfully created", LogLevel.INFO)
except socket.error as err:
    log("Socket creation failed with error %s" %(err), LogLevel.ERROR)
 
# default port for socket
port = 6000
 
try:
    host_ip = "192.168.6.2"
except socket.gaierror:
    # this means could not resolve the host
    log("there was an error resolving the host", LogLevel.ERROR)
    sys.exit()
 
# connecting to the server
sock.connect((host_ip, port))

log("The socket has successfully connected.", LogLevel.INFO)
log("Sending test message from client to server.", LogLevel.DEBUG)
sock.send("Ping client from server".encode())

# receive data from the server and decoding to get the string.
print(sock.recv(1024).decode())
time.sleep(1)
sock.send("test.bmp".encode())
print(sock.recv(1024).decode())


bmp_path = '../bmp_temp/barkmitzvah.bmp'
file_size = os.path.getsize(bmp_path)
log("Size of file {} is {} bytes".format(bmp_path, file_size), LogLevel.INFO)
f = open(bmp_path,'rb')
log("Sending file to server", LogLevel.INFO)
l = f.read(1024)
while (l):
    sock.send(l)
    l = f.read(1024)
log("Done sending file to server", LogLevel.INFO)

f.close()

# close the connection
sock.close()    
     