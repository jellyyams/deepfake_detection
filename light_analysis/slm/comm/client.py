import socket # for socket
import sys
import os
import time
from enum import Enum
import glob

class OrderedEnum(Enum):
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

class LogLevel(OrderedEnum):
    ERROR = 4
    WARNING = 3
    INFO = 2
    DEBUG = 1

LOG_LEVEL  = LogLevel.DEBUG
    
def log(message, log_level):
    if log_level >= LOG_LEVEL:
        if log_level == LogLevel.DEBUG:
            print("CLIENT [DEBUG]: {}".format(message))
        elif log_level == LogLevel.INFO:
            print("CLIENT [INFO]: {}".format(message))
        elif log_level == LogLevel.ERROR:
            print("CLIENT [ERROR: {}".format(message))
        elif log_level == LogLevel.WARNING:
            print("CLIENT [WARNING]: {}".format(message))

def display_message(message):
    print("CLIENT [MESSAGE]: {}".format(message))

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
sock.send("Ping SLM server from client".encode())


# receive data from the server and decoding to get the string.
display_message(sock.recv(1024).decode())

frequency = 1
delay = 1/(frequency * 2)
repeat = 5
num_frames = 2

metadata = "METADATA FREQ:{} REP:{} NUMFRAMES:{}".format(frequency, repeat, num_frames)
sock.send(metadata.encode())

display_message(sock.recv(1024).decode()) #wait till files are requested

bmp_dir_path = '../bmp_temp'
bmp_paths = glob.glob(bmp_dir_path + "/*")
for bmp_path in bmp_paths:
    display_message(sock.recv(1024).decode()) #wait till confirmation server is ready for a new file
    file_size = os.path.getsize(bmp_path)
    log("Size of file {} is {} bytes".format(bmp_path, file_size), LogLevel.INFO)
    f = open(bmp_path,'rb')
    log("Sending file to server", LogLevel.INFO)
    l = f.read(1024)
    while (l):
        sock.send(l)
        l = f.read(1024)
        sock.recv(1024).decode()
    log("Done sending file {} to server".format(bmp_path), LogLevel.INFO)
    sock.send("DONE SENDING FILE".encode())
    f.close()
    
   
# close the connection
sock.close()    
     