import socket # for socket
import sys
import os
import time

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print ("Socket successfully created")
except socket.error as err:
    print ("socket creation failed with error %s" %(err))
 
# default port for socket
port = 6000
 
try:
    host_ip = "192.168.6.2"
    print(host_ip)
except socket.gaierror:
 
    # this means could not resolve the host
    print ("there was an error resolving the host")
    sys.exit()
 
# connecting to the server
s.connect((host_ip, port))
 
print ("the socket has successfully connected")
print("sending test message to server")
s.send("hi from client".encode())
# receive data from the server and decoding to get the string.
print (s.recv(1024).decode())
time.sleep(1)
s.send("test.bmp".encode())

print(s.recv(1024).decode())


bmp_path = '../barkmitzvah.bmp'
file_size = os.path.getsize(bmp_path)
print("File Size is :", file_size, "bytes")
f = open(bmp_path,'rb')
print('Sending...')
l = f.read(1024)
while (l):
    s.send(l)
    l = f.read(1024)
f.close()
print("Done Sending")
# # open image
# myfile = open('../barkmitzvah.bmp', 'rb')
# bytes = myfile.read()
# size = len(bytes)

# # send image size to server
# size_msg = "SIZE %s" % size
# print(size)
# s.sendall(size_msg.encode())
# answer = s.recv(1024)
# print(answer)
# # send image to server
# if answer == b'GOT SIZE':
#     print("Server got the bitmap size!")
#     print("Sending {} bytes of bitmap data to server.".format(len(bytes)))
#     s.sendall(bytes)
#     bmp_recv_ack = s.recv(1024)
#     print(bmp_recv_ack)

# s.send("DONE".encode())

#     # check what server send
#     answer = sock.recv(4096)
#     print 'answer = %s' % answer

#     if answer == 'GOT IMAGE' :
#         sock.sendall("BYE BYE ")
#         print 'Image successfully send to server'

# myfile.close()


# close the connection
# s.close()    
     