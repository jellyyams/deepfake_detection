#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#define PORT 6000
#define BUFFSIZE 4096
#define MAX_LINE 4096

int StartsWith(const char *a, const char *b)
//https://stackoverflow.com/questions/15515088/how-to-check-if-string-starts-with-certain-string-in-c
{
   if(strncmp(a, b, strlen(b)) == 0) return 1;
   return 0;
}

void writefile(int sockfd, FILE *fp);
ssize_t total=0;

//https://www.geeksforgeeks.org/socket-programming-cc/
int main(int argc, char const* argv[])
{
    int server_fd, new_socket, valread;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    char buffer[1000000] = { 0 };
    char* hello = "Hello from server";
  
    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }
  
    // Forcefully attaching socket to the port 8080
    if (setsockopt(server_fd, SOL_SOCKET,
                   SO_REUSEADDR | SO_REUSEPORT, &opt,
                   sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);
  
    // Forcefully attaching socket to the port 8080
    if (bind(server_fd, (struct sockaddr*)&address,
             sizeof(address))
        < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }
    if ((new_socket
         = accept(server_fd, (struct sockaddr*)&address,
                  (socklen_t*)&addrlen))
        < 0) {
        perror("accept");
        exit(EXIT_FAILURE);
    }
    valread = read(new_socket, buffer, 1024);
    printf("%s\n", buffer);
    send(new_socket, hello, strlen(hello), 0);
    printf("Hello message sent\n");
  
 
    //try to receive file
    char filename[BUFFSIZE] = {0}; 
    if (recv(new_socket, filename, BUFFSIZE, 0) == -1) 
    {
        perror("Can't receive filename");
        exit(1);
    } else {
        printf("Ready to receive file %s", filename);
    }
    send(new_socket, "send it over" ,strlen("send it over"), 0);

    FILE *fp = fopen(filename, "wb");
    if (fp == NULL) 
    {
        perror("Can't open file");
        exit(1);
    }
    
    char addr[INET_ADDRSTRLEN];
    struct sockaddr_in clientaddr;
    printf("Start receive file: %s from %s\n", filename, inet_ntop(AF_INET, &clientaddr.sin_addr, addr, INET_ADDRSTRLEN));
    writefile(new_socket, fp);
    printf("Receive Success, NumBytes = %ld\n", total);
    

    // fclose(fp);
    // close(new_socket);
    // return 0;
    
    // while (1) {
    //     valread = read(new_socket, buffer, 1000000);
    //     printf("%d\n", valread);
    //     //receive bitmap data in loop until client communicates it's finished sending
    //     // printf("hi");
    //     if (StartsWith(buffer, "SIZE")){
    //         char *size_str = strtok(buffer, " ");
    //         size_str = strtok(NULL, " ");
    //         int size = atoi(size_str);
    //         printf("Size is %d\n", size);
    //         send(new_socket, "GOT SIZE", strlen("GOT SIZE"), 0);
    //     } else if (StartsWith(buffer, "DONE")) {
    //         char *bye = "Server got termination message. Bye!";
    //         send(new_socket, bye, strlen(bye), 0);
    //         // // closing the connected socket
    //         // close(new_socket);
    //         // // closing the listening socket
    //         // shutdown(server_fd, SHUT_RDWR);
    //         // return 0;
    //     } else if (valread > 0){
    //         printf("Got some bitmap data");
    //         char *recv_ack = "Server got bitmap data";
    //         send(new_socket, recv_ack, strlen(recv_ack), 0);
    //     } 

    // }
}


void writefile(int sockfd, FILE *fp)
{
    ssize_t n;
    char buff[MAX_LINE] = {0};
    while ((n = recv(sockfd, buff, MAX_LINE, 0)) > 0) 
    {
	    total+=n;
        if (n == -1)
        {
            perror("Receive File Error");
            exit(1);
        }
        
        if (fwrite(buff, sizeof(char), n, fp) != n)
        {
            perror("Write File Error");
            exit(1);
        }
        memset(buff, 0, MAX_LINE);
    }
}