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

enum LogLevel{INFO, DEBUG, WARNING, ERROR};

#define LOG_LEVEL LogLevel.DEBUG

void server_log(char *message, enum LogLevel log_level){
    if (log_level >= LOG_LEVEL) {
        if (log_level == INFO){
            printf("SERVER INFO: %s\n", message);
        } else if (log_level == DEBUG)
        {
            printf("SERVER DEBUG: %s\n", message);
        } else if (log_level == WARNING)
        {
            printf("SERVER WARNING: %s\n", message);
        } else {
        printf("SERVER ERROR: %s\n", message); 
        }  
    }
    
}

void display_message(char *message){
    printf("Server got following message from client: %s\n", message);
}

int StartsWith(const char *a, const char *b)
//https://stackoverflow.com/questions/15515088/how-to-check-if-string-starts-with-certain-string-in-c
{
   if(strncmp(a, b, strlen(b)) == 0) return 1;
   return 0;
}

void writefile(int sockfd, FILE *fp);
ssize_t total=0;

//https://www.geeksforgeeks.org/sock-programming-cc/
int main(int argc, char const* argv[])
{
    int server_fd, sock, valread;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    char buffer[1000000] = { 0 };
    char* hello = "Ping client from SLM server.";
  
    // Creating sock file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("sock failed");
        exit(EXIT_FAILURE);
    }
  
    // Forcefully attaching sock to the port 8080
    if (setsockopt(server_fd, SOL_SOCKET,
                   SO_REUSEADDR | SO_REUSEPORT, &opt,
                   sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);
  
    // Forcefully attaching sock to the port 8080
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
    if ((sock
         = accept(server_fd, (struct sockaddr*)&address,
                  (socklen_t*)&addrlen))
        < 0) {
        perror("accept");
        exit(EXIT_FAILURE);
    }
    valread = read(sock, buffer, 1024);
    display_message(buffer);
    
    send(sock, hello, strlen(hello), 0);
    server_log("Sent hellos message to client", DEBUG);
  
    //try to receive file
    char filename[BUFFSIZE] = {0}; 
    if (recv(sock, filename, BUFFSIZE, 0) == -1) 
    {
        perror("Can't receive filename");
        exit(1);
    } else {
        char message[100];
        sprintf(message, "Server ready to receive file %s from client.", filename);
        server_log(message, INFO);
    }
    send(sock, "Ready to receive." ,strlen("Ready to receive"), 0);

    FILE *fp = fopen(filename, "wb");
    if (fp == NULL) 
    {
        perror("Can't open file");
        exit(1);
    }
    
    char addr[INET_ADDRSTRLEN];
    struct sockaddr_in clientaddr;
    char message[100];
    sprintf(message, "Receiving file: %s from %s...\n", filename, inet_ntop(AF_INET, &clientaddr.sin_addr, addr, INET_ADDRSTRLEN));
    server_log(message, INFO);
    writefile(sock, fp);
    sprintf(message, "File receive success. NumBytes = %ld\n", total);
    server_log(message, INFO);
    

    // fclose(fp);
    // close(sock);
    // return 0;
    
    // while (1) {
    //     valread = read(sock, buffer, 1000000);
    //     printf("%d\n", valread);
    //     //receive bitmap data in loop until client communicates it's finished sending
    //     // printf("hi");
    //     if (StartsWith(buffer, "SIZE")){
    //         char *size_str = strtok(buffer, " ");
    //         size_str = strtok(NULL, " ");
    //         int size = atoi(size_str);
    //         printf("Size is %d\n", size);
    //         send(sock, "GOT SIZE", strlen("GOT SIZE"), 0);
    //     } else if (StartsWith(buffer, "DONE")) {
    //         char *bye = "Server got termination message. Bye!";
    //         send(sock, bye, strlen(bye), 0);
    //         // // closing the connected sock
    //         // close(sock);
    //         // // closing the listening sock
    //         // shutdown(server_fd, SHUT_RDWR);
    //         // return 0;
    //     } else if (valread > 0){
    //         printf("Got some bitmap data");
    //         char *recv_ack = "Server got bitmap data";
    //         send(sock, recv_ack, strlen(recv_ack), 0);
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