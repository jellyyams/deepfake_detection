#include "slm_server.h"

#define PORT 6000
#define BUFFSIZE 4096
#define MAX_LINE 4096
#define LOG_LEVEL DEBUG

void server_log(char *message, enum LogLevel log_level){
    if (log_level >= LOG_LEVEL) {
        if (log_level == INFO){
            printf("SERVER [INFO]: %s\n", message);
        } else if (log_level == DEBUG)
        {
            printf("SERVER [DEBUG]: %s\n", message);
        } else if (log_level == WARNING)
        {
            printf("SERVER [WARNING]: %s\n", message);
        } else {
        printf("SERVER [ERROR]: %s\n", message); 
        }  
    }
    
}

void display_message(char *message){
    printf("SERVER [MESSAGE]: %s\n", message);
}

int StartsWith(const char *a, const char *b){
    //https://stackoverflow.com/questions/15515088/how-to-check-if-string-starts-with-certain-string-in-c
    if(strncmp(a, b, strlen(b)) == 0) return 1;
    return 0;
}




void update_metadata(char *metadata_str, int * num_frames, int * repeat, int * delay){
    char * rest_metadata = metadata_str;
    char * token = strtok_r(metadata_str, " ", &rest_metadata);
    int i = 0;
    for (i; i < 3; i++){
        token = strtok_r(NULL, " ",  &rest_metadata);
        char* subtoken = strtok(token, ":");
        if (strcmp(subtoken, "FREQ") == 0){
            char * freq_str = strtok(NULL, ":");
            // printf("freq: %s\n", freq_str);
            float freq = (float)strtof(freq_str, NULL);
            *delay = (int)(((1/(freq*2)))*1000000);
            // printf("delay updated to %d\n", *delay);
        }else if (strcmp(subtoken, "REP") == 0){
            char * rep_str = strtok(NULL, ":");
            // printf("rep: %s\n", rep_str);
            *repeat = (int)strtol(rep_str, NULL, 10);
        }else if (strcmp(subtoken, "NUMFRAMES") == 0){
            char * nf_str = strtok(NULL, ":");
            // printf("num frames: %s\n", nf_str);
            *num_frames = (int)strtol(nf_str, NULL, 10);
        }
    }
}

//https://www.geeksforgeeks.org/sock-programming-cc/
// int run_server(pthread_mutex_t *bmp_folder_lock, int * num_frames, int * repeat, int * delay){
void* run_server(void* input){
    //unpack args
    struct run_server_args * args = (struct run_server_args *)input;
    pthread_mutex_t * bmp_folder_lock = args->bmp_folder_lock;
    int * num_frames = args->num_frames;
    int * repeat = args->repeat;
    int * delay = args->delay;
    char * bmp_dir_path = args->bmp_dir_path;
  
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
    
    while (1){
        //try to receive folder metadata/display info 
        char metadata[BUFFSIZE] = {0};
        int mm = 0;
        server_log("Waiting for metadata from client.", INFO);
        while (mm <= 0){
            mm = recv(sock, metadata, BUFFSIZE, 0);
            if (mm == -1) 
            {
                perror("Error while receiving metadata.");
                exit(1);
            } 
        }
        update_metadata(metadata, num_frames, repeat, delay);

      
        send(sock, "send stuff", strlen("send stuff"), 0);
        int i = 0;
        pthread_mutex_lock(bmp_folder_lock);
        for (i; i < *num_frames; i++){
            char message[100];
            sprintf(message, "Server ready to receive frame %d from client.", i);
            server_log(message, INFO);
            send(sock, message, strlen(message), 0);

            char framename[20];
            sprintf(framename, "%s/frame%d.bmp", bmp_dir_path, i);
            FILE *fp = fopen(framename, "wb");
            if (fp == NULL) 
            {
                perror("Can't open file");
                exit(1);
            }
            
            char addr[INET_ADDRSTRLEN];
            struct sockaddr_in clientaddr;
            sprintf(message, "Receiving frame %d from %s...", i, inet_ntop(AF_INET, &clientaddr.sin_addr, addr, INET_ADDRSTRLEN));
            server_log(message, INFO);
            
            int num_filebytes = writefile(sock, fp);
            sprintf(message, "File receive success. NumBytes = %d", num_filebytes);
            server_log(message, INFO);
            fclose(fp);
        }
        pthread_mutex_lock(bmp_folder_lock);
    }
    
    close(sock);
    return 0;

}


int writefile(int sockfd, FILE *fp)
{
    ssize_t n = 1; 
    int num_filebytes = 0;
    char buff[MAX_LINE] = {0};
    while (n > 0) 
    {
        n = recv(sockfd, buff, MAX_LINE, 0);
        if (strcmp(buff, "DONE SENDING FILE") == 0){break;}
        if (n == -1)
        {
            perror("Receive File Error");
            server_log("Error receiving file.", ERROR);
            exit(1);
        }
        
        if (fwrite(buff, sizeof(char), n, fp) != n)
        {
            perror("Write File Error");
            server_log("Error writing file.", ERROR);
            exit(1);
        }
        num_filebytes += n;
        memset(buff, 0, MAX_LINE);
        send(sockfd, "next chunk pls", strlen("next chunk pls"), 0);
    }
    return num_filebytes;
}