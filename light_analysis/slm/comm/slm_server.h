
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <pthread.h>

struct run_server_args {
    pthread_mutex_t * bmp_folder_lock;
    int * num_frames;
    int * repeat;
    int * delay;
    char * bmp_dir_path;
};

enum LogLevel{DEBUG, INFO, WARNING, ERROR};
// int run_server(pthread_mutex_t *bmp_folder_lock, int * num_frames, int * repeat, int * delay);
void* run_server(void * input);
void server_log(char *message, enum LogLevel log_level);
void display_message(char *message);
int StartsWith(const char *a, const char *b);
int writefile(int sockfd, FILE *fp);
