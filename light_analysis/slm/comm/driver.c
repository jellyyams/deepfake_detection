#include "slm_server.h"
#include "display_app.h"
#include <pthread.h>

int main(int argc, char** argv) {
    char * bmp_dir_path = "test";
	int num_frames = 2; //default
	float frequency = 1; //default
	int delay = (int)(((1/(frequency*2)))*1000000); 
	int repeat = 5; //default

    pthread_mutex_t bmp_folder_lock;
    if (pthread_mutex_init(&bmp_folder_lock, NULL) != 0) {
        printf("Mutex init failed\n");
        return 1;
    }
    pthread_t tid;
    
    struct run_server_args * server_args = (struct run_server_args *)malloc(sizeof(struct run_server_args));
    server_args->bmp_folder_lock = &bmp_folder_lock;
    server_args->num_frames = &num_frames;
    server_args->repeat = &repeat;
    server_args->delay = &delay;
    server_args->bmp_dir_path = bmp_dir_path;

    int error = pthread_create(&(tid), NULL, &run_server, (void*)server_args);
    if (error != 0)
        printf("\nThread can't be created :[%s]", strerror(error));
    pthread_join(tid, NULL);

    //run_server(&bmp_folder_lock, &num_frames, &repeat, &delay);
    // run_display(&num_frames, &repeat, &delay, bmp_dir_path);
}