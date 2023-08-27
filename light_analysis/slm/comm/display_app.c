#include "display_app.h"
#include "open_bmp.h"

int main(int argc, char** argv) {
	// Variable declarationss
	struct fb_fix_screeninfo fix_info;
	struct fb_var_screeninfo var_info;
	int fb, delay = 1000000, repeat = 1, framerate = 10, num_images = 0, i = 0, video_mode = 0, restart_x = 0;
	char ** image_names;
	long screensize;
	char * bmp_dir_path = "";
	uint8_t *fbp, *buffer;

	// Allocate image structure which will be used to store image names
	// increase num images from 100 to 255 -HS
	image_names = (char**)malloc(255 * sizeof(char*));
	for (i = 0; i < 255; i++) {
		image_names[i] = (char*)malloc(200 * sizeof(char));
	}
	
	// Handle command line arguments
	if (argc <= 1) {
		printf("Usage: pattern_disp [options] framerate repetitions\n\n");
		printf("Options:\n");
		printf(" -framerate(-f)  SLM refresh rate");
		printf(" -repeat(-r) Number of timesto repeat display of bmps in folder\n");
		printf(" -bmpdir (-b)  Path to directory containing bmps. Default is curr directory\n");
		return EXIT_FAILURE;
	}
	
	
	i = 1;
	// Handle flags set from command line arguments
	while (i < argc && argv[i][0] == '-') { // while there are flags to handle
		if ((strcmp("-r",argv[i]) == 0) || (strcmp("-rep",argv[i]) == 0)) {
			repeat = (int)strtol(argv[i+1],NULL,10);
			i++; // increment additional time to get to next option
		}

		if ((strcmp("-b",argv[i]) == 0) || (strcmp("-bmpdir",argv[i]) == 0)) {
			bmp_dir_path = argv[i+1];
			printf("Using bmps from ");
			printf(bmp_dir_path);
			i++; // increment additional time to get to next option
		}

		if ((strcmp("-f",argv[i]) == 0) || (strcmp("-framerate",argv[i]) == 0)) {
			printf("Got framerate");
			framerate = (int)strtol(argv[i+1],NULL,10);
		}

		i++;
	}


	// Setup framebuffer 
	if (setup_fb(&fix_info, &var_info, &fb, &screensize, &fbp, &buffer, video_mode) == EXIT_FAILURE) {
		printf("Unable to setup framebuffer\n");
		return EXIT_FAILURE;
	}
	
	// Display images
	if(load_image_files(&num_images, image_names, bmp_dir_path) == EXIT_FAILURE) {
		return EXIT_FAILURE;
	}
	qsort(image_names, num_images, sizeof(image_names[0]), compare); // load files in alphabetical order

	display_images(image_names, num_images, fbp, buffer, &var_info, &fix_info, delay, repeat, screensize); // Display loaded images


	// Cleanup open files
	if (cleanup(fb, fbp, buffer, screensize, restart_x, video_mode, image_names) == EXIT_FAILURE){
		printf("Error cleaning up files\n");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}


int display_images(char **image_names, int num_images, uint8_t* fbp, uint8_t* bbp, struct fb_var_screeninfo* var_info, struct fb_fix_screeninfo* fix_info, int delay, int repeat, long screensize) {
	// Variable declarations
	int i, ii, usecs = 0;
	long x, y, location;
	long x_max = var_info->xres_virtual;
	long y_max = var_info->yres_virtual;
	uint32_t pix = 0x123456;// Pixel to draw
	pixel** img;
	struct timeval start, stop, start2, stop2;
	double totalusecs = 0;
	
	// Initial checks
	if (num_images <= 0) {
		printf("No images to display\n");
		return EXIT_FAILURE;
	}

	// Allocate image structure which will be used to load images
	img = (pixel**)malloc(IMG_Y * sizeof(pixel*));
	for (i = 0; i < IMG_Y; i++) {
		img[i] = (pixel*)malloc(IMG_X * sizeof(pixel));
	}

	// Will loop through displaying all images
	for (ii = 0; ii < repeat; ii++) {
		for (i = 0; i < num_images; i++) {
			// Open image and ensure it's successful. Inefficent to load file everytime but fine at BeagleBone's low effective video framerates
			if (open_bmp(*(image_names+i), img) == EXIT_FAILURE) {
				return EXIT_FAILURE;
			}
	
			// Transfer image structure to the buffer
			for (y=0; y<y_max; y++) {
				for (x=0; x<x_max; x++) {
					location = (x+var_info->xoffset) * (var_info->bits_per_pixel / 8) + (y + var_info->yoffset) * fix_info->line_length; // offset where we write pixel value
					pix = pixel_color(img[y][x].r, img[y][x].g, img[y][x].b, var_info); // get pixel in correct format
					*((uint32_t*)(bbp + location)) = pix; // write pixel to buffer	
				}
			}

			// Wait until delay is over
			if (!(ii == 0 && i == 0)) { // as long as it's not the first time through the loop we have to wait
				do {
					usleep(10);
					gettimeofday(&stop, NULL);
					usecs = (stop.tv_usec - start.tv_usec) + (stop.tv_sec - start.tv_sec)*1000000;

				
				} while (usecs < (delay-EXTRA_TIME)); // -EXTRA_TIME which is approximate buffer load time 
				
				if (DEBUG_TIME) {
					printf("Delay goal is %ius versus actual of %ius. Difference: %.1fms\n",delay,usecs,(usecs-delay)/1000.0);
					totalusecs+=usecs;
				}
			}
			

			// Freeze update buffer of DLP2000. This is so it won't display garbage data as we update the Beagles framebuffer
			system("i2cset -y 2 0x1b 0xa3 0x00 0x00 0x00 0x01 i");

			// Display image
			memcpy(fbp, bbp, screensize); // load framebuffer from buffered location

			// Start timer that will be used for next image
			gettimeofday(&start, NULL);

			usleep(delay/3); // allow framebuffer to finish loading
			system("i2cset -y 2 0x1b 0xa3 0x00 0x00 0x00 0x00 i"); // Unfreeze update buffer of DLP2000
			usleep(delay/10); // allow DLP2000 to update
		}
	}

	// Wait for last image to be done
	do {
		usleep(10);
		gettimeofday(&stop, NULL);
		usecs = (stop.tv_usec - start.tv_usec) + (stop.tv_sec - start.tv_sec)*1000000;
	} while (usecs < (delay-EXTRA_TIME)); // -EXTRA_TIME which is approximate buffer load time 


	if (DEBUG_TIME) {
		printf("Delay goal is %ius versus actual of %ius. Difference: %.1fms\n",delay,usecs,(usecs-delay)/1000.0);
		totalusecs+=usecs;
		printf("Average difference: %.1fms\n\n", (delay-totalusecs/repeat/num_images)/1000.0);
	}

	// Cleanup image memory
	for (i = 0; i < IMG_Y; i++) {
		free(img[i]);
	}
	free(img);


	clear_screen(fbp, bbp, var_info, fix_info, screensize);

	return EXIT_SUCCESS;
}


int compare (const void * a, const void * b)
{
	char * b_str = *(char **) b;
	char * a_str = *(char **) a;
	char * a_str_trunc;
	char * b_str_trunc;
	a_str_trunc  = strrchr(a_str, '/');
	b_str_trunc  = strrchr(b_str, '/');
	return strcmp(a_str_trunc, b_str_trunc);
}