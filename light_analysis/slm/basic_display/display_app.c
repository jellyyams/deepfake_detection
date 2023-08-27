/*
 DLPDLCR2000EVM Example Test Script Suite
 Implements basic structured light functionality of  the DLP LightCrafter
 Display 2000 EVM using the provided code

 Copyright (C) 2018 Texas Instruments Incorporated - http://www.ti.com/


  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

    Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the  
    distribution.

    Neither the name of Texas Instruments Incorporated nor the names of
    its contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "display_app.h"
#include "open_bmp.h"

int main(int argc, char** argv) {
	// Variable declarationss
	struct fb_fix_screeninfo fix_info;
	struct fb_var_screeninfo var_info;
	int fb, delay = 1000000, repeat = 1, num_images = 0, num_frames = 0, i = 0, video_mode = 0, restart_x = 0;
	float frequency = 10;
	char ** image_names;
	long screensize;
	char * bmp_dir_path = "";
	uint8_t *fbp, **buffers;

	// Allocate image structure which will be used to store image names
	image_names = (char**)malloc(100 * sizeof(char*));
	for (i = 0; i < 100; i++) {
		image_names[i] = (char*)malloc(200 * sizeof(char));
	}
	
	// Handle command line arguments
	if (argc < 5) {
		printf("Usage: ./display [bmpdir] [num_images] [frequency] [repeat]\n");
		return EXIT_FAILURE;
	}
	bmp_dir_path = argv[1];
	num_frames = (int)strtol(argv[2],NULL, 10);
	printf("Num images is %d\n", num_images);
	frequency = (float)strtof(argv[3],NULL);
	printf("Frequency is %f\n", frequency);
	delay = (int)(((1/(frequency*2)))*1000000);
	printf("Delay is %d\n", delay);
	repeat = (int)strtol(argv[4],NULL,10);
	printf("Repeat is %d\n", repeat);
	
	// Setup framebuffer 
	if (setup_fb(&fix_info, &var_info, &fb, &screensize, &fbp, video_mode) == EXIT_FAILURE) {
		printf("Unable to setup framebuffer\n");
		return EXIT_FAILURE;
	}

	// Malloc bitmap buffers
	buffers = (uint8_t**)malloc(num_frames * sizeof(uint8_t*));
	for (i = 0; i < num_frames; i++) {
		buffers[i] = (uint8_t*)malloc(screensize* sizeof(uint8_t));
		setup_buffer(&fix_info, &var_info, screensize, num_frames, &buffers[i]);
	}

	if(load_image_files(&num_images, image_names, bmp_dir_path) == EXIT_FAILURE) {
		return EXIT_FAILURE;
	}
	qsort(image_names, num_images, sizeof(image_names[0]), compare); // load files in alphabetical order

	buffer_images(image_names, num_frames, buffers, &var_info, &fix_info);
	
	display_images(image_names, num_images, fbp, buffers, &var_info, &fix_info, delay, repeat, screensize); // Display loaded images

	// // Cleanup open files
	// if (cleanup(fb, fbp, buffer, screensize, restart_x, video_mode, image_names) == EXIT_FAILURE){
	// 	printf("Error cleaning up files\n");
	// 	return EXIT_FAILURE;
	// }

	// return EXIT_SUCCESS;
}


int buffer_images(char **image_names, int num_frames, uint8_t** buffers, struct fb_var_screeninfo* var_info, struct fb_fix_screeninfo* fix_info) {
	// Variable declarations
	int i = 0;
	long x, y, location;
	long x_max = var_info->xres_virtual;
	long y_max = var_info->yres_virtual;
	uint32_t pix = 0x123456;// Pixel to draw
	pixel** img;
	
	// Allocate image structure which will be used to load images
	img = (pixel**)malloc(IMG_Y * sizeof(pixel*));
	for (i = 0; i < IMG_Y; i++) {
		img[i] = (pixel*)malloc(IMG_X * sizeof(pixel));
	}

	for (i = 0; i < num_frames; i++) {
			// Open image and ensure it's successful. Inefficent to load file everytime but fine at BeagleBone's low effective video framerates
			if (open_bmp(*(image_names+i), img) == EXIT_FAILURE) {
				return EXIT_FAILURE;
			}
	
			// Transfer image structure to the buffer
			for (y=0; y<y_max; y++) {
				for (x=0; x<x_max; x++) {
					location = (x+var_info->xoffset) * (var_info->bits_per_pixel / 8) + (y + var_info->yoffset) * fix_info->line_length; // offset where we write pixel value
					pix = pixel_color(img[y][x].r, img[y][x].g, img[y][x].b, var_info); // get pixel in correct format
					*((uint32_t*)(buffers[i] + location)) = pix; // write pixel to buffer	
				}
			}
	}
}

int display_images(char **image_names, int num_images, uint8_t* fbp, uint8_t** buffers, struct fb_var_screeninfo* var_info, struct fb_fix_screeninfo* fix_info, int delay, int repeat, long screensize) {
	// Variable declarations
	int i, ii, usecs = 0;
	long x, y, location;
	long x_max = var_info->xres_virtual;
	long y_max = var_info->yres_virtual;
	uint32_t pix = 0x123456;// Pixel to draw
	pixel** img;
	struct timeval start, stop;
	double totalusecs = 0;
	
	// Initial checks
	if (num_images <= 0) {
		printf("No images to display\n");
		return EXIT_FAILURE;
	}


	// Allocate image structure which will be used to load images
	// img = (pixel**)malloc(IMG_Y * sizeof(pixel*));
	// for (i = 0; i < IMG_Y; i++) {
	// 	img[i] = (pixel*)malloc(IMG_X * sizeof(pixel));
	// }


	// Will loop through displaying all images
	for (ii = 0; ii < repeat; ii++) {
		for (i = 0; i < num_images; i++) {
			// // Open image and ensure it's successful. Inefficent to load file everytime but fine at BeagleBone's low effective video framerates
			// if (open_bmp(*(image_names+i), img) == EXIT_FAILURE) {
			// 	return EXIT_FAILURE;
			// }
	
			// // Transfer image structure to the buffer
			// for (y=0; y<y_max; y++) {
			// 	for (x=0; x<x_max; x++) {
			// 		location = (x+var_info->xoffset) * (var_info->bits_per_pixel / 8) + (y + var_info->yoffset) * fix_info->line_length; // offset where we write pixel value
			// 		pix = pixel_color(img[y][x].r, img[y][x].g, img[y][x].b, var_info); // get pixel in correct format
			// 		*((uint32_t*)(bbp + location)) = pix; // write pixel to buffer	
			// 	}
			// }
			
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
			memcpy(fbp, buffers[i], screensize); // load framebuffer from buffered location

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

	// // Cleanup image memory
	// for (i = 0; i < IMG_Y; i++) {
	// 	free(img[i]);
	// }
	// free(img);

	// clear_screen(fbp, bbp, var_info, fix_info, screensize);

	// return EXIT_SUCCESS;
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