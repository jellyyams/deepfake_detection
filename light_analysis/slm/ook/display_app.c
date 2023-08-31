#include "display_app.h"
#include "open_bmp.h"
#include <sys/time.h>
#include <time.h>
#include <math.h>


int python_main_handler(char * on_img_path, char * off_img_path, char * log_path, int on, int off, int repeat) {
	setbuf(stdout, NULL); //otherwise sometimes won't get any output on screen

	// Variable declarationss
	struct fb_fix_screeninfo fix_info;
	struct fb_var_screeninfo var_info;
	int fb, i = 0;
	long screensize;
	
	uint8_t *fbp, *img_buffer, *blank_buffer;

	// Setup framebuffer
	if (setup_fb(&fix_info, &var_info, &fb, &screensize, &fbp, &img_buffer, &blank_buffer) == EXIT_FAILURE) {
		printf("Unable to setup framebuffer\n");
		return EXIT_FAILURE;
	}

	//start displaying 
	display(on_img_path, off_img_path, log_path, fbp, img_buffer, blank_buffer, &var_info, &fix_info, on, off, repeat, screensize);
}

int display(char* on_img_path, char* off_img_path, char* log_path, uint8_t* fbp, uint8_t* on_img_bbp, uint8_t* off_img_bbp, struct fb_var_screeninfo* var_info, struct fb_fix_screeninfo* fix_info, int on, int off, int repeat, long screensize) {

	// Variable declarations
	int i, ii;
	long x, y, location;
	long x_max = var_info->xres_virtual;
	long y_max = var_info->yres_virtual;
	uint32_t pix = 0x123456;// Pixel to draw
	pixel** img;
	double totalusecs = 0;

	// initialize log file pointer
	FILE *fp;
	fp = fopen(log_path, "w+");
	
	// Allocate image structure which will be used to load images
	img = (pixel**)malloc(IMG_Y * sizeof(pixel*));
	for (i = 0; i < IMG_Y; i++) {
		img[i] = (pixel*)malloc(IMG_X * sizeof(pixel));
	}

	// Open image and ensure it's successful. Inefficent to load file everytime but fine at BeagleBone's low effective video framerates
	if (open_bmp(on_img_path, img) == EXIT_FAILURE) {
		return EXIT_FAILURE;
	}
	
	// Transfer image structure to the buffer
	for (y=0; y<y_max; y++) {
		for (x=0; x<x_max; x++) {
			location = (x+var_info->xoffset) * (var_info->bits_per_pixel / 8) + (y + var_info->yoffset) * fix_info->line_length; // offset where we write pixel value
			pix = pixel_color(img[y][x].r, img[y][x].g, img[y][x].b, var_info); // get pixel in correct format
			*((uint32_t*)(on_img_bbp + location)) = pix; // write pixel to buffer	
		}
	}

	// Open image and ensure it's successful. Inefficent to load file everytime but fine at BeagleBone's low effective video framerates
	if (open_bmp(off_img_path, img) == EXIT_FAILURE) {
		return EXIT_FAILURE;
	}

	// Transfer blank image structure to the buffer
	for (y=0; y<y_max; y++) {
		for (x=0; x<x_max; x++) {
			location = (x+var_info->xoffset) * (var_info->bits_per_pixel / 8) + (y + var_info->yoffset) * fix_info->line_length; // offset where we write pixel value
			pix = pixel_color(img[y][x].r, img[y][x].g, img[y][x].b, var_info); // get pixel in correct format
			*((uint32_t*)(off_img_bbp + location)) = pix; // write pixel to buffer	
		}
	}

	// Flash image at interavl specified by on/off
	if (repeat > 0) {
		for (ii = 0; ii < repeat; ii++) {
			if (ii == 0){
				flash_on_off(fbp, on_img_bbp, off_img_bbp, fp, var_info, fix_info, on, off, screensize, 1);
			} else {
				flash_on_off(fbp, on_img_bbp, off_img_bbp, fp, var_info, fix_info, on, off, screensize, 0);
			}
		}
	} else {
		int first = 0;
		while (1) {
			if (first == 0){
				flash_on_off(fbp, on_img_bbp, off_img_bbp, fp, var_info, fix_info, on, off, screensize, 1);
				first = 1;
			} else {
				flash_on_off(fbp, on_img_bbp, off_img_bbp, fp, var_info, fix_info, on, off, screensize, 0);
			}
		}
	}
	printf("Finished display. Cleaning up.");

	// Cleanup image memory
	for (i = 0; i < IMG_Y; i++) {
		free(img[i]);
	}
	free(img);

	//close log file
	fclose(fp);

	// clear_screen(fbp, on_img_bbp, var_info, fix_info, screensize);

	return EXIT_SUCCESS;
}

int flash_on_off(uint8_t* fbp, uint8_t* on_img_bbp,  uint8_t* off_img_bbp, FILE* fp, struct fb_var_screeninfo* var_info, struct fb_fix_screeninfo* fix_info, int on, int off, long screensize, int first_disp) {

	int usecs;
	struct timeval start, stop;

	// Wait until delay is over
	if (first_disp == 0) { // as long as it's not the first time through the loop we have to wait
		do {
			usleep(10);
			gettimeofday(&stop, NULL);
			usecs = (stop.tv_usec - start.tv_usec) + (stop.tv_sec - start.tv_sec)*1000000;	
		} while (usecs < (off-EXTRA_TIME)); // -EXTRA_TIME which is approximate buffer load time 
		
		// if (DEBUG_TIME) {
		// 	printf("Off time goal is %ius versus actual of %ius. Difference: %.1fms\n",off,usecs,(usecs-off)/1000.0);
		// 	totalusecs+=usecs;
		// }
	}
	
	// Freeze update buffer of DLP2000. This is so it won't display garbage data as we update the Beagles framebuffer
	system("i2cset -y 2 0x1b 0xa3 0x00 0x00 0x00 0x01 i");

	// Display image
	// time_t t_on = time(NULL);
  	// struct tm tm_on = *localtime(&t_on);
	char buffer[26];
	int millisec;
	struct tm* tm_info;
	struct timeval tv;

	
	memcpy(fbp, on_img_bbp, screensize); // load framebuffer from buffered location
	// fprintf(fp, "On: %d-%02d-%02d %02d:%02d:%02d\n", tm_on.tm_year + 1900, tm_on.tm_mon + 1, tm_on.tm_mday, tm_on.tm_hour, tm_on.tm_min, tm_on.tm_sec);
	

	// Start timer that will be used for next image
	gettimeofday(&start, NULL);

	

	usleep(off/3); // allow framebuffer to finish loading
	system("i2cset -y 2 0x1b 0xa3 0x00 0x00 0x00 0x00 i"); // Unfreeze update buffer of DLP2000

	//write to log
	gettimeofday(&tv, NULL);
	millisec = lrint(tv.tv_usec/1000.0); // Round to nearest millisec
	if (millisec>=1000) { // Allow for rounding up to nearest second
		millisec -=1000;
		tv.tv_sec++;
	}

	tm_info = localtime(&tv.tv_sec);

	strftime(buffer, 26, "%Y:%m:%d %H:%M:%S", tm_info);
	fprintf(fp, "On: %s.%03d\n", buffer, millisec);


	usleep(off/10); // allow DLP2000 to update

	// BLANK DISPLAY
	// Wait until delay is over
	do {
		usleep(10);
		gettimeofday(&stop, NULL);
		usecs = (stop.tv_usec - start.tv_usec) + (stop.tv_sec - start.tv_sec)*1000000;
	} while (usecs < (on-EXTRA_TIME)); // -EXTRA_TIME which is approximate buffer load time 
	
	// if (DEBUG_TIME) {
	// 	printf("On goal is %ius versus actual of %ius. Difference: %.1fms\n",on,usecs,(usecs-on)/1000.0);
	// 	totalusecs+=usecs;
	// }
	

	// Freeze update buffer of DLP2000. This is so it won't display garbage data as we update the Beagles framebuffer
	system("i2cset -y 2 0x1b 0xa3 0x00 0x00 0x00 0x01 i");

	// Display blank image
	// time_t t_off = time(NULL);
  	// struct tm tm_off = *localtime(&t_off);
	char buffer_off[26];
	int millisec_off;
	struct tm* tm_info_off;
	struct timeval tv_off;

	
	memcpy(fbp, off_img_bbp, screensize); // load framebuffer from buffered location
	// fprintf(fp, "Off: %d-%02d-%02d %02d:%02d:%02d\n", tm_off.tm_year + 1900, tm_off.tm_mon + 1, tm_off.tm_mday, tm_off.tm_hour, tm_off.tm_min, tm_off.tm_sec);
	

	// Start timer that will be used for next image
	gettimeofday(&start, NULL);

	usleep(on/3); // allow framebuffer to finish loading
	system("i2cset -y 2 0x1b 0xa3 0x00 0x00 0x00 0x00 i"); // Unfreeze update buffer of DLP2000

	//write to log
	gettimeofday(&tv_off, NULL);
	millisec_off = lrint(tv_off.tv_usec/1000.0); // Round to nearest millisec
	if (millisec_off>=1000) { // Allow for rounding up to nearest second
		millisec_off -=1000;
		tv_off.tv_sec++;
	}

	tm_info_off = localtime(&tv_off.tv_sec);

	strftime(buffer_off, 26, "%Y:%m:%d %H:%M:%S", tm_info_off);
	fprintf(fp, "Off: %s.%03d\n", buffer_off, millisec_off);

	usleep(on/10); // allow DLP2000 to update
}

int compar (const void * a, const void * b)
{
    return strcmp(*(char **) b, *(char **) a);
}