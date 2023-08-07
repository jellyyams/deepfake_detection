#include "display_app.h"
#include "open_bmp.h"
#include <sys/time.h>
#include <time.h>
#include <math.h>

int main(int argc, char** argv) {
	setbuf(stdout, NULL); //otherwise sometimes won't get any output on screen

	// Variable declarationss
	struct fb_fix_screeninfo fix_info;
	struct fb_var_screeninfo var_info;
	int fb, repeat = 1, i = 0;
	long screensize;
	char * img_path;
	char * log_path;
	int on = 1000000, off = 1000000;
	uint8_t *fbp, *img_buffer, *blank_buffer;
	
	// Handle command line arguments
	if (argc <= 1) {
		printf("Usage: pattern_disp [options] \n");
		printf("Options:\n");
		printf(" -impath (-i) path to image to flash\n");
		printf(" -logpath (-l) path to log with flash timestamps\n");
		printf(" -on (-n) time to keep image on (ms) \n");
		printf(" -off (-f) time to keep image off (ms)\n");
		printf(" -repeat (-r) number of times to flash image. Default is infinite. \n");
		return EXIT_FAILURE;
	}
	
	
	i = 1;
	// Handle flags set from command line arguments
	while (i < argc && argv[i][0] == '-') { // while there are flags to handle
		
		if ((strcmp("-i",argv[i]) == 0) || (strcmp("-impath",argv[i]) == 0)) {
			img_path = argv[i+1];
			i++; // increment additional time to account for input following the flag
		}
		else if ((strcmp("-l",argv[i]) == 0) || (strcmp("-logpath",argv[i]) == 0)) {
			log_path = argv[i+1];
			i++; // increment additional time to account for input following the flag
		}
		else if ((strcmp("-n",argv[i]) == 0) || (strcmp("-on",argv[i]) == 0)) {
			on =  (int)strtol(argv[i+1],NULL,10) * 1000; 
			i++; // increment additional time to account for input following the flag
		}
		else if ((strcmp("-f",argv[i]) == 0) || (strcmp("-off",argv[i]) == 0)) {
			off =  (int)strtol(argv[i+1],NULL,10) * 1000; 
			i++; // increment additional time to account for input following the flag
		}
		else if ((strcmp("-r",argv[i]) == 0) || (strcmp("-repeat",argv[i]) == 0)) {
			repeat =  (int)strtol(argv[i+1],NULL,10);
			i++; // increment additional time to account for input following the flag
		}

		i++;
	}

	// Setup framebuffer
	if (setup_fb(&fix_info, &var_info, &fb, &screensize, &fbp, &img_buffer, &blank_buffer) == EXIT_FAILURE) {
		printf("Unable to setup framebuffer\n");
		return EXIT_FAILURE;
	}

	//start displaying 
	display(img_path, log_path, fbp, img_buffer, blank_buffer, &var_info, &fix_info, on, off, repeat, screensize);
}

int python_main_handler(char * img_path, char * log_path, int on, int off, int repeat) {
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
	display(img_path, log_path, fbp, img_buffer, blank_buffer, &var_info, &fix_info, on, off, repeat, screensize);
}

int display(char* img_path, char* log_path, uint8_t* fbp, uint8_t* img_bbp, uint8_t* blank_bbp, struct fb_var_screeninfo* var_info, struct fb_fix_screeninfo* fix_info, int on, int off, int repeat, long screensize) {

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
	if (open_bmp(img_path, img) == EXIT_FAILURE) {
		return EXIT_FAILURE;
	}
	
	// Transfer image structure to the buffer
	for (y=0; y<y_max; y++) {
		for (x=0; x<x_max; x++) {
			location = (x+var_info->xoffset) * (var_info->bits_per_pixel / 8) + (y + var_info->yoffset) * fix_info->line_length; // offset where we write pixel value
			pix = pixel_color(img[y][x].r, img[y][x].g, img[y][x].b, var_info); // get pixel in correct format
			*((uint32_t*)(img_bbp + location)) = pix; // write pixel to buffer	
		}
	}

	// Open image and ensure it's successful. Inefficent to load file everytime but fine at BeagleBone's low effective video framerates
	if (open_bmp("blank.bmp", img) == EXIT_FAILURE) {
		return EXIT_FAILURE;
	}

	// Transfer blank image structure to the buffer
	for (y=0; y<y_max; y++) {
		for (x=0; x<x_max; x++) {
			location = (x+var_info->xoffset) * (var_info->bits_per_pixel / 8) + (y + var_info->yoffset) * fix_info->line_length; // offset where we write pixel value
			pix = pixel_color(img[y][x].r, img[y][x].g, img[y][x].b, var_info); // get pixel in correct format
			*((uint32_t*)(blank_bbp + location)) = pix; // write pixel to buffer	
		}
	}

	// Flash image at interavl specified by on/off
	if (repeat > 0) {
		for (ii = 0; ii < repeat; ii++) {
			if (ii == 0){
				flash_image(fbp, img_bbp, blank_bbp, fp, var_info, fix_info, on, off, screensize, 1);
			} else {
				flash_image(fbp, img_bbp, blank_bbp, fp, var_info, fix_info, on, off, screensize, 0);
			}
		}
	} else {
		int first = 0;
		while (1) {
			if (first == 0){
				flash_image(fbp, img_bbp, blank_bbp, fp, var_info, fix_info, on, off, screensize, 1);
				first = 1;
			} else {
				flash_image(fbp, img_bbp, blank_bbp, fp, var_info, fix_info, on, off, screensize, 0);
			}
		}
	}
	

	// Cleanup image memory
	for (i = 0; i < IMG_Y; i++) {
		free(img[i]);
	}
	free(img);

	//close log file
	fclose(fp);

	// clear_screen(fbp, img_bbp, var_info, fix_info, screensize);

	return EXIT_SUCCESS;
}

int flash_image(uint8_t* fbp, uint8_t* img_bbp,  uint8_t* blank_bbp, FILE* fp, struct fb_var_screeninfo* var_info, struct fb_fix_screeninfo* fix_info, int on, int off, long screensize, int first_disp) {

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

	
	memcpy(fbp, img_bbp, screensize); // load framebuffer from buffered location
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

	
	memcpy(fbp, blank_bbp, screensize); // load framebuffer from buffered location
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