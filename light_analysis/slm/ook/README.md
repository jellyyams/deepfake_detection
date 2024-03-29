# bbb_structured_light


This is a modified version of the structured light app provided in the official beaglebone black DLPDLCR2000 cape. The primary difference between this version and the included version is that here you can specify an input directory containing the 
bitmap files to display, rather than having to put all the bitmaps to display in the current directory. The below text is from the README in the original version.

This is an application that enables the BeagleBone Black to display a series of images on the TI DLPDLCR2000EVM. It is designed to enable structured light applications such as 3D scanning. It aims to provide an accurate display of images on the LCR2000, display the images at user configurable framerates, and provide input and output trigger for camera synchronization. It will display .bmp files that reside in the current program directory in alphabetical order by file name. Users are encouraged to adapt the code to their own use while following the license at the top of the source code files.

### to run
``` bassh
make # first time you run the program
./pattern_disp [options] framerate repetitions 
```

### program structure
 - display_app: handles command line arguments and the main display loop that loads images and synchronizes triggers
 - display_core: core functionality such as setting up triggers, clearing the screen, running test patterns, etc.
 - open_bmp: code related to opening and reading the user provided bitmap images
