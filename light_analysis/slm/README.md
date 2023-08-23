Beagblone/SLM Code

basic_display: Out of the box beaglebone test script, enhanced to not just accept a bitmap path, but also accept a folder path and display all bitmaps in that folder. This code is all written in C (i.e., does not have a Python driver associated with it).

alternate_blank: Display a specified bitmap in alternation with nothing at a fixed frequency. This code is integrated with a python driver. 

ook: accept two bitmap paths - one to a bitmap assumed to be an OOK barcode in the pilot ON stage, and one to a bitmap assumed to be an OOK barcode in the pilot OFF stage. Alternate the ON and OFF images at specified frequency. This code is integrated with a python driver. 

comm: start of code for accepting bitmaps from TCP connection and updating display in real time as bmps arrive. 