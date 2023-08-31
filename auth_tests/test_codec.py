import cv2
import numpy as np
import struct




# #test recording of video
# import cv2
# # import skvideo.io
# import os
# import struct

def slice_vid(input_video_path):
    input_video_name = input_video_path.split('/')[-1][:-4]
    slice_w = 250
    slice_h = 100
    offset_x = 350
    offset_y = 970
    
    cap = cv2.VideoCapture(input_video_path)
    input_cap_fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(input_cap_fps)
    W , H = int(cap.get(3)), int(cap.get(4)) #input video dimensions
    out = cv2.VideoWriter('hi3.avi', cv2.VideoWriter_fourcc(*'RGBA'), input_cap_fps, (slice_w, slice_h))
    # out = cv2.VideoWriter()
    # out.open("foo.avi", -1, 25, (slice_w, slice_h), True)
    while(True):
        ret, frame = cap.read()
        if ret == True:
            out.write(frame[offset_y:offset_y + slice_h, offset_x:offset_x + slice_w, :])
            # out.write(frame)
        else:
            break

    out.release()
    cap.release()



slice_vid('../../led_tests_jun26/people/red_hadleigh2_low_2000mson_2000msoff.mp4')

# command = 'ffmpeg -i led_tests_jun26/people/red_hadleigh2_low_2000mson_2000msoff.mov -pix_fmt yuv420p -c:v libx265 -crf 23 -preset fast -c:a aac -b:a 128k output.mp4'
# os.system(command)

# capture=cv2.VideoCapture('led_tests_jun26/people/red_hadleigh2_low_2000mson_2000msoff.mp4') #open video file
# outputfile = "test_codec.mp4"   #our output filename
# writer = skvideo.io.FFmpegWriter(outputfile, outputdict={
#   '-vcodec': 'libx265',  #use the h.264 codec
#    '-crf': '23'  
# }) 
# #   '-crf': '0',           #set the constant rate factor to 0, which is lossless
# #   '-x265-params' : 'lossless=1'
# while True:
#     ret,frame=capture.read()
#     if ret==False:
#         print("Bad frame")
#         break
#     print('hi')
#     writer.writeFrame(frame[:,:,::-1])  #write the frame as RGB not BGR
   
# writer.close() #close the writer
# capture.release()
# cv2.destroyAllWindows()