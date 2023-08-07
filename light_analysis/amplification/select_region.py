import cv2
 
# Lists to store the bounding box coordinates
top_left_corner=[]
bottom_right_corner=[]
 
# function which will be called on mouse input
def drawRectangle(action, x, y, flags, *userdata):
  # Referencing global variables 
  global top_left_corner, bottom_right_corner
  # Mark the top left corner when left mouse button is pressed
  if action == cv2.EVENT_LBUTTONDOWN:
    top_left_corner = [(x,y)]
    # When left mouse button is released, mark bottom right corner
  elif action == cv2.EVENT_LBUTTONUP:
    bottom_right_corner = [(x,y)]    
    # Draw the rectangle
    cv2.rectangle(image, top_left_corner[0], bottom_right_corner[0], (0,255,0),2, 8)
    cv2.imshow("Window",image)
 

output_vid_path = 'aug6_cropped_videos'
input_vid_path = 'aug6_input_videos/r60_g0_b0_1000mson_1000msoff_N30_buff30_rep15.MP4'


input_capture = cv2.VideoCapture(input_vid_path)

first_frame = None
while input_capture.isOpened():
    ret, frame = input_capture.read()
    if ret:
        first_frame = frame
        break
    else:
        break
input_capture.release()

# Read Images
image = first_frame
# Make a temporary image, will be useful to clear the drawing
temp = image.copy()
# Create a named window
cv2.namedWindow("Window")
# highgui function called when mouse events occur
cv2.setMouseCallback("Window", drawRectangle)
 
k=0
# Close the window when key q is pressed
while k!=113:
  # Display the image
  cv2.imshow("Window", image)
  k = cv2.waitKey(0)
  # If c is pressed, clear the window, using the dummy image
  if (k == 99):
    image= temp.copy()
    cv2.imshow("Window", image)
 
print(top_left_corner)
print(bottom_right_corner)
cv2.destroyAllWindows()

if top_left_corner[0][0] > bottom_right_corner[0][0]:
   temp = top_left_corner
   top_left_corner = bottom_right_corner[0]
   bottom_right_corner = temp[0]
else:
    top_left_corner = top_left_corner[0]
    bottom_right_corner = bottom_right_corner[0]

input_capture_fresh = cv2.VideoCapture(input_vid_path)
frame_width = bottom_right_corner[0] - top_left_corner[0]
frame_height =  bottom_right_corner[1] - top_left_corner[1]
frame_size = (frame_width,frame_height)
fps = input_capture_fresh.get(cv2.CAP_PROP_FPS)
input_vid_name = input_vid_path.split('/')[-1].split('.')[0]
output_video_mpjpeg = cv2.VideoWriter(f'{output_vid_path}/{input_vid_name}_mjpeg.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, frame_size)
output_video_cropped_rgba = cv2.VideoWriter(f'{output_vid_path}/{input_vid_name}_croppedrgba.avi', cv2.VideoWriter_fourcc('R', 'G', 'B', 'A'), fps, frame_size)

frame_num = 0
avi_lim = 1000
while input_capture_fresh.isOpened():
    ret, frame = input_capture_fresh.read()
    if ret:
        crop = frame[top_left_corner[1]:bottom_right_corner[1],top_left_corner[0]:bottom_right_corner[0]]
        output_video_mpjpeg.write(crop)
        if frame_num <= avi_lim:
            output_video_cropped_rgba.write(crop)
        frame_num += 1 
    else:
        break

input_capture_fresh.release()
output_video_cropped_rgba.release()
output_video_mpjpeg.release()