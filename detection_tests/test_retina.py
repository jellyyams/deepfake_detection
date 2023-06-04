from retinaface import RetinaFace
import cv2
import time

def main():

 
    input_vid_path = '../inputs/two_pose_test4_p1.mp4'
    input_cap = cv2.VideoCapture(input_vid_path)

    W , H = int(input_cap.get(3)), int(input_cap.get(4))
    #out_vid = cv2.VideoWriter('test_det.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 15, (W, H))

    tot_det_time = 0
    tot_frames = 0
    while input_cap.isOpened():

        ret, frame = input_cap.read()
        if not ret:
            break
        start = time.time()
        resp = RetinaFace.detect_faces(frame)
        end = time.time()
        tot_det_time += (end - start)
        tot_frames += 1
        print(resp)
    
    print("Avg FPS: ", tot_frames / tot_det_time)
    input_cap.release()
    #out_vid.release()


if __name__ == '__main__':
    main()
