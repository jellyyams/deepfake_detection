import cv2
import glob
import json
import logging
import logging.config

def reconstruct_vid(input_video_path, evm_data_dir, auth_data_dir, log_level='INFO'):
    logging.info('Reconstructing video')
    LOGGING_CONFIG = { 
        'version':1,
        'disable_existing_loggers': True,
        'formatters': { 
            'standard': { 
                'format': '%(levelname)s in reconstruct_vid: %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
            }
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': 'INFO',
                'propagate': True
            },
        },
        'root': {
            'handlers': ['default'],
            'level': log_level
        }
    }
    logging.config.dictConfig(LOGGING_CONFIG)
    
    with open(auth_data_dir + '/data.json', 'r') as openfile:
        data_json_object = json.load(openfile)

    target_landmarks = data_json_object['target_landmarks']
    
    evm_video_paths = glob.glob(evm_data_dir + '*')
    evm_video_caps = []
    for l_num in target_landmarks:
        target_region_vid_path = list(filter(lambda x: (f'target_region{l_num}' in x) and ('channel_vis' not in x), evm_video_paths))[0]
        evm_video_caps.append(cv2.VideoCapture(target_region_vid_path))


    main_input_video_cap = cv2.VideoCapture(input_video_path)
    main_video_len = int(main_input_video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) #frames. The EVM videos will be 10 frames shorter (specifically, the last 10 frames are truncated) due to the Matlab implementation idiosyncrasies. 
    W , H = int(main_input_video_cap.get(3)), int(main_input_video_cap.get(4)) #input video dimensions
    input_cap_fps = int(main_input_video_cap.get(cv2.CAP_PROP_FPS))

    # out_vid_name = 'reconstructed_output/reconstructed_{}.mp4'.format(auth_data_dir.split('/')[-1])
    out_vid_name = '{}reconstructed.mp4'.format(evm_data_dir)
    out_vid = cv2.VideoWriter(out_vid_name, cv2.VideoWriter_fourcc(*'mp4v'), input_cap_fps, (W, H))
    logging.info('Reconstructed video being written to {}'.format(out_vid_name))

    frame_num = 0 
    while main_input_video_cap.isOpened():
        ret, main_frame = main_input_video_cap.read()
        if ret:
            frame_num += 1
            # if frame_num == main_video_len - 10: #uncomment if using Matlab EVM
            #     break
            for i, l_num in enumerate(target_landmarks):
                #read frame form appropriate evm-d target region video
                _, target_region_frame = evm_video_caps[i].read()
                #get x,y coordinate on face of this region's center at this particular frame
                x, y  = data_json_object['frames']['frame' + str(frame_num)]["target_landmark_coords"][str(l_num)]
                #replace portion of main frame with evm-ed target region frame
                box_width = int(evm_video_caps[i].get(3))
                left = int(max(0, x - (box_width/2)))
                top = int(max(0, y - (box_width/2)))
                right = int(min(W, x + (box_width/2)))
                bottom = int(min(H, y + (box_width/2)))
                main_frame[top:bottom, left:right, :] = target_region_frame
            out_vid.write(main_frame)
        else:
            break

    main_input_video_cap.release()
    out_vid.release()
    logging.info('Done reconstructing video')
    return out_vid_name