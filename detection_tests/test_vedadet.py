import argparse

import cv2
import numpy as np
import torch
import time
import sys

sys.path.append('../vedadet')
from vedacore.image import imread, imwrite
from vedacore.misc import Config, color_val, load_weights
from vedacore.parallel import collate, scatter
from vedadet.datasets.pipelines import Compose
from vedadet.engines import build_engine


def prepare(cfg):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'
    engine = build_engine(cfg.infer_engine)

    engine.model.to(device)
    load_weights(engine.model, cfg.weights.filepath)

    data_pipeline = Compose(cfg.data_pipeline)
    return engine, data_pipeline, device


def plot_result(result, frame, class_names):
    font_scale = 0.5
    bbox_color = 'green'
    text_color = 'green'
    thickness = 1

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)
    img = frame

    bboxes = np.vstack(result)
    labels = [
        np.full(bbox.shape[0], idx, dtype=np.int32)
        for idx, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox[:4].astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness)
        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    return img


def main():

    cfg = Config.fromfile('../vedadet/configs/infer/tinaface/tinaface_r50_fpn_bn_video.py')
    class_names = cfg.class_names
    
    engine, data_pipeline, device = prepare(cfg)
    
    input_vid_path = '../inputs/two_pose_test4_p1.mp4'
    input_cap = cv2.VideoCapture(input_vid_path)

    W , H = int(input_cap.get(3)), int(input_cap.get(4))
    out_vid = cv2.VideoWriter('test_det.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 15, (W, H))

    tot_det_time = 0
    tot_frames = 0
    while input_cap.isOpened():

        ret, frame = input_cap.read()
        if not ret:
            break
        start = time.time()
        data = dict(img=frame, filename=None, img_prefix=None, ori_filename=None, ori_shape=frame.shape)
        data = data_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if device != 'cpu':
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            # just get the actual data from DataContainer
            data['img_metas'] = data['img_metas'][0].data
            data['img'] = data['img'][0].data
        result = engine.infer(data['img'], data['img_metas'])[0]
        end = time.time()
        tot_det_time += (end - start)
        tot_frames += 1
        annotated = plot_result(result, frame, class_names)
        out_vid.write(annotated)
    
    print("Avg FPS: ", tot_frames / tot_det_time)
    input_cap.release()
    out_vid.release()


if __name__ == '__main__':
    main()
