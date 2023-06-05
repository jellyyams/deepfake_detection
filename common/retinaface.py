"""
Most of code from detect.py in Pytorch_Retinaface (https://github.com/biubug6/Pytorch_Retinaface)
"""


from __future__ import print_function
import os
from operator import itemgetter
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import time
import cv2
import logging
import sys
sys.path.append('/home/hadleigh/Pytorch_Retinaface')
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    logging.info('Loading pretrained RetinaFace model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


class RetinaFaceDetector(object):
    def __init__(self, network, confidence_threshold=0.02, top_k=5000, nms_threshold=0.4, keep_top_k=750):
        self.confidence_threshold = confidence_threshold
        self.top_k  = top_k
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k

        torch.set_grad_enabled(False)
        self.cfg = None
        if network == "mobile0.25":
            self.cfg = cfg_mnet
            trained_model = '../common/weights/mobilenet0.25_Final.pth' 
        elif network == "resnet50":
            self.cfg = cfg_re50
            trained_model = '../common/weights/Resnet50_Final.pth'
        # net and model
        self.net = RetinaFace(cfg=self.cfg, phase = 'test')
        self.net = load_model(self.net, trained_model, False)
        self.net.eval()
        # print(net)
        cudnn.benchmark = True
        self.device = torch.device("cuda")
        self.net = self.net.to(self.device)

        self.resize = 1
    
    def detect(self, img_raw):
        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)
        loc, conf, landms = self.net(img)  # forward pass


        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        if len(dets) == 0:
            return
        
        #sort detections by confidence and return one with highest confidence to be target
        sorted(dets, key=itemgetter(4))
        target_det = dets[0]
        b = list(map(int, target_det))
        return (b[0], b[1], b[2], b[3])
        




      