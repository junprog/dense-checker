import os
import time
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import argparse

import cv2
import numpy as np

import torch

from src.counter import Counter
from src.particle_filter import ParicleFilter

def parse_args():
    parser = argparse.ArgumentParser(description='Dense Check Parameters')
    
    parser.add_argument('--model', default='vgg', type=str, help='set the model architecture')
    parser.add_argument('--data-dir', default='data', type=str, help='set the directory where weights and movies put')

    parser.add_argument('--weight-path', default='ucf_vgg_best_model.pth', type=str, help='set the pre-train weight file name')
    parser.add_argument('--media-path', default='IMG_1884.JPG', type=str, help='if use an existing movie, set the file name')
    
    args = parser.parse_args()
    return args

class DenseChecker(object):
    def __init__(self, ctr, media_path=None):
        self.media_path = media_path

        # Define Counter
        self.counter = ctr

    def check(self):
        cv2_img = cv2.imread(self.media_path)

        # BGR(cv2) -> RGB(numpy)
        img = self._cvimg2np(cv2_img)

        # regress a density map and counts from image
        dm, count = self.counter.regression(img)
        out = cv2.resize(dm, dsize=(int(dm.shape[1]*8), int(dm.shape[0]*8)))

        # plot
        out = cv2.applyColorMap(self._norm_uint8(out), cv2.COLORMAP_JET)

        out = cv2.addWeighted(cv2_img, 0.5, out, 0.5, 0)
        cv2.putText(out, "People Count : {}".format(count), (20, 100), cv2.FONT_HERSHEY_PLAIN, 8, (255, 255, 255), 4, cv2.LINE_AA)
        
        cv2.imshow('output', out)
        cv2.imwrite(os.path.join('data', 'out.jpg'), out)

    def _cvimg2np(self, img):
        out = np.zeros_like(img)
        out[:,:,0] = img[:,:,2]
        out[:,:,1] = img[:,:,1] 
        out[:,:,2] = img[:,:,0]

        return out
    
    def _norm_uint8(self, img, axis=None): 
        Min = img.min(axis=axis, keepdims=True)
        Max = img.max(axis=axis, keepdims=True)
        out = (img-Min)/(Max-Min)
        
        return (out * 255).astype(np.uint8)

if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    counter = Counter(model=args.model, model_path=os.path.join(args.data_dir, args.weight_path))

    checker = DenseChecker(counter, media_path=os.path.join(args.data_dir, args.media_path))
    checker.check()