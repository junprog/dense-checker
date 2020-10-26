import os
import time
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import argparse

import cv2
import numpy as np

import torch
from torchvision import transforms

from src.calc_fps import FPSCalculator
from src.counter import Counter
from src.particle_filter import ParicleFilter

def parse_args():
    parser = argparse.ArgumentParser(description='Dense Check Parameters')
    
    parser.add_argument('--model', default='vgg', help='set the model architecture')
    parser.add_argument('--data-dir', default='data', help='set the directory where weights and movies put')
    parser.add_argument('--weight-path', default='ucf_vgg_best_model.pth', help='set the model architecture')

    parser.add_argument('--particle-num', default=1000, help='set the number of particle')

    parser.add_argument('--use-movie', action='store_false', help='if use an existing movie, set this option')
    parser.add_argument('--media-path', default='shinjuku1.mp4', help='if use an existing movie, set the file name of movie')
    
    args = parser.parse_args()
    return args

class DenseChecker(object):
    def __init__(self, ctr, pf, use_camera=True, media_path=None):
        self.use_camera = use_camera

        if self.use_camera == False:
            assert media_path is not None, "If use an existing movie, you specify the movie's path : modea_path"
            self.media_path = media_path

        # fps calculator instance
        self.fps = FPSCalculator()

        # Define Counter and Particle Filter
        self.counter = ctr
        self.particle_filter = pf

    def check(self):
        if self.use_camera: # Capture video from camera
            cap = cv2.VideoCapture(0)
            mirror=True
        else: # Capture video from an existing movie
            cap = cv2.VideoCapture(self.media_path)
            mirror = False

        self.particle_filter.initialize()
        while True:
            _, frame = cap.read()

            # flip
            if mirror is True:
                frame = frame[:,::-1]

            #if max(frame.shape) > 1300:
            #    frame = cv2.resize(frame, dsize=(int(frame.shape[1]*0.5), int(frame.shape[0]*0.5)))

            # BGR(cv2) -> RGB(numpy)
            img = self._cvimg2np(frame)

            # regress a density map and counts from image
            dm, count = self.counter.regression(img)
            out = cv2.resize(dm, dsize=(int(dm.shape[1]*8), int(dm.shape[0]*8)))

            # pick out coordinates of human centroid from dense map
            idx = np.unravel_index(np.argmax(out), out.shape)
            human_coords = idx[1], idx[0]
            print("human_coords: ", idx[1], idx[0])
            # apply particle filter
            x, y = self.particle_filter.filtering(out,human_coords)

            # calculate FPS
            self.fps.tick_tack()

            # plot
            out = cv2.applyColorMap(self._norm_uint8(out), cv2.COLORMAP_JET)

            out = cv2.circle(out, (int(x), int(y)), 10, (255, 255, 255), -1)
            print("particle: ", int(x), int(y))
            for i in range(self.particle_filter.particles_num):
                out = cv2.circle(out, (int(self.particle_filter.particles[0,i]),int(self.particle_filter.particles[1,i])), 2, (255, 255, 255), -1)

            # cv2.putText(out, "FPS : {:.3f}   People Count : {}".format(self.fps.getFPS(), count), (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('camera capture', cv2.resize(frame, dsize=(int(frame.shape[1]*1), int(frame.shape[0]*1))))
            cv2.imshow('output', out)

            k = cv2.waitKey(1) # wait 1 [msec]
            if k == 27: # exit press : [Esc]
                break

        cap.release()
        cv2.destroyAllWindows()

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
    particlefilter = ParicleFilter(args.particle_num)

    checker = DenseChecker(counter, particlefilter, use_camera=args.use_movie, media_path=os.path.join(args.data_dir, args.media_path))
    checker.check()