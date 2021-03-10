import os
import time
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import argparse

import cv2
import numpy as np

import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Dense Check Parameters')
    
    parser.add_argument('--model', default='vgg', type=str, help='set the model architecture')
    parser.add_argument('--data-dir', default='data', type=str, help='set the directory where weights and movies put')
    parser.add_argument('--weight-path', default='ucf_vgg_best_model.pth', type=str, help='set the model architecture')

    parser.add_argument('--particle-num', default=1000, type=int, help='set the number of particle')

    parser.add_argument('--use-movie', action='store_false', help='if use an existing movie, set this option')
    parser.add_argument('--media-path', default='shinjuku1.mp4', type=str, help='if use an existing movie, set the file name of movie')
    
    args = parser.parse_args()
    return args

class VideoDenseChecker(object):
    def __init__(self, ctr, fps ,use_camera=True, media_path=None):
        self.use_camera = use_camera

        if self.use_camera: # Capture video from camera
            self.video = cv2.VideoCapture(0)
            self.mirror = True

        else: # Capture video from an existing movie
            assert media_path is not None, "If use an existing movie, you specify the movie's path : modea_path"
            self.media_path = media_path
            self.video = cv2.VideoCapture(self.media_path)
            self.mirror = False

        # fps calculator instance
        self.fps = fps

        # Define Counter
        self.counter = ctr

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, frame = self.video.read()
        
        # flip
        if self.mirror is True:
            frame = frame[:,::-1]

        if max(frame.shape) > 1300:
            frame = cv2.resize(frame, dsize=(int(frame.shape[1]*0.5), int(frame.shape[0]*0.5)))

        # BGR(cv2) -> RGB(numpy)
        img = self._cvimg2np(frame)

        # regress a density map and counts from image
        dm, count = self.counter.regression(img)
        out = cv2.resize(dm, dsize=(int(dm.shape[1]*8), int(dm.shape[0]*8)))

        # pick out coordinates of human centroid from dense map
        idx = np.unravel_index(np.argmax(out), out.shape)
        human_coords = idx[1], idx[0]

        # calculate FPS
        self.fps.tick_tack()

        # plot
        out = cv2.applyColorMap(self._norm_uint8(out), cv2.COLORMAP_JET)

        frame = cv2.resize(frame, dsize=(int(frame.shape[1]*1), int(frame.shape[0]*1)))
        out = cv2.addWeighted(frame, 0.7, out, 0.3, 0)
        cv2.putText(out, "FPS : {:.3f}   People Count : {}".format(self.fps.getFPS(), count), (20, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)

        sys.stdout.write("\r FPS: {:.3f}".format(self.fps.getFPS()))
        sys.stdout.flush()

        _, frame = cv2.imencode('.jpg', out)

        return frame

    def check(self):
        while True:
            _, frame = self.video.read()

            # flip
            if self.mirror is True:
                frame = frame[:,::-1]

            if max(frame.shape) > 1300:
                frame = cv2.resize(frame, dsize=(int(frame.shape[1]*0.5), int(frame.shape[0]*0.5)))

            # BGR(cv2) -> RGB(numpy)
            img = self._cvimg2np(frame)

            # regress a density map and counts from image
            dm, count = self.counter.regression(img)
            out = cv2.resize(dm, dsize=(int(dm.shape[1]*8), int(dm.shape[0]*8)))

            # pick out coordinates of human centroid from dense map
            idx = np.unravel_index(np.argmax(out), out.shape)
            human_coords = idx[1], idx[0]

            # calculate FPS
            self.fps.tick_tack()

            # plot
            out = cv2.applyColorMap(self._norm_uint8(out), cv2.COLORMAP_JET)

            frame = cv2.resize(frame, dsize=(int(frame.shape[1]*1), int(frame.shape[0]*1)))
            out = cv2.addWeighted(frame, 0.7, out, 0.3, 0)
            cv2.putText(out, "FPS : {:.3f}   People Count : {}".format(self.fps.getFPS(), count), (20, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
            
            #cv2.imshow('camera capture', cv2.resize(frame, dsize=(int(frame.shape[1]*1), int(frame.shape[0]*1))))
            cv2.imshow('output', out)

            sys.stdout.write("\r FPS: {:.3f}".format(self.fps.getFPS()))
            sys.stdout.flush()

            k = cv2.waitKey(1) # wait 1 [msec]
            if k == 27: # exit press : [Esc]
                sys.stdout.write("\n")
                sys.stdout.flush()
                break

        self.video.release()
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

    checker = VideoDenseChecker(counter, use_camera=args.use_movie, media_path=os.path.join(args.data_dir, args.media_path))
    checker.check()