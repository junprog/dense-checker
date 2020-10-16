import os
import time

import cv2
import numpy as np

import torch
from torchvision import transforms

from calc_fps import fpsCalculator
from models.vgg import vgg19

class denseChecker(object):
    def __init__(self, use_camera=True, media_path=None, model_path='data/ucf_best_model.pth'):
        self.use_camera = use_camera

        if self.use_camera == False:
            assert media_path is not None, "If use an existing movie, you specify the movie's path"
            self.media_path = media_path
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        self.model = vgg19()
        # load pre-trained model
        self.model.load_state_dict(torch.load(model_path, self.device))
        # model to GPU or CPU
        self.model.to(self.device)

        # fps calculator instance
        self.fps = fpsCalculator()

    def check(self, mirror=True):
        if self.use_camera: # Capture video from camera
            cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        else: # Capture video from an existing movie
            cap = cv2.VideoCapture(self.media_path)
            mirror = False

        while True:
            _, frame = cap.read()

            # flip
            if mirror is True:
                frame = frame[:,::-1]

            if max(frame.shape) > 1300:
                frame = cv2.resize(frame, dsize=(int(frame.shape[1]*0.5), int(frame.shape[0]*0.5)))

            # BGR(cv2) -> RGB(numpy)
            img = self._cvimg2np(frame)

            # regress density map from image
            out = self._regression(img)

            # count
            num = int(round(out.sum()))
            out = cv2.resize(out, dsize=(int(out.shape[1]*8), int(out.shape[0]*8)))

            self.fps.tick_tack()

            # plot
            out = cv2.applyColorMap(self._norm_uint8(out), cv2.COLORMAP_JET)
            cv2.putText(out, "FPS : {:.3f}   Poeple Count : {}".format(self.fps.getFPS(), num), (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)

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

    def _regression(self, img):
        self.model.eval()
        with torch.no_grad():
            img = self.trans(img).unsqueeze_(0)
            img = img.to(self.device)
            out = self.model(img)

        out = out.to('cpu').detach().numpy().copy()
        out = np.squeeze(out)

        return out
    
    def _norm_uint8(self, img, axis=None): 
        Min = img.min(axis=axis, keepdims=True)
        Max = img.max(axis=axis, keepdims=True)
        out = (img-Min)/(Max-Min)
        
        return (out * 255).astype(np.uint8)

if __name__ == '__main__':
    #checker = denseChecker(use_camera=False, media_path='data/shinjuku1.mp4')
    checker = denseChecker()
    checker.check()