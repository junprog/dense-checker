import os
import time

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

from calc_fps import fpsCalculator
from models.vgg import vgg19

def capture_camera(trans, model, device, mirror=True):
    # Capture video from camera
    #cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW) # device num

    # Use existing video
    cap = cv2.VideoCapture('data/shinjuku1.mp4')
    mirror = False

    # fps calculator instance
    fps = fpsCalculator()

    # model to GPU or CPU
    model.to(device)

    while True:
        _, frame = cap.read()

        # flip
        if mirror is True:
            frame = frame[:,::-1]

        # BGR -> RGB
        rgb_img = np.zeros_like(frame)
        rgb_img[:,:,0] = frame[:,:,2]
        rgb_img[:,:,1] = frame[:,:,1] 
        rgb_img[:,:,2] = frame[:,:,0]

        out = extracter(rgb_img, model, trans, device)
        num = out.sum()
        out = cv2.resize(out, dsize=(int(out.shape[1]*4), int(out.shape[0]*4)))

        fps.tick_tack()

        cv2.putText(out, "FPS:{:.3f}   Poeple #:{:.3f}".format(fps.getFPS(), num), (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # plot
        cv2.imshow('camera capture', cv2.resize(frame, dsize=(int(frame.shape[1]*0.5), int(frame.shape[0]*0.5))))
        cv2.imshow('output', out)

        k = cv2.waitKey(1) # wait 1 [msec]
        if k == 27: # exit press : [Esc]
            break

    cap.release()
    cv2.destroyAllWindows()

def extracter(img, model, trans, device):
    model.eval()
    with torch.no_grad():
        img = trans(img).unsqueeze_(0)
        img = img.to(device)
        out = model(img)

    out = out.to('cpu').detach().numpy().copy()
    out = np.squeeze(out)

    return out

def dense_checker():
    ##
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    """
    model = models.resnet18(pretrained=True)
    modules = list(model.children())[:-4]
    model = nn.Sequential(*modules)
    model.add_module('output_layer', nn.Conv2d(128,1,(3,3),padding=(1,1)))
    """

    model = vgg19()
    model.load_state_dict(torch.load('data/ucf_best_model.pth', device))

    print(model)

    capture_camera(trans, model, device)

if __name__ == '__main__':
    dense_checker()