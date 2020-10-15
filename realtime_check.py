import os
import time

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import torchvision.transforms.functional as F

#mean = np.array([0.485, 0.456, 0.406])
#std = np.array([0.229, 0.224, 0.225])

def extracter(img, model, trans):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model.to(device)

    model.eval()
    with torch.no_grad():
        img = trans(img).unsqueeze_(0)
        #img.to('cuda')

        out = model(img)

    out = out.to('cpu').detach().numpy().copy()
    out = np.squeeze(out)
    #out = out*std+mean

    return out

def dense_checker():
    ##
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    model = models.resnet18(pretrained=True)
    modules = list(model.children())[:-2]
    model = nn.Sequential(*modules)
    model.add_module('output_layer', nn.Conv2d(512,1,(3,3),padding=(1,1)))

    print(model)

    capture_camera(trans, model)
    ##

def capture_camera(trans, model, mirror=True):
    # Capture video from camera
    cap = cv2.VideoCapture(0+cv2.CAP_DSHOW) # device num

    while True:
        ret, frame = cap.read()

        # flip
        if mirror is True:
            frame = frame[:,::-1]

        # BGR -> RGB
        rgb_img = np.zeros_like(frame)
        rgb_img[:,:,0] = frame[:,:,2]
        rgb_img[:,:,1] = frame[:,:,1] 
        rgb_img[:,:,2] = frame[:,:,0]

        out = extracter(rgb_img, model, trans)
        out = cv2.resize(out,dsize=(640, 480))
        # plot
        cv2.imshow('camera capture', out)

        k = cv2.waitKey(1) # wait 1 [msec]
        if k == 27: # exit press : [Esc]
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    dense_checker()