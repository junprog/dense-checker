import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np

class ParicleFilter(object):
    def __init__(self, particles_num):
        self.img_w = 640
        self.img_h = 480
        self.particles_num = particles_num
        self.particles = np.empty([2, self.particles_num]) #x, y
        self.measure_noise = 20
        self.system_noise_x = 3
        self.system_noise_y = 2
        self.ref_x = 0
        self.ref_y = 0
    
    def initialize(self):
        # generate random particles
        self.particles[0,:] = np.random.randint(0, self.img_w, (1,self.particles_num))
        self.particles[1,:] = np.random.randint(0, self.img_h, (1,self.particles_num))

    def predict(self):
        self.particles[0, :] += np.random.randint(0, self.system_noise_x, (self.particles_num)) 
        self.particles[1, :] += np.random.randint(0, self.system_noise_y, (self.particles_num)) 
    
    def normalize(self, weight):
        return weight / np.sum(weight)

    def calcLikelihood4TrackingWhite(self, img):
        mean, std = 250.0, 10.0
        intensity = []

        for i in range(self.particles_num):
            x, y = int(self.particles[0,i]), int(self.particles[1,i])
            if y >= 0 and y < self.img_h and x >= 0 and x < self.img_w:
                intensity.append(img[y,x])
            else:
                intensity.append(-1)

        weights = 1.0 / np.sqrt(2 * np.pi * std) * np.exp(-(np.array(intensity) - mean)**2 /(2 * std**2))
        weights[intensity == -1] = 0
        weights = self.normalize(weights)
        return weights
        
    def calcLikelihood4Densemap(self, img):
        std = self.measure_noise * self.measure_noise 
        dif_xy = []
        for i in range(self.particles_num):
            x, y = int(self.particles[0,i]), int(self.particles[1,i])
            if y >= 0 and y <= self.img_h and x >= 0 and x <= self.img_w:
                xy_tmp = np.array([self.ref_x -x, self.ref_y - y])
                dif_xy.append(np.linalg.norm(xy_tmp, ord=2))
            else:
                dif_xy.append(-1)

        weights = 1.0 / np.sqrt(2 * np.pi * std) * np.exp(-(np.array(dif_xy))**2 /(2 * std))
        weights[dif_xy == -1] = 0
        weights = self.normalize(weights)
        return weights
    
    def resampling(self, weight):
        index = np.arange(self.particles_num)
        sample = []

        for i in range(self.particles_num):
            idx = np.random.choice(index, p=weight)
            sample.append(idx)

        return sample

    def filtering(self, img, human_coords):
        self.ref_x, self.ref_y = human_coords[0], human_coords[1]
        self.predict()
        # weights = self.calcLikelihood4TrackingWhite(img)
        weights = self.calcLikelihood4Densemap(img)
        index = self.resampling(weights)
       
        self.particles[1,:] = self.particles[1,index]
        self.particles[0,:] = self.particles[0,index]
         
        return np.sum(self.particles[0,:]) / float(len(self.particles[0,:])), np.sum(self.particles[1,:]) / float(len(self.particles[1,:]))