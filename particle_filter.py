import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np

class ParicleFilter(object):
    def __init__(self, particles_num):
        self.img_w = 0
        self.img_h = 0
        self.particles_num = particles_num
        self.particles = np.empty([2, self.particles_num]) #x, y
        self.measure_noise = 20
        self.system_noise = 20

    
    def initialize(self, img):
        self.img_h, self.img_w, intensity = img.shape
        # generate random particles
        self.particles[0,:] = np.random.randint(0, self.img_w, (1,self.particles_num))
        self.particles[1,:] = np.random.randint(0, self.img_h, (1,self.particles_num))

    def predict(self):
        self.particles[0, :] += np.random.random(self.particles_num) * self.system_noise - 10 
        self.particles[1, :] += np.random.random(self.particles_num) * self.system_noise - 10
    
    def normalize(self, weight):
        return weight / np.sum(weight)

    def calcLikelihood4TrackingWhite(self,img):
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
    
    def resampling(self, weight):
        index = np.arange(self.particles_num)
        sample = []

        for i in range(self.particles_num):
            idx = np.random.choice(index, p=weight)
            sample.append(idx)

        return sample

    def filtering(self, img):
        self.predict()
        weights = self.calcLikelihood4TrackingWhite(img)
        index = self.resampling(weights)
       
        self.particles[1,:] = self.particles[1,index]
        self.particles[0,:] = self.particles[0,index]
         
        return np.sum(self.particles[0,:]) / float(len(self.particles[0,:])), np.sum(self.particles[1,:]) / float(len(self.particles[1,:]))

 
    def main(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        self.initialize(frame)

        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            x, y = self.filtering(gray)
            
            frame = cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), -1)
            for i in range(self.particles_num):
                frame = cv2.circle(frame, (int(self.particles[0,i]),int(self.particles[1,i])), 2, (0, 0, 255), -1)
            
            cv2.imshow("frame", frame)
        
            if cv2.waitKey(20) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    filter = ParicleFilter(1000)  #particles_num
    filter.main()