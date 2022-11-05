from matplotlib import pyplot as plt
import numpy as np

class Particle:
    def __init__(self,r, phi, B=1., charge=1.,granularity = 100):
        self.granularity = granularity
        self.radius = r
        self.phi = phi
        self.B = B
        self.charge = charge

        self.x = self.radius*np.sin(self.phi+self.charge*np.pi/2)
        self.y = self.radius*np.cos(self.phi+self.charge*np.pi/2)
        
    def momentum(self):
        return abs(self.radius) * self.B * abs(self.charge)

    def trace_array(self):
        self.x = self.radius*np.sin((self.phi+self.charge*np.pi/2))
        self.y = self.radius*np.cos((self.phi+self.charge*np.pi/2))
        theta = np.linspace(-np.pi/2+self.phi,+np.pi/2+self.phi,self.granularity)
        x = abs(self.radius)*np.sin(theta)+ self.x
        y = abs(self.radius)*np.cos(theta)+ self.y
        return np.array([x,y])    
        