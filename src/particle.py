from matplotlib import pyplot as plt
import numpy as np
granularity = 100

class Particle:
    def __init__(self,r, phi, B=1., charge=1.):
        self.radius = r
        self.phi = phi
        self.B = B
        self.charge = charge

        self.x = self.radius*np.sin(self.phi+self.charge*np.pi/2)
        self.y = self.radius*np.cos(self.phi+self.charge*np.pi/2)
        
    def momentum(self):
        return abs(self.radius) * self.B * abs(self.charge)
    
    def draw(self, ax):
        self.x = self.radius*np.sin((self.phi+self.charge*np.pi/2))
        self.y = self.radius*np.cos((self.phi+self.charge*np.pi/2))
        theta = np.linspace(-np.pi/2+self.phi,+np.pi/2+self.phi)
        x = abs(self.radius)*np.sin(theta)+ self.x
        y = abs(self.radius)*np.cos(theta)+ self.y
        c = "blue" #if self.radius<0 else "red"
        ax.plot(x,y,color = c, label=f'$p_T$ = {round(self.momentum(),5)} GeV, Q = {self.charge}')
        