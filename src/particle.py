from matplotlib import pyplot as plt
import numpy as np
granularity = 100

class Particle:
    def __init__(self,r, phi, B=1., charge=1.):
        self.x = r*np.cos(phi)
        self.y = r*np.sin(phi)
        self.radius = r
        self.phi = phi
        self.B = B
        self.charge = charge
        
    def momentum(self):
        return abs(self.radius) * self.B * abs(self.charge)
    
    def draw(self, ax):
        self.x = self.charge*self.radius*np.cos(self.phi)
        self.y = self.charge*self.radius*np.sin(self.phi)
        theta = np.linspace(0,np.pi,granularity)
        x = abs(self.radius)*np.cos(theta+self.phi)+ self.x
        y = abs(self.radius)*np.sin(theta+self.phi)+ self.y
        c = "blue" #if self.radius<0 else "red"
        ax.plot(x,y,color = c, label=f'$p_T$ = {round(self.momentum(),5)} GeV, Q = {self.charge}')
        

class ToyParticle:
    def __init__(self,r, phi, B=1., charge=1., momentum=1.):
        self.x = charge*r*np.cos(phi)
        self.y = charge*r*np.sin(phi)
        self.radius = r
        self.phi = phi
        self.B = B
        self.momentum = momentum
        self.charge = charge
        
    #def momentum(self):
    #    return abs(self.radius) * self.B * abs(self.charge)
    
    def draw(self, ax):
        self.x = self.charge*self.radius*np.cos(self.phi)
        self.y = self.charge*self.radius*np.sin(self.phi)
        theta = np.linspace(0,np.pi,granularity)
        x = abs(self.radius)*np.cos(theta+self.phi)+ self.x
        y = abs(self.radius)*np.sin(theta+self.phi)+ self.y
        c = "blue" #if self.radius<0 else "red"
        ax.plot(x,y,color = c, label=f'$p_T$ = {round(self.momentum,5)} GeV, Q = {self.charge}')
        plt.legend()