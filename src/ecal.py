from matplotlib.pyplot import sca
import pandas as pd
from copy import deepcopy
import numpy as np

#test

from matplotlib.colors import to_rgba_array
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

class ECal:
    def __init__(self, nrows, ncols, particles,  crystal_edge = 0.5, noise_rate = 0, linewidth = 1.75):
        column_names = ["x", "y", "edge", "content", "edgecolor", "facecolor", "center", "patch"]
        self.particles = particles
        self.n_particles = len(self.particles)
        self.crystals_df = pd.DataFrame(columns = column_names)
        self.center = self.particles.idxmax(axis=1).to_numpy().astype('int')
        self.select_particles = deepcopy(self.particles)
        self.select_particles.loc[:,:] = 0
        counter = 0
        for c in range(ncols):
            for r in range(nrows):
                content = 0
                if np.random.rand() < noise_rate:
                    content = np.random.normal(loc = 0.025, scale = 0.01)
                x = r*crystal_edge
                y =  c*crystal_edge
                edge = crystal_edge-linewidth/4
                patch = Rectangle((x, y), edge, edge, edgecolor = "black", facecolor = "gray", linewidth = linewidth)
                self.crystals_df.loc[counter,column_names] = [x,y, edge, content, "black", "gray", False, patch]
                counter += 1
        for i in range(self.n_particles):
            self.crystals_df.loc[:,"content"] += self.particles.iloc[i].to_numpy()
        self.set_colors(0)
        self.collection = PatchCollection(self.crystals_df.loc[:,"patch"], match_original=True)
        colors = to_rgba_array(self.crystals_df.loc[:,"edgecolor"].to_numpy())
        self.collection.set_edgecolors(colors)
        
    
    def set_colors(self, selected_index):
        not_centers_mask = self.crystals_df.loc[:,"center"] == False
        centers_mask = self.crystals_df.loc[:,"center"] == True
        center = self.center[selected_index]
        self.crystals_df.loc[:,"facecolor"] = "gray"
        self.crystals_df.loc[:,"edgecolor"] = "gray"

        selected_mask = self.select_particles.loc[selected_index,:].to_numpy()>0
        self.crystals_df.loc[selected_mask, "edgecolor"] = "blue"
        hidden_mask = np.zeros(len(self.crystals_df))
        for i in range(self.n_particles):
            if i != selected_index:
                hidden_mask += self.select_particles.loc[i,:].to_numpy()
        hidden_mask = hidden_mask > 0
        self.crystals_df.loc[hidden_mask, "edgecolor"] = "lightyellow"
        self.crystals_df.loc[center,"edgecolor"] = "black"
        hit_mask = self.crystals_df["content"] > 0
        self.crystals_df.loc[hit_mask,"facecolor"] = "red"
    
    def get_crystal(self, rectangle):
        x,y = rectangle.get_xy()
        for i,r in enumerate(self.crystals):
            for j,c in enumerate(r):
                if x == c.x and y==c.y:
                    return i,j