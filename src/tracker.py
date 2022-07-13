from copy import deepcopy
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd


granularity = 100

def get_track_line(radius, begin, end):
    t = np.linspace(begin, end, granularity)
    return radius*np.cos(t), radius*np.sin(t)

def get_ecl_lines(radius, begin, end, width):
        t = np.linspace(begin, end, granularity)
        inner_x = radius*np.cos(t)
        inner_y = radius*np.sin(t)
        outer_x = (radius+width)*np.cos(t)
        outer_y = (radius+width)*np.sin(t)
        right_x = np.linspace(inner_x[0], outer_x[0], granularity)
        right_y = np.linspace(inner_y[0], outer_y[0], granularity)
        left_x = np.linspace(inner_x[-1], outer_x[-1], granularity)
        left_y = np.linspace(inner_y[-1], outer_y[-1], granularity)
        return [inner_x, inner_y], [outer_x, outer_y], [right_x, right_y], [left_x, left_y]

class Tracker:
    def __init__(self, layers, n_segments, k = 2, noise = False):
        self.noise = noise
        self.n_segments = n_segments
        self.col_names = ["radius", "begin", "end", "lines", "size", "content", "selected", "type", "facecolor", "edgecolor"]
        self.segments = pd.DataFrame(columns = self.col_names)
        counter = 0
        for l in range(layers):
            len_segment = 2*np.pi/(n_segments+k*l)
            for i in range(n_segments+k*l):
                radius = l
                begin = len_segment*i+(0.2/(l+1))
                end = len_segment*(i+1)-(0.2)/(l+1)
                content = "noise" if np.random.rand()<self.noise else "empty"
                selected = "not"
                lines = get_track_line(radius, begin, end)
                size = 3
                self.segments.loc[counter] = [radius, begin, end, lines, size, content, selected, "tracking", "gray", "black"]
                counter += 1
        l = layers
        len_segment = 2*np.pi/(n_segments+k*l)
        for i in range(n_segments+k*l):
            radius = l
            begin = len_segment*i+(0.2/(l+1))
            end = len_segment*(i+1)-(0.2)/(l+1)
            content = "noise" if np.random.rand()<self.noise else "empty"
            selected = "not"
            lines = get_ecl_lines(radius, begin, end, 1)
            size = 2
            self.segments.loc[counter] = [radius, begin, end, lines, size, content, selected, "ecl", "gray", "black"]
            counter += 1
        self.lines = []
        
        # tack edgeline list at the front
        for lx, ly in self.segments.query("type=='tracking'")["lines"]:
            self.lines.append([lx,ly])
        self.edge_lines = deepcopy(self.lines)
        for pair_list in self.segments.query("type=='ecl'")["lines"]:
            for pair in pair_list:
                self.lines.append(pair)
        for lx, ly in self.segments.query("type=='tracking'")["lines"]:
            self.lines.append([lx,ly])
        self.lines = np.array(self.lines)
        self.lines = np.moveaxis(self.lines,1,-1)
        
        self.sizes = []
        for s in self.segments.query("type=='tracking'")["size"]:
            self.sizes.append(8)
        for s in self.segments.query("type=='ecl'")["size"]:
            for i in range(4):
                self.sizes.append(s)
        for s in self.segments.query("type=='tracking'")["size"]:
            self.sizes.append(s)
        
    def get_colors(self):
        colors_tracking = self.segments.query("type=='tracking'")["edgecolor"]
        colors_ecl = self.segments.query("type=='ecl'")["edgecolor"].repeat(4)
        colors_edges= self.segments.query("type=='tracking'")["facecolor"]
        colors = pd.concat([colors_tracking, colors_ecl, colors_edges])
        return colors

    
    def get_collection(self):
        self.set_colors()
        line_collection = LineCollection(self.lines, color = self.get_colors(), linewidths = self.sizes)
        return line_collection
    
    def check_hit(self, particle):
        d = particle.radius*particle.charge
        a=(self.segments.radius**2)/(2*d)
        h=np.sqrt(abs(self.segments.radius**2-a**2))
        x2=a*(particle.x)/d   
        y2=a*(particle.y)/d   

        x4=x2-h*(particle.y)/d
        y4=y2+h*(particle.x)/d
        
        theta = np.arctan2(y4,x4)

        mask = theta < 0
        theta[mask] += 2*np.pi

        return (theta > self.segments.begin) & (theta < self.segments.end) & (self.segments.radius <= 2*particle.radius)
    
    def mark_hits(self, particle):
        hit_segments = self.check_hit(particle)
        self.segments.loc[hit_segments,"content"] = "hit"
    
    def set_particle_selection(self, particle, hidden = False):
        hit_segments = self.check_hit(particle)
        self.segments.loc[hit_segments,"selected"] = "selected" if not hidden else "hidden"
    
    def set_colors(self):
        self.segments.loc[:,"edgecolor"] = "gray"
        self.segments.loc[:,"facecolor"] = "gray"
        self.segments.loc[self.segments["content"]!='empty', "facecolor"] = "red"
        self.segments.loc[self.segments["selected"]=='selected',"edgecolor"] = "blue"
        self.segments.loc[self.segments["selected"]=='hidden' , "edgecolor"] = "teal"
