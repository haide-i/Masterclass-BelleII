from copy import deepcopy
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd


granularity = 100  #must be multiple of 4

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

def get_ecl_lines_single_array(radius, begin, end, width):
        t = np.linspace(begin, end, int(granularity/4))   
        ti = np.linspace(end, begin, int(granularity/4))
        inner_x = radius*np.cos(t)
        inner_y = radius*np.sin(t)
        outer_x = (radius+width)*np.cos(ti)
        outer_y = (radius+width)*np.sin(ti)
        
        right_x = np.linspace(inner_x[-1], outer_x[0], int(granularity/4))
        right_y = np.linspace(inner_y[-1], outer_y[0], int(granularity/4))
        left_x = np.linspace(outer_x[-1], inner_x[0], int(granularity/4))
        left_y = np.linspace(outer_y[-1], inner_y[0], int(granularity/4))

        retx=np.append(inner_x,[right_x,outer_x,left_x])
        rety=np.append(inner_y,[right_y,outer_y,left_y])

        return retx,rety

class Tracker:
    def __init__(self, layers, n_segments, ecl_segments, k = 2, dist = 0.2, noise = False, linewidth = 8):
        self.layers = layers
        self.noise = noise
        self.n_segments = n_segments
        self.ecl_segments = ecl_segments
        self.just_lines = []
        self.linewidths = []                                              
        self.noisemask = [] 
        self.col_names = ["radius", "begin", "end", "lines", "size", "content", "selected", "type", "facecolor", "edgecolor"]
        self.segments = pd.DataFrame(columns = self.col_names)
        counter = 0
        for l in range(1,layers+1):
            len_segment = 2*np.pi/(n_segments+k*l)
            for i in range(n_segments+k*l):
                radius = l
                begin = len_segment*i+dist/(l+1)+0.1
                end = len_segment*(i+1)-dist/(l+1)+0.1
                content = "noise" if np.random.rand()<self.noise else "empty"
                selected = "not"
                lines = get_track_line(radius, begin, end)
                size = linewidth
                self.noisemask.append(True if np.random.rand()<self.noise else False) 
                self.linewidths.append(size*2) 
                self.just_lines.append(np.array(lines))
                self.segments.loc[counter,:] = [radius, begin, end, lines, size, content, selected, "tracking", "gray", "black"]
                counter += 1
        l = layers+1
        len_segment = 2*np.pi/(self.ecl_segments)#+k*l)
        for i in range(self.ecl_segments):#+k*l):
            radius = l
            begin = len_segment*i+(dist/(l+1))
            end = len_segment*(i+1)-(dist)/(l+1)
            content = "noise" if np.random.rand()<self.noise else "empty"
            selected = "not"
            lines = get_ecl_lines(radius, begin, end, 1)
            size = 3
            self.linewidths.append(size*1.3)                      
            self.noisemask.append(True if np.random.rand()<self.noise else False) 
            self.just_lines.append(np.array(get_ecl_lines_single_array(radius, begin, end, 1)))
            self.segments.loc[counter,:] = [radius, begin, end, lines, size, content, selected, "ecl", "gray", "black"]
            counter += 1

        self.just_lines=np.array(self.just_lines).transpose((0,2,1))
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
            self.sizes.append(s*2)
        for s in self.segments.query("type=='ecl'")["size"]:
            for i in range(4):
                self.sizes.append(s)
        for s in self.segments.query("type=='tracking'")["size"]:
            self.sizes.append(s)
        
        self.segments.loc[:,"radius"] = self.segments.loc[:,"radius"].astype("float")

    def get_tracker_collection(self,truth_particles):     
        tracker = self.just_lines
        colors = ["gray"]*tracker[:,0,0].size
        linewidth = self.linewidths

        for i in range(len(truth_particles)):
            hit_lines=self.get_hit_lines(truth_particles[i])
            tracker=np.append(tracker,hit_lines,axis=0)
            colors.extend(["red"]*hit_lines[:,0,0].size)
            linewidth.extend([2.5]*hit_lines[:,0,0].size)

        noise_hits=self.just_lines[self.noisemask,:,:]
        tracker=np.append(tracker,noise_hits,axis=0)
        colors.extend(["red"]*noise_hits[:,0,0].size)
        linewidth.extend([2.5]*noise_hits[:,0,0].size)

        return LineCollection(tracker, color = colors, linewidths = linewidth)

    def get_colors(self):
        colors_tracking = self.segments.query("type=='tracking'")["edgecolor"]
        colors_ecl = self.segments.query("type=='ecl'")["facecolor"].repeat(4)
        select_mask = self.segments.loc[:,"selected"] =="selected"
        colors_ecl[select_mask] = self.segments.query("type=='ecl'")[select_mask]["edgecolor"].repeat(4)
        colors_edges= self.segments.query("type=='tracking'")["facecolor"]
        colors = pd.concat([colors_tracking, colors_ecl, colors_edges])
        return colors

    def get_collection(self):
        self.set_colors()
        line_collection = LineCollection(self.lines, color = self.get_colors(), linewidths = self.sizes)
        return line_collection

    def get_hit_lines(self, particle):
        return self.just_lines[self.check_hit(particle),:,:]
    
    def check_hit(self, particle):
        d = particle.radius*particle.charge
        a=(self.segments.radius**2)/(2*d)
        h=np.sqrt(abs(self.segments.radius**2-a**2))

        x4=(a*particle.x-h*particle.y)/d
        y4=(a*particle.y+h*particle.x)/d
        
        theta = np.arctan2(y4,x4)

        mask = theta < 0
        theta[mask] += 2*np.pi

        return (theta > self.segments.begin) & (theta < self.segments.end) & (self.segments.radius <= abs(2*particle.radius))
    
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
        self.segments.loc[self.segments["selected"]=='hidden' , "edgecolor"] = "yellow"
