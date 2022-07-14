import ipywidgets as widgets
from matplotlib import pyplot as plt
import numpy as np
from src.particle import Particle, ToyParticle
from src.ecal import ECal
from src.tracker import Tracker
import pandas as pd

from copy import deepcopy

from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.colors import to_rgba_array

class TrackingWidget:
    def __init__(self, data_path, B = 0.1, layers = 8, n_segments = 1, k = 2, noise = False):
        self.particles_df = pd.read_hdf(data_path)
        self.particles_df.loc[:,'charge'] = self.particles_df.loc[:,'pdg']/abs(self.particles_df.loc[:,'pdg'])
        self.particles_df.loc[:,'phi'] = self.particles_df.loc[:,'phi']*np.pi/180
        self.particles_df.reset_index(inplace = True, drop = True)
        self.tracker = Tracker(layers = layers, n_segments = n_segments,k=k ,noise = noise)
        self.n_particles = len(self.particles_df)
        self.B = B
        self.particles_df.loc[:, "radius"] = self.particles_df.loc[:,"pt"]/(self.particles_df.loc[:,"charge"]*self.B)
        self.particles = []
        self.select_particles = []
        self.index = 0
        for i in range(self.n_particles):
            # buidl actual particles
            p_df = self.particles_df.iloc[i]
            p = Particle(p_df["radius"], p_df["phi"], B, p_df["charge"])
            self.particles.append(p)
            self.tracker.mark_hits(p)
            # build dummy particles used for selection
            p = Particle(np.random.rand()*10, np.random.rand()*2*np.pi -np.pi, B, np.random.randint(0,1)*2-1)
            self.select_particles.append(p)
    
    def change_particle(self,change):
        self.index = self.tabs.selected_index
        self.update(1)

    def update(self,change):
        [l.remove() for l in self.ax.lines]
        self.tracker.segments["selected"] = "not"
        for i, wr in enumerate(self.r):
            self.select_particles[i].radius = wr.value
        for i, wphi in enumerate(self.phi):
            self.select_particles[i].phi = wphi.value
        for i, wcharge in enumerate(self.charge):
            self.select_particles[i].charge = wcharge.value*2-1

        for j in range(self.n_particles):
            self.tracker.set_particle_selection(self.select_particles[j], hidden = True)
        self.tracker.set_particle_selection(self.select_particles[self.index], hidden = False)
        tracker_collection = self.tracker.get_collection()
        self.select_particles[self.index].draw(self.ax)
        self.ax.add_collection(tracker_collection)
            
    def show(self):
        self.fig, self.ax = plt.subplots(figsize=(10,10))
        self.ax.set_ylim([-10,10])
        self.ax.set_xlim([-10,10])
        tracker_collection = self.tracker.get_collection()
        self.ax.add_collection(tracker_collection)
        #self.particle = widgets.Dropdown(options = [i for i in range(self.n_particles)], value = 0, description = "Particle")
        #self.particle.observe(self.change_particle, names = "value")
        self.r = []
        self.phi = []
        self.charge = []
        self.box_list = []
        for i in range(self.n_particles):
            self.r.append(widgets.FloatSlider(self.select_particles[i].radius ,min = 0, max = 20, step = 0.1, description = "radius"))
            self.r[i].observe(self.update, names = "value")
            self.phi.append(widgets.FloatSlider(self.select_particles[0].phi ,min = -np.pi, max = np.pi, step = 0.1, description = f"$\phi$"))
            self.phi[i].observe(self.update, names = "value")
            self.charge.append(widgets.Checkbox((self.select_particles[0].charge > 0), description = "positive charge"))
            self.charge[i].observe(self.update, names = "value")
            p_box = widgets.VBox([self.r[i],self.phi[i], self.charge[i]])
            self.box_list.append(p_box)
        self.tabs = widgets.Tab()
        self.tabs.children = self.box_list
        for i in range(self.n_particles):
            self.tabs.set_title(i,f"particle {i}")
        self.tabs.observe(self.change_particle, names = "selected_index")
        self.out = widgets.Output()
        display(self.tabs, self.out)  
        self.update(1)            



class TestDetektor:
    def __init__(self, B=0.1, layers=8, n_segments=1, k=2):
        self.tracker = Tracker(layers = layers, n_segments = n_segments,k=k ,noise = False)
        self.B = B
        self.particle = Particle(1, 0, B,-1)
        self.pt = 10

    def update(self,change):
        [l.remove() for l in self.ax.lines]
        self.tracker.segments["content"] = "empty"
        self.particle.charge = self.charge_widget.value*2-1
        self.B = self.b_widget.value
        self.particle.B = self.B

        self.particle.radius = self.pt/self.B if self.B != 0 else 100000
        self.tracker.mark_hits(self.particle)
        tracker_collection = self.tracker.get_collection()
        self.particle.draw(self.ax)
        self.ax.add_collection(tracker_collection)

    def show(self):
        self.fig, self.ax = plt.subplots(figsize=(7,7))
        lim = self.tracker.layers*1.5
        self.ax.set_ylim([-lim,lim])
        self.ax.set_xlim([-lim,lim])
        tracker_collection = self.tracker.get_collection()
        self.ax.add_collection(tracker_collection)
        
        self.b_widget = widgets.FloatSlider(1 ,min = 0.000001, max = 10, step = 0.1, description = "B-Feld")
        self.b_widget.observe(self.update, names = "value")
        self.charge_widget= widgets.Checkbox((True), description = "positive charge")
        self.charge_widget.observe(self.update, names = "value")
        self.out = widgets.Output()
        p_box = widgets.VBox([self.b_widget,self.charge_widget])
        display(p_box, self.out)  
        self.update(1)    
        self.update(1)   

class ptWidget:
    def __init__(self,  noise=0.0, nlayers=8,nsegments=1, axlim=[-10,10]):
        self.tracker = Tracker(nlayers,nsegments,k=2, noise = noise)
        self.B = 0
        self.axlim=axlim
        p = ToyParticle(1., 0, self.B, np.random.randint(0,1)*2-1)
        self.select_particles=[p]
        self.n_particles=1
        self.index=0
        self.scale_factor=5
    def show(self):
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_ylim(self.axlim)
        ax.set_xlim(self.axlim)
        tracker_collection = self.tracker.get_collection()
        ax.add_collection(tracker_collection)
            
        def update(**particle_dict):
            [l.remove() for l in ax.lines]
            if particle_dict["particle"] != self.index:
                change_particle(particle_dict["particle"])
                
            self.index = particle_dict["particle"]
            self.tracker.segments["selected"] = "not"
            self.select_particles[self.index].charge = particle_dict["charge"]*2-1
            self.select_particles[self.index].momentum = particle_dict["p_t"]
            if particle_dict[f"B"]!=0.:
                self.select_particles[self.index].radius= particle_dict[f"p_t"]/(self.select_particles[self.index].charge*particle_dict[f"B"]/self.scale_factor)
            else:
                self.select_particles[self.index].radius=30000.
            self.select_particles[self.index].B = particle_dict[f"B"]/self.scale_factor
            for j in range(self.n_particles):
                self.tracker.set_particle_selection(self.select_particles[j], hidden = True)
            self.tracker.set_particle_selection(self.select_particles[self.index], hidden = False)
            tracker_collection = self.tracker.get_collection()
            self.select_particles[self.index].draw(ax)
            ax.add_collection(tracker_collection)
            
        def change_particle(i):
            #r.value = self.select_particles[i].radius
            #phi.value = self.select_particles[i].phi
            charge.value = (self.select_particles[i].charge > 0 )
            print(i)

        particle = widgets.Dropdown(options = [i for i in range(self.n_particles)], value = 0, description = "Particle")
        pt   = widgets.FloatSlider(self.select_particles[0].momentum ,min = 0, max = 5, step = 0.1, description = f"$p_t$")
        B = widgets.FloatSlider(self.select_particles[0].B ,min = 0, max = 4, step = 0.1, description = f"$B$")
        charge = widgets.Checkbox((self.select_particles[0].charge > 0), description = "positive Ladung")
        p_box = widgets.VBox([particle, pt,B, charge])

        out = widgets.interactive_output(update,{"p_t": pt, "B": B, "charge": charge, 'particle':particle})
        display(p_box, out)
  
class ECLWidget:

    def __init__(self, data_path):
        data = pd.read_hdf(data_path)
        coords = [f'{i}' for i in np.arange(0, 6624)]
        hits = data[coords]
        hits = hits.reset_index(drop=True)
        self.ecal = ECal(144,46,hits, crystal_edge=5, noise = 0.01)   
        content = deepcopy(self.ecal.crystals_df["content"])
        #content = np.log(content)
        self.alphas = np.clip(content,0.25,1)
        #self.subplot_kw = dict(xlim=(-5,725), ylim=(-5,235), autoscale_on=False)
        
        fig, ax = plt.subplots(figsize=(16,9))#, subplot_kw=self.subplot_kw, dpi=400)
        ax.add_collection(self.ecal.collection)
        self.crystall_points = ax.scatter(self.ecal.crystals_df["x"], self.ecal.crystals_df["y"], s=0)
        self.xys = self.crystall_points.get_offsets()
        self.Npts = len(self.xys)
        self.canvas = ax.figure.canvas
        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []
        self.particle_index = 0
        
    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.ecal.select_particles.loc[self.particle_index, :] = 0
        self.ecal.select_particles.loc[self.particle_index, self.ind.astype(str)] = 1
        self.ecal.set_colors(self.particle_index)
        facecolors = to_rgba_array(self.ecal.crystals_df.loc[:,"facecolor"].to_numpy())
        content_mask = (self.ecal.crystals_df["content"]>0).to_numpy()
        facecolors.T[-1] = 0.5 
        facecolors[content_mask,-1] = self.alphas[content_mask]
        edgecolors = to_rgba_array(self.ecal.crystals_df.loc[:,"edgecolor"].to_numpy())
        self.ecal.collection.set_edgecolors(edgecolors)
        self.ecal.collection.set_facecolors(facecolors)
        self.canvas.draw_idle()
        particle_mask = self.ecal.select_particles.loc[self.particle_index, :].to_numpy()>0
        energy = self.ecal.crystals_df.loc[particle_mask, "content"].sum()
        self.energy_label.value = f"Energy of selected Cluster: {str(round(energy,4))} GeV"
        
    def change_particle(self,change):
        self.particle_index = self.particle.value
        self.onselect([(0,0)])
        
    def show(self):
        self.particle = widgets.Dropdown(options = [i for i in range(self.ecal.n_particles)], value = 0, description = "Particle")
        self.particle.observe(self.change_particle, names = "value")
        self.energy_label = widgets.Label("Energy of selected Cluster: 0 GeV")
        self.box = widgets.HBox([self.particle, self.energy_label])
        self.out = widgets.Output()
        display(self.box, self.out)
        self.onselect([(0,0)])
        plt.show()
        
               
        
        