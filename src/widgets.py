import ipywidgets as widgets
from matplotlib import pyplot as plt
import numpy as np
from src.particle import Particle, ToyParticle
from src.tracker import Tracker


class TrackingWidget:
    def __init__(self, particles_df, B = 0.1, noise=0.1):
        self.tracker = Tracker(8,1,k=2, noise = noise)
        self.particles_df = particles_df
        self.n_particles = len(particles_df)
        self.B = B
        self.particles_df.loc[:, "radius"] = self.particles_df.loc[:,"pt"]/(self.particles_df.loc[:,"charge"]*self.B)
        self.particles = []
        self.select_particles = []
        self.index = 0
        for i in range(self.n_particles):
            # buidl actual particles
            p_df = particles_df.iloc[i]
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
        
        