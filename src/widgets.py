import ipywidgets as widgets
from matplotlib import pyplot as plt
import numpy as np
from src.particle import Particle
from src.tracker import Tracker


class TrackingWidget:
    def __init__(self, particles_df, B = 0.1):
        self.tracker = Tracker(8,1,k=2, noise = 0.1)
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
            
    def show(self):
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_ylim([-10,10])
        ax.set_xlim([-10,10])
        tracker_collection = self.tracker.get_collection()
        ax.add_collection(tracker_collection)
            
        def update(**particle_dict):
            [l.remove() for l in ax.lines]
            if particle_dict["particle"] != self.index:
                change_particle(particle_dict["particle"])
            self.index = particle_dict["particle"]
            self.tracker.segments["selected"] = "not"
            self.select_particles[self.index].radius = particle_dict[f"radius"]
            self.select_particles[self.index].phi = particle_dict[f"phi"]
            self.select_particles[self.index].charge = particle_dict["charge"]*2-1
            for j in range(self.n_particles):
                self.tracker.set_particle_selection(self.select_particles[j], hidden = True)
            self.tracker.set_particle_selection(self.select_particles[self.index], hidden = False)
            tracker_collection = self.tracker.get_collection()
            self.select_particles[self.index].draw(ax)
            ax.add_collection(tracker_collection)
            
        def change_particle(i):
            r.value = self.select_particles[i].radius
            phi.value = self.select_particles[i].phi
            charge.value = (self.select_particles[i].charge > 0 )
            print(i)

        particle = widgets.Dropdown(options = [i for i in range(self.n_particles)], value = 0, description = "Particle")
        r   = widgets.FloatSlider(self.select_particles[0].radius ,min = 0, max = 20, step = 0.1, description = "radius")
        phi = widgets.FloatSlider(self.select_particles[0].phi ,min = -np.pi, max = np.pi, step = 0.1, description = f"$\phi$")
        charge = widgets.Checkbox((self.select_particles[0].charge > 0), description = "positive charge")
        p_box = widgets.VBox([particle,r,phi, charge])

        out = widgets.interactive_output(update,{"radius": r, "phi": phi, "charge": charge, "particle": particle})
        display(p_box, out)