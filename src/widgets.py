import ipywidgets as widgets
from matplotlib import pyplot as plt
import numpy as np
from particle import Particle
from tracker import Tracker


class TrackingWidget:
    def __init__(self, particles_df, B = 0.1):
        self.tracker = Tracker(8,1,k=2, noise = 0.1)
        self.particles_df = particles_df
        self.n_particles = len(particles_df)
        self.B = B
        self.particles_df.loc[:, "radius"] = self.particles_df.loc[:,"pt"]/(self.particles_df.loc[:,"charge"]*self.B)
        self.particles = []
        for i in range(self.n_particles):
            p_df = particles_df.iloc[i]
            p = Particle(p_df["radius"], p_df["phi"], B, p_df["charge"])
            self.particles.append(p)
            self.tracker.mark_hits(p)
            
    def show(self):
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_ylim([-10,10])
        ax.set_xlim([-10,10])
        tracker_collection = self.tracker.get_collection()
        ax.add_collection(tracker_collection)
            
        def update(**particle_dict):
            i = particle_dict["particle"]
            #[l.remove() for l in ax.lines]
            ax.cla()
            ax.set_ylim([-10,10])
            ax.set_xlim([-10,10])
            self.tracker.segments["selected"] = "not"
            self.particles[i].radius = particle_dict[f"radius"]
            self.particles[i].phi = particle_dict[f"phi"]
            self.particles[i].charge = particle_dict["charge"]*2-1
            for j in range(self.n_particles):
                self.tracker.set_particle_selection(self.particles[j], hidden = True)
            self.tracker.set_particle_selection(self.particles[i], hidden = False)
            tracker_collection = self.tracker.get_collection()
            self.particles[i].draw(ax)
            ax.add_collection(tracker_collection)
            
        wids = []
        w_boxes = []
        particle = widgets.Dropdown(options = [i for i in range(self.n_particles)], value = 0, description = "Particle")
        r   = widgets.FloatSlider(10,min = 0, max = 20, step = 0.1, description = "radius")
        phi = widgets.FloatSlider(0,min = -np.pi, max = np.pi, step = 0.1, description = f"$\phi$")
        charge = widgets.Checkbox(False, description = "positive charge")
        p_box = widgets.VBox([particle,r,phi, charge])
        out = widgets.interactive_output(update,{"radius": r, "phi": phi, "charge": charge, "particle": particle})
        display(p_box, out)