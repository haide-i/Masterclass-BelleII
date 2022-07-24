from faulthandler import disable
import ipywidgets as widgets
from matplotlib import pyplot as plt
import numpy as np
from src.particle import Particle, ToyParticle
from src.ecal import ECal
from src.tracker import Tracker
import pandas as pd

from copy import deepcopy

from  matplotlib.patches import FancyArrowPatch

from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.colors import to_rgba_array

class TrackingWidget:
    def __init__(self, data_path, B = 0.1, layers = 8, n_segments = 1, k = 2, dist = 0.2, noise = 0.2, show_truthbutton = False):
        if layers > 20:
            print("Es sind Maximal 20 Ebenen möglich!")
            layers = 20
        self.show_truthbutton = show_truthbutton
        self.particles_df = pd.read_hdf(data_path)
        self.particles_df.loc[:,'charge'] = self.particles_df.loc[:,'pdg']/abs(self.particles_df.loc[:,'pdg'])
        self.particles_df.loc[:,'phi'] = self.particles_df.loc[:,'phi']*np.pi/180
        self.particles_df.reset_index(inplace = True, drop = True)
        self.tracker = Tracker(layers = layers, n_segments = n_segments,k=k,dist=dist,noise = noise)
        self.n_particles = len(self.particles_df)
        self.B = B
        self.particles_df.loc[:, "radius"] = self.particles_df.loc[:,"pt"]/(self.particles_df.loc[:,"charge"]*self.B)
        self.particles = []
        self.select_particles = []
        self.truth_particles = []
        self.index = 0
        self.arrows_phi = []
        for i in range(self.n_particles):
            # buidl actual particles
            p_df = self.particles_df.iloc[i]
            p = Particle(p_df["radius"], p_df["phi"], B, p_df["charge"])
            self.truth_particles.append(p)
            # make an arrow to corresponding ecl crystal
            mask = self.tracker.check_hit(p)
            phi = self.tracker.segments[mask].iloc[-1][["begin", "end"]].mean()
            self.particles.append(p)
            self.tracker.mark_hits(p)
            # build dummy particles used for selection
            p = Particle(0.00001, 0, B, np.random.randint(0,1)*2-1)
            self.select_particles.append(p)
            
            self.arrows_phi.append(phi)
    
    def change_particle(self,change):
        self.index = self.tabs.selected_index
        self.update(1)

    def update(self,change):
        [l.remove() for l in self.ax.lines]
        [p.remove() for p in self.ax.patches]
        self.tracker.segments["selected"] = "not"
        for i, wphi in enumerate(self.phi):
            self.select_particles[i].phi = -wphi.value
        for i, wcharge in enumerate(self.charge):
            self.select_particles[i].charge = -1 if wcharge.value == "negative Ladung" else 1
        for i, wr in enumerate(self.r):
            self.select_particles[i].radius = wr.value/self.B
        for j in range(self.n_particles):
            self.tracker.set_particle_selection(self.select_particles[j], hidden = True)
        self.tracker.set_particle_selection(self.select_particles[self.index], hidden = False)
        tracker_collection = self.tracker.get_collection()
        if self.show_truthbutton:
            if self.truthbutton.value:
                self.truth_particles[self.index].draw(self.ax)
            else:
                self.select_particles[self.index].draw(self.ax)
        else:
            self.select_particles[self.index].draw(self.ax)

        self.ax.add_collection(tracker_collection)
        if self.n_particles > 1:
            phi = self.arrows_phi[self.index]
            r = self.tracker.layers+2
            rs = r+1
            x,y = (np.cos(phi)*r, np.sin(phi)*r)
            xs,ys = (np.cos(phi)*rs, np.sin(phi)*rs)
            arrow = FancyArrowPatch(posA = (xs,ys), posB = (x,y), mutation_scale = 10)  #FancyArrowPatch(**self.arrows[self.index])
            self.ax.add_patch(arrow)
        self.ax.legend()
            
    def show(self):
        self.fig, self.ax = plt.subplots(figsize=(10,10))
        limit = self.tracker.layers +3
        self.ax.set_ylim([-limit, limit])
        self.ax.set_xlim([-limit, limit])
        tracker_collection = self.tracker.get_collection()
        self.ax.add_collection(tracker_collection)
        self.r = []
        self.phi = []
        self.charge = []
        self.box_list = []
        if self.show_truthbutton:
            self.truthbutton = widgets.ToggleButton(value = False, description = "Zeige wahres Teilchen")
            self.truthbutton.observe(self.update, names = "value")
        for i in range(self.n_particles):
            self.r.append(widgets.FloatSlider(0 ,min = 0, max = 5, step = 0.01, description = r"$p_T$"))
            self.r[i].observe(self.update, names = "value")
            self.phi.append(widgets.FloatSlider(self.particles_df.loc[i,"phi"] ,min = -np.pi, max = np.pi, step = 0.01, description = f"$\phi$"))
            self.phi[i].observe(self.update, names = "value")
            self.charge.append(widgets.RadioButtons(options=['positive Ladung', 'negative Ladung'],  description=''))
            self.charge[i].observe(self.update, names = "value")
            if self.show_truthbutton:
                p_box = widgets.VBox([self.r[i],self.phi[i], self.charge[i], self.truthbutton])
            else:
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

    @property
    def get_fitted_particles(self):
        df = pd.DataFrame(columns = ["pt", "phi", "Ladung", "radius"])
        for i in range(self.n_particles):
            df.loc[i,:] = [self.select_particles[i].momentum(), self.select_particles[i].phi, self.select_particles[i].charge, self.select_particles[i].radius]
        df.loc[:,"px"] = np.cos(df.loc[:,"phi"].astype("float"))*df.loc[:,"pt"]
        df.loc[:,"py"] = np.sin(df.loc[:,"phi"].astype("float"))*df.loc[:, "pt"]
        df.loc[:, "pz"] = self.particles_df.loc[:,"pz"]
        return df


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
        self.B = self.b_widget.value/5
        self.particle.B = self.B

        self.particle.radius = self.pt_widget.value/self.B if self.B != 0 else 100000
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
        
        self.pt_widget= widgets.FloatSlider(1 ,min = 0.1, max = 4, step = 0.1, description = f"$p_t$")
        self.pt_widget.observe(self.update, names = "value")
        self.b_widget= widgets.Checkbox((False), description = "B-Feld")
        self.b_widget.observe(self.update, names = "value")
        self.charge_widget= widgets.Checkbox((True), description = "positive Ladung")
        self.charge_widget.observe(self.update, names = "value")
        self.out = widgets.Output()
        p_box = widgets.VBox([self.pt_widget, self.b_widget,self.charge_widget])
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

    def __init__(self, data_path, noise_rate = 0.01):
        data = pd.read_hdf(data_path)
        coords = [f'{i}' for i in np.arange(0, 6624)]
        hits = data[coords]
        hits = hits.reset_index(drop=True)
        self.ecal = ECal(144,46,hits, crystal_edge=5, noise_rate = noise_rate)   
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
    
    @property
    def get_particles_energy(self):
        energys = []
        for i in range(self.ecal.n_particles):
            particle_mask = self.ecal.select_particles.loc[i, :].to_numpy()>0
            energys.append(self.ecal.crystals_df.loc[particle_mask, "content"].sum())
        return pd.DataFrame(energys, columns = ["Energie"])


truth_particles = pd.DataFrame(columns = ["Ladung", "Masse"], data=[[1,0.0511],[-1, 0.0511]], index=["e+", "e-"])

class MatchingWidget:
    def __init__(self, ew, tw) -> None:
        self.energies = ew.get_particles_energy
        self.momenta = tw.get_fitted_particles  
        columns = ["Ladung", "Energie", "Momentum", "Masse"]
        self.res_df = pd.DataFrame(data = np.zeros((len(self.energies), len(columns))), columns = columns)
            
    def update(self, change = 0):
        sele_index = self.tabs.selected_index
        self.res_df.loc[sele_index, "Energie"] = self.energies.loc[sele_index, "Energie"]
        self.res_df.loc[sele_index, "Ladung"] = self.momenta.loc[sele_index, "Ladung"]
        self.res_df.loc[sele_index, "Momentum"] = np.sqrt((self.momenta.loc[sele_index, ["px", "py", "pz"]]**2).sum().astype("float"))
        self.res_df.loc[:, "Masse"] = np.sqrt(self.res_df.loc[:, "Energie"]**2 - self.res_df.loc[:, "Momentum"]**2)
        self.charge_comp[sele_index].value = str(self.res_df.loc[sele_index, "Ladung"] - truth_particles.loc[self.part_ids[sele_index].value, "Ladung"])
        self.mass_comp[sele_index].value = str(self.res_df.loc[sele_index, "Masse"] - truth_particles.loc[self.part_ids[sele_index].value, "Masse"])
        for i in range(len(self.res_df)):
            self.energy_txt[i].value = str(self.res_df.loc[sele_index, "Energie"])
            self.charge_txt[i].value = str(self.res_df.loc[sele_index, "Ladung"])
            self.moment_txt[i].value = str(self.res_df.loc[sele_index, "Momentum"])
            self.invmas_txt[i].value = str(self.res_df.loc[sele_index, "Masse"])

    def show(self):
        boxes = []
        self.energy_txt = []
        self.charge_txt = []
        self.moment_txt = []
        self.invmas_txt = []
        self.mass_comp = []
        self.charge_comp = []
        self.part_ids = []
        for i in range(len(self.res_df)):
            self.energy_txt.append(widgets.Text(placeholder = "kein Teilchen ausgewählt", description = "Energie", disabled = True))
            self.charge_txt.append(widgets.Text(placeholder = "kein Teilchen ausgewählt", description = "Ladung", disabled = True))
            self.moment_txt.append(widgets.Text(placeholder = "kein Teilchen ausgewählt", description = "Momentum", disabled = True))
            self.invmas_txt.append(widgets.Text(placeholder = "kein Teilchen ausgewählt", description = "Masse", disabled = True))
            self.partic_list = widgets.HTML(value= truth_particles.to_html(), description = "bekannte Teilchen")
            self.part_ids.append(widgets.Select(options = truth_particles.index, value = "e+", description = "Teilchenname"))
            self.out = widgets.Output()
            self.res_box = widgets.VBox(children=[self.energy_txt[i], self.charge_txt[i], self.moment_txt[i], self.invmas_txt[i]])
            self.mass_comp.append(widgets.Text(placeholder = "kein Teilchen ausgewählt", description = "Massendifferenz", disabled = True))
            self.charge_comp.append(widgets.Text(placeholder = "kein Teilchen ausgewählt", description = "Ladungsdifferenz", disabled = True))
            self.comb_box = widgets.VBox(children=[self.mass_comp[i], self.charge_comp[i]])
            hbox = widgets.HBox(children=[self.res_box, self.part_ids[i], self.comb_box])
            box = widgets.VBox(children=[hbox, self.partic_list])
            boxes.append(box)
        self.tabs = widgets.Tab(children=boxes)
        self.tabs.observe(self.update, "selected_index")
        for i in range(len(self.res_df)):
            self.tabs.set_title(i,f"Teilchen {i}")
        self.update()
        display(self.tabs, self.out)