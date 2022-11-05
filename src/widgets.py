from faulthandler import disable
import ipywidgets as widgets
from matplotlib import pyplot as plt
import numpy as np
from src.particle import Particle
from src.ecal import ECal
from src.tracker import Tracker
import pandas as pd

from copy import deepcopy

from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.colors import to_rgba_array

def make_box_layout():
     return widgets.Layout(
        border='solid 1px black',
        margin='0px 10px 10px 0px',
        padding='5px 5px 5px 5px',
        height = "1000px",
        width = "1000px"
     )

def get_arrow(posx,posy,phi,size=1): #returns (100,2) x,y array of an arrow at position x,y pointing at angle phi 
    x=(np.array([0,0,-2,2,0,0])*np.cos(phi)-np.array([-3,3,0,0,3,-3])*np.sin(phi))*size
    y=(np.array([0,0,-2,2,0,0])*np.sin(phi)+np.array([-3,3,0,0,3,-3])*np.cos(phi))*size
    arrow = np.append(np.array([x,y]).T,np.zeros((94,2)),axis=0)
    arrow = arrow + [posx,posy]
    return arrow

class BlitManager: #manages the blitting for tracker and ecal widget
    def __init__(self, canvas, artist):
        """copy from matplotlib website (blitting)"""
        self.canvas = canvas
        self._bg = None
        self.artist = artist

        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()


    def _draw_animated(self):
        fig = self.canvas.figure
        fig.draw_artist(self.artist)

    def update(self):
        cv = self.canvas
        fig = cv.figure
        if self._bg is None:
            self.on_draw(None)
        else:
            cv.restore_region(self._bg)
            self._draw_animated()
            cv.blit(fig.bbox)
        cv.flush_events()

class TrackingWidget:
    def __init__(self, data_path, B = 0.2, layers = 15, n_segments = 5, ecl_segments = 30, k = 3, dist = 0.1, noise = 0.05, linewidth = 5, show_truthbutton = False, continuous_update=True, truthvalues=False, ignore_noise=False ):
        self.continuous_update=continuous_update
        self.truthvalues=truthvalues
        self.granularity=100      #must be multiple of 4
        self.show_truthbutton = show_truthbutton
        self.particles_df = pd.read_hdf(data_path)
        self.particles_df.loc[:,'charge'] = self.particles_df.loc[:,'pdg']/abs(self.particles_df.loc[:,'pdg'])
        self.particles_df.reset_index(inplace = True, drop = True)
        self.tracker = Tracker(layers = layers, n_segments = n_segments, ecl_segments=ecl_segments, k=k,dist=dist, noise = noise, linewidth = linewidth, ignore_noise = ignore_noise,granularity=self.granularity)
        self.n_particles = len(self.particles_df)
        self.B = B*15/layers
        self.particles_df.loc[:, "radius"] = self.particles_df.loc[:,"pt"]/self.B
        self.particles = []
        self.select_particles = []
        self.truth_particles = []
        self.index = 0
        self.arrows = []
        for i in range(self.n_particles):
            # build actual particles
            p_df = self.particles_df.iloc[i]
            p = Particle(p_df["radius"], p_df["phi"], B, p_df["charge"], granularity=self.granularity)
            self.truth_particles.append(p)

            # make an arrow to corresponding ecl crystal
            arrow_phi = self.tracker.get_arrowangle(p)
            arrow_x = np.cos(arrow_phi)*(self.tracker.layers+2.8)
            arrow_y = np.sin(arrow_phi)*(self.tracker.layers+2.8)
            self.arrows.append(get_arrow(posx=arrow_x,posy=arrow_y,phi = arrow_phi + np.pi/2,size=0.18))

            # build dummy particles used for selection
            p = Particle(0.00001, 0, B, np.random.randint(0,1)*2-1)
            self.select_particles.append(p)

        self.tracker.make_tracker_mask(self.truth_particles)    
            
    def change_particle(self,change):
        self.index = self.tabs.selected_index
        self.update(1)

    def update(self,change):

        if self.index is None:
            drawtrace = False  
        else:   #alles mit self.index kann nur abgefragt werden, wenn self.index nicht nonetype ist  
            self.select_particles[self.index].phi = self.phi[self.index].value+self.phi_fine[self.index].value
            self.select_particles[self.index].charge = -1 if self.charge[self.index].value == "negative Ladung" else 1
            self.select_particles[self.index].radius = (self.r[self.index].value+self.r_fine[self.index].value)/self.B
            self.r_label[self.index].value = str(round(self.select_particles[self.index].radius*self.B,6))
            self.phi_label[self.index].value = str(round(self.select_particles[self.index].phi,6))

            drawtrace = True
            if self.show_truthbutton:
                if self.truthbutton.value:
                    trace=self.truth_particles[self.index].trace_array()
                else:
                    trace=self.select_particles[self.index].trace_array()
            else:
                trace=self.select_particles[self.index].trace_array()

            hits,misses=self.tracker.get_hits_and_misses(self.select_particles[self.index],self.index)
            self.hit_n_misses[self.index].value = str(hits) + " hits & " + str(misses) + " misses"
        
        segments,colors=self.tracker.get_hit_lines(self.select_particles,self.index)
        if drawtrace == True:
            segments=np.append(segments,[trace.T],axis=0)
            segments=np.append(segments,[self.arrows[self.index]],axis=0)
            colors=np.append(colors,["blue","green"])

        self.artist.set_segments(segments)
        self.artist.set_color(colors)
        self.bm.update()

            
    def show(self):
        self.out = widgets.Output()
        with self.out:
            self.fig, self.ax = plt.subplots(figsize=(7,7),constrained_layout=True)

        limit = self.tracker.layers +3
        self.ax.set_xlim(-limit,limit)
        self.ax.set_ylim(-limit,limit)

        artist = self.ax.add_collection(LineCollection([]))
        self.artist=artist
        self.artist.set_animated(True)
        self.fig.canvas.toolbar_position = 'left'
        self.ax.add_collection(self.tracker.get_tracker_collection())
        self.bm = BlitManager(self.fig.canvas , self.artist)

        self.hit_n_misses = []
        self.r_label = []
        self.r = []
        self.r_fine = []
        self.phi_label = []
        self.phi = []
        self.phi_fine = []
        self.charge = []
        self.box_list = []
        if self.show_truthbutton:
            self.truthbutton = widgets.ToggleButton(value = False, description = "Zeige wahres Teilchen")
            self.truthbutton.observe(self.update, names = "value")
        for i in range(self.n_particles):
            self.hit_n_misses.append(widgets.Text(description = "", value = "0 hits & 0 misses", disabled=True))
            self.r_label.append(widgets.Text(description = "$p_T$:", value = "0", disabled=True))
            self.r.append(widgets.FloatSlider(self.particles_df.loc[i,"pt"] if self.truthvalues == True else 0,min = 0, max = 5, step = 0.01, description = "$p_T$",continuous_update=self.continuous_update))
            self.r[i].observe(self.update, names = "value")
            self.r_fine.append(widgets.FloatSlider(0 ,min = 0, max = 0.2, step = 0.001, description = "$p_T fine$",continuous_update=self.continuous_update))
            self.r_fine[i].observe(self.update, names = "value")
            self.phi_label.append(widgets.Text(description = "$\phi$:", value = "0", disabled=True))
            self.phi.append(widgets.FloatSlider(self.particles_df.loc[i,"phi"] if self.truthvalues == True else 0,min = -np.pi, max = np.pi, step = 0.01, description = "$\phi$",continuous_update=self.continuous_update))
            self.phi[i].observe(self.update, names = "value")
            self.phi_fine.append(widgets.FloatSlider(0 ,min = -0.15, max = 0.15, step = 0.001, description = "$\phi fine$",continuous_update=self.continuous_update))
            self.phi_fine[i].observe(self.update, names = "value")
            self.charge.append(widgets.RadioButtons(options=['positive Ladung', 'negative Ladung'],  description=''))
            self.charge[i].observe(self.update, names = "value")
            if self.show_truthbutton:
                p_box = widgets.VBox([self.hit_n_misses[i],self.r_label[i],self.r[i], self.r_fine[i],self.phi_label[i], self.phi[i], self.phi_fine[i], self.charge[i], self.truthbutton])
            else:
                p_box = widgets.VBox([self.hit_n_misses[i],self.r_label[i],self.r[i], self.r_fine[i],self.phi_label[i], self.phi[i], self.phi_fine[i], self.charge[i]])
            self.box_list.append(p_box)
        
        self.tabs = widgets.Accordion()
        self.tabs.children = self.box_list
        for i in range(self.n_particles):
            self.tabs.set_title(i,f"particle {i}")
        self.tabs.observe(self.change_particle, names = "selected_index")
        self.tabs_box = widgets.HBox([self.tabs])
        self.tabs_box.layout =widgets.Layout(
                                border='solid 1px black',
                                margin='0px 10px 10px 0px',
                                padding='5px 5px 5px 5px',
                                height = "750px ",
                                width = "500px"
                            )
        self.plot_box = widgets.HBox([self.out])
        self.plot_box.layout = widgets.Layout(
                                border='solid 1px black',
                                margin='0px 10px 10px 0px',
                                padding='5px 5px 5px 5px',
                                height = "750px ",
                                width = "750px"
                            )
        self.final_box = widgets.HBox([  self.tabs_box,self.plot_box])
        with self.out:
            plt.show()
        #plt.pause(.1)
        self.fig.canvas.draw() 
        display(self.final_box)  
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
    def __init__(self, B=0.1, layers=8, n_segments=1,ecl_segments=10, k=2):
        if layers > 20:
            print("Es sind maximal 20 Schichten möglich!")
            layers = 20
        self.tracker = Tracker(layers = layers, n_segments = n_segments,k=k ,ecl_segments=ecl_segments,noise = False, linewidth = 4)
        self.B = B
        self.particle = Particle(1, 0, B,-1)
        self.pt = 10

    def update(self,change):
        [l.remove() for l in self.ax.lines]
        self.tracker.segments["content"] = "empty"
        self.particle.charge = -1 if self.charge_widget.value == "negative Ladung" else 1
        
        #self.particle.charge = self.charge_widget.value*2-1
        self.B = self.b_widget.value/5
        self.particle.B = self.B

        self.particle.radius = self.pt_widget.value/self.B if self.B != 0 else 100000
        self.particle.draw(self.ax)
        self.tracker.mark_hits(self.particle)
        tracker_collection = self.tracker.get_collection()
        
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
        self.charge_widget= widgets.RadioButtons(options=['positive Ladung', 'negative Ladung'],  description='')
        self.charge_widget.observe(self.update, names = "value")
        self.out = widgets.Output()
        p_box = widgets.VBox([self.pt_widget, self.b_widget,self.charge_widget])
        plt.show()
        display(self.out, p_box)
        self.update(1)    
        self.update(1)   

  
class ECLWidget:

    def __init__(self, data_path, noise_rate = 0.05, idx=None):
        data = pd.read_hdf(data_path)
        coords = [f'{i}' for i in np.arange(0, 6624)]
        if idx is None:
            hits = data[coords]
        else:
            hits = data[coords].iloc[idx:(idx+1)]
        hits = hits.reset_index(drop=True)
        self.edge_size = 5
        self.ecal = ECal(144,46,hits, crystal_edge = self.edge_size, noise_rate = noise_rate)   
        content = deepcopy(self.ecal.crystals_df["content"])
        #content = np.log(content)
        self.alphas = np.clip(content,0.25,1)
        
        self.out = widgets.Output()
        with self.out:
            fig, ax = plt.subplots(figsize=(15,6),constrained_layout=True)
        ax.set_ylim(-10,46*5+10)
        ax.set_xlim(-10,144*5+10)
        self.artist = ax.add_collection(self.ecal.collection)
        self.xys = np.array([self.ecal.crystals_df["x"] + self.edge_size/2,self.ecal.crystals_df["y"] + self.edge_size/2],dtype = "float64").T
        self.Npts = len(self.xys)
        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.artist.set_animated(True)
        self.bm_ecal = BlitManager(fig.canvas , self.artist)
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
        self.bm_ecal.update()
        particle_mask = self.ecal.select_particles.loc[self.particle_index, :].to_numpy()>0
        energy = self.ecal.crystals_df.loc[particle_mask, "content"].sum()
        self.energy_labels[self.particle_index].value = str(round(energy,4))
        
    def change_particle(self,change):
        if self.particle.selected_index is not None:
            self.particle_index = self.particle.selected_index
        self.onselect([(0,0)])
        
    def show(self):
        self.particle = widgets.Accordion()
        self.particle.observe(self.change_particle, names = "selected_index")
        self.energy_labels = []
        self.box_list = []
        for i in range(self.ecal.n_particles):
            self.particle.set_title(i, f"Teilchen {i}")
            self.energy_labels.append(widgets.Text(description = "Gesamte Energie der ausgewählten Kristalle in GeV:", value = "0", disabled=True))
            self.box_list.append(widgets.HBox([self.energy_labels[i]]))
        self.particle.children = self.box_list
        self.final_box = widgets.VBox(children=[self.particle, self.out])
        with self.out:
            plt.show()
        display(self.final_box)
        self.onselect([(0,0)])
    
    @property
    def get_particles_energy(self):
        energys = []
        for i in range(self.ecal.n_particles):
            particle_mask = self.ecal.select_particles.loc[i, :].to_numpy()>0
            energys.append(self.ecal.crystals_df.loc[particle_mask, "content"].sum())
        return pd.DataFrame(energys, columns = ["Energie"])
    
    @property
    def get_particles_radius(self):
        radius = []
        for i in range(self.ecal.n_particles):
            particle_mask = self.ecal.select_particles.loc[i, :].to_numpy()>0
            selected_crystals = self.ecal.crystals_df.loc[particle_mask]
            selected_hits = selected_crystals.query("content>0").index
            xdiff = abs(int(self.ecal.center[i]/144.)%46 - (selected_hits/144).astype('int')%46)
            ydiff = abs(self.ecal.center[i]%144 - selected_hits%144)
            radius.append(max(xdiff.max(), ydiff.max()))
        return pd.DataFrame(radius, columns= ["Radius"])






true_particle_data = [[0.511, 1],
                      [0.511, -1],
                      [105., +1],
                     [105., -1],
                     [1776., +1],
                     [1776., -1],
                     [938.3, +1],
                     [938.3, -1],
                     [939.6, 0],
                      [135, 0],
                     [139.6, +1],
                      [139.6, -1],
                      [497.6, 0],
                     [493.7, +1],
                      [493.7, -1]]
true_particle_names = ["e+", "e-", "mu+", "mu-", "tau+", "tau-", "Proton", "Antiproton", "Neutron", "pi0", "pi+", "pi-", "K0", "K+", "K-"]
truth_particles = pd.DataFrame(columns = ["Masse", "Ladung"], data=true_particle_data, index=true_particle_names)
truth_particles.loc[:, "Masse"] = truth_particles["Masse"]*10**(-3)

class MatchingWidget:
    def __init__(self, ew, tw) -> None:
        self.energies = ew.get_particles_energy
        self.radius = ew.get_particles_radius
        self.momenta = tw.get_fitted_particles  
        columns = ["Ladung", "Energie", "Momentum", "Masse", "Radius"]
        self.true_df = tw.particles_df
        self.res_df = pd.DataFrame(data = np.zeros((len(self.energies), len(columns))), columns = columns)
        #self.res_df.loc[:,"pt"] = np.sqrt(( self.momenta.loc[:,["px", "py"]]**2).sum())
        self.diff_mask = ((self.true_df["pt"]-self.momenta["pt"])<1e-2).to_numpy()
        self.momenta.loc[self.diff_mask, ["px", "py", "pz"]] =  self.true_df.loc[self.diff_mask, ["px", "py", "pz"]]
        self.energies.loc[self.diff_mask, "Energie"] = self.true_df.loc[self.diff_mask, "energy"]   

            
    def update(self, change = 0):
        sele_index = self.tabs.selected_index
        self.res_df.loc[sele_index, "Energie"] = self.energies.loc[sele_index, "Energie"]
        self.res_df.loc[sele_index, "Radius"] = np.nan_to_num(self.radius.loc[sele_index, "Radius"])
        self.res_df.loc[sele_index, "Ladung"] = self.momenta.loc[sele_index, "Ladung"]
        self.res_df.loc[sele_index, "Momentum"] = np.sqrt((self.momenta.loc[sele_index, ["px", "py", "pz"]]**2).sum().astype("float"))
        # if self.res_df.loc[:, "Energie"] > self.res_df.loc[:, "Momentum"]:
        self.res_df.loc[:, "Masse"] = np.sqrt(self.res_df.loc[:, "Energie"]**2 - self.res_df.loc[:, "Momentum"]**2)
        self.res_df.loc[:, "Masse"] = self.res_df.loc[:, "Masse"].fillna(0)
        self.charge_comp[sele_index].value = str(self.res_df.loc[sele_index, "Ladung"] - truth_particles.loc[self.part_ids[sele_index].value, "Ladung"])
        self.mass_comp[sele_index].value = str(self.res_df.loc[sele_index, "Masse"] - truth_particles.loc[self.part_ids[sele_index].value, "Masse"])
        for i in range(len(self.res_df)):
            self.energy_txt[i].value = str(self.res_df.loc[sele_index, "Energie"])
            self.charge_txt[i].value = str(self.res_df.loc[sele_index, "Ladung"])
            self.moment_txt[i].value = str(self.res_df.loc[sele_index, "Momentum"])
            self.invmas_txt[i].value = str(self.res_df.loc[sele_index, "Masse"])
            self.radius_txt[i].value = str(self.res_df.loc[sele_index, "Radius"])
            self.px_txt[i].value = str(self.momenta.loc[sele_index, "px"])
            self.py_txt[i].value = str(self.momenta.loc[sele_index, "py"])
            self.pz_txt[i].value = str(self.momenta.loc[sele_index, "pz"])

    def show(self):
        boxes = []
        self.energy_txt = []
        self.px_txt = []
        self.py_txt = []
        self.pz_txt = []
        self.charge_txt = []
        self.moment_txt = []
        self.invmas_txt = []
        self.radius_txt = []
        self.mass_comp = []
        self.charge_comp = []
        self.part_ids = []
        for i in range(len(self.res_df)):
            self.px_txt.append(widgets.Text(placeholder = "kein Teilchen ausgewählt", description = "$p_x$", disabled = True))
            self.py_txt.append(widgets.Text(placeholder = "kein Teilchen ausgewählt", description = "$p_y$", disabled = True))
            self.pz_txt.append(widgets.Text(placeholder = "kein Teilchen ausgewählt", description = "$p_z$", disabled = True))
            self.energy_txt.append(widgets.Text(placeholder = "kein Teilchen ausgewählt", description = "Energie", disabled = True))
            self.charge_txt.append(widgets.Text(placeholder = "kein Teilchen ausgewählt", description = "Ladung", disabled = True))
            self.moment_txt.append(widgets.Text(placeholder = "kein Teilchen ausgewählt", description = "Momentum", disabled = True))
            self.invmas_txt.append(widgets.Text(placeholder = "kein Teilchen ausgewählt", description = "Masse", disabled = True))
            self.radius_txt.append(widgets.Text(placeholder = "kein Teilchen ausgewählt", description = "Radius", disabled = True))
            self.partic_list = widgets.HTML(value= truth_particles.to_html(), description = "bekannte Teilchen")
            self.part_ids.append(widgets.Select(options = truth_particles.index, value = "e+", description = "Teilchenname"))
            self.part_ids[i].observe(self.update, "value")
            self.out = widgets.Output()
            self.res_box = widgets.VBox(children=[self.energy_txt[i], self.charge_txt[i], self.moment_txt[i], self.invmas_txt[i], self.radius_txt[i]])
            self.vec_box = widgets.VBox(children=[self.energy_txt[i], self.px_txt[i], self.py_txt[i], self.pz_txt[i]])
            self.mass_comp.append(widgets.Text(placeholder = "kein Teilchen ausgewählt", description = "Massendifferenz", disabled = True))
            self.charge_comp.append(widgets.Text(placeholder = "kein Teilchen ausgewählt", description = "Ladungsdifferenz", disabled = True))
            self.comb_box = widgets.VBox(children=[self.mass_comp[i], self.charge_comp[i]])
            hbox = widgets.HBox(children=[self.res_box, self.part_ids[i], self.comb_box])
            hbox2 = widgets.HBox(children=[self.partic_list,self.vec_box])
            box = widgets.VBox(children=[hbox, hbox2])
            boxes.append(box)
        self.tabs = widgets.Tab(children=boxes)
        self.tabs.observe(self.update, "selected_index")
        for i in range(len(self.res_df)):
            self.tabs.set_title(i,f"Teilchen {i}")
            # self.mass_comp[i].value = str(
        self.update()
        display(self.tabs, self.out)
    
class MissingWidget():

    def calc_missing_part(self, dummy):
        fourvecs = np.zeros((4,4))
        for i in range(4):
            for j in range(4):
                fourvecs[i,j] = self.boxes[i].children[j+1].value
        missing_part = np.array([15.580,0,0,0])-fourvecs.sum(0)
        print(missing_part)
        for j in range(4):
            self.boxes[4].children[j+1].value = missing_part[j]
        self.boxes[4].children[5].value = np.sqrt(((missing_part*np.array([1,-1,-1,-1]))**2).sum())


    def show(self):
        self.boxes = []
        for i in range(4):
            w_label1 = widgets.Label(value=f"Teilchen {i}")
            w_energy1 = widgets.FloatText(description="Energie", placeholder = "kein Eintrag", value = 0., layout = widgets.Layout(width="200px"))
            w_px1 = widgets.FloatText(description="$p_x$", placeholder = "kein Eintrag", value = 0., layout = widgets.Layout(width="200px"))
            w_py1 = widgets.FloatText(description="$p_y$", placeholder = "kein Eintrag", value = 0., layout = widgets.Layout(width="200px"))
            w_pz1 = widgets.FloatText(description="$p_z$", placeholder = "kein Eintrag", value = 0., layout = widgets.Layout(width="200px"))
            t1_box = widgets.VBox(children=[w_label1, w_energy1, w_px1, w_py1, w_pz1])
            self.boxes.append(t1_box)
        w_label1 = widgets.Label(value=f"fehlendes Teilchen")
        w_energy1 = widgets.FloatText(description="Energie", placeholder = "kein Eintrag", disabled = True, layout = widgets.Layout(width="200px"))
        w_px1 = widgets.FloatText(description="$p_x$", placeholder = "kein Eintrag", disabled = True, layout = widgets.Layout(width="200px"))
        w_py1 = widgets.FloatText(description="$p_y$", placeholder = "kein Eintrag", disabled = True, layout = widgets.Layout(width="200px"))
        w_pz1 = widgets.FloatText(description="$p_z$", placeholder = "kein Eintrag", disabled = True, layout = widgets.Layout(width="200px"))
        w_mass = widgets.FloatText(description="Masse",  placeholder = "kein Eintrag", disabled = True)
        t1_box = widgets.VBox(children=[w_label1, w_energy1, w_px1, w_py1, w_pz1, w_mass])
        self.boxes.append(t1_box)
        button = widgets.Button(description="Berechne fehlendes Teilchen")
        button.on_click(self.calc_missing_part)
        self.boxes.append(button)
        self.final_box = widgets.HBox(children=self.boxes)
        display(self.final_box)