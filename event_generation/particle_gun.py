import basf2
import b2luigi as luigi
from b2luigi.basf2_helper import Basf2PathTask
from ROOT import Belle2

import os
import numpy as np
import pandas as pd
from pandas import IndexSlice as idx

from fourvector_generation import FourVecGenTask
from events import events_dict, base_path

try:
    from simulation import add_simulation
    from reconstruction import add_reconstruction
    from mdst import add_mdst_output
except ImportError:
    # Output expected ImportErrors.
    print("Import Error")

class getECLInfo(basf2.Module):
    
    def __init__(self, output, pdg, theta, phi, p, px, py, pz, pt, E):
        super().__init__()
        self.output = output
        self.obj_eclgeometrypar = Belle2.ECL.ECLGeometryPar.Instance()
        self.barrel = np.arange(1153, 7777, dtype=int)
        self.col_names = [f'{i}' for i in self.barrel]
        self.pdg = pdg
        self.theta = theta
        self.phi = phi
        self.p = p
        self.px = px
        self.py = py
        self.pz = pz
        self.pt = pt
        self.E = E

    def initialize(self):
        self.eventinfo = Belle2.PyStoreObj('EventMetaData')
        self.mcparticles = Belle2.PyStoreArray('MCParticles')
        self.eclcaldigits = Belle2.PyStoreArray('ECLCalDigits')
        self.eclclusters = Belle2.PyStoreArray('ECLClusters')
        self.ecllocalmaxima = Belle2.PyStoreArray('ECLLocalMaximums')
        
        self.index = 0

        self.tot_data = pd.DataFrame()

    def event(self):

        correct_pdg = 0
        correct_mass = 0
        tot_energy = 0

        energyinbarrel = False
        ignore_event = False
        correctpdg = False

        print("in event: ", self.p, self.theta)

        if len(self.mcparticles) == 1:
            ignore_event = True

        if not ignore_event:
            

            cells = dict.fromkeys(self.barrel, 0)

            for caldigit in self.eclcaldigits:
                rec_energy = caldigit.getEnergy()

                ids = caldigit.getCellId()

                if ids in cells:

                    cells[ids] = rec_energy
                    energyinbarrel = True

            if energyinbarrel:

                for mc_idx, mc_particle in enumerate(self.mcparticles):
                    mcrelations = mc_particle.getRelationsWith('ECLCalDigits')

                    pdg = mc_particle.getPDG()
                    print(pdg)
                    
                    if int(pdg) == int(self.pdg):
                        correctpdg = True
                        mass = mc_particle.getMass()
                        energy = mc_particle.getEnergy()

                        for mc_id in range(mcrelations.size()):
                            # mc_energy = mcrelations.weight(mc_id)
                            id = mcrelations.object(mc_id).getCellId()

                            if id in cells:
                                correct_pdg = pdg
                                correct_mass = mass
                                tot_energy = energy

                if correctpdg:
                
                    all_energy = [[e for e in cells.values()]]
                    print(np.shape(np.array(all_energy)))
                                
                    data = pd.DataFrame(columns=self.col_names, data=all_energy)
                    
                    data['event'] = self.index
                    data['pdg'] = correct_pdg
                    data['mass'] = correct_mass
                    data['energy'] = tot_energy
                    data['theta'] = self.theta.item()
                    data['phi'] = self.phi.item()
                    data['p'] = self.p.item()
                    data['px'] = self.px.item()
                    data['py'] = self.py.item()
                    data['pz'] = self.pz.item()
                    data['pt'] = self.pt.item()

                    self.tot_data = pd.concat([self.tot_data, data])
        self.index += 1

    def terminate(self):
        
        if len(self.tot_data>0):
            print(self.tot_data['mass'])
            self.tot_data = self.tot_data[self.tot_data['pdg']!=0]
            self.tot_data.to_csv(self.output+'.csv')
        else:
            print("no matching events generated")
            raise RuntimeError 

class EvtGenTask(Basf2PathTask):
    name = luigi.Parameter()
    index = luigi.Parameter()

    batch_system = "local"


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pdgs = events_dict[self.name]

    @property
    def input_dict(self):
        self._input_dict = None
        if self._input_dict is None:
            self._input_dict = {k: v.path for d in self.input() if isinstance(d, dict) for k, v in d.items()}
        return self._input_dict

    def read_input(self):
        input_key = f"{self.name}_fourvec_generation"
        return pd.read_csv(self.input_dict[input_key], header = [0], index_col=0)
    

    def get_log_file_dir(self):
        path = os.path.join(base_path, "particle_gun", self.name, str(self.index))
        os.makedirs(path, exist_ok=True)
        return path

    def create_path(self):
        main = basf2.create_path()
        df = self.read_input()
        particle = df.iloc[self.index]
        print(particle)
        particlegun = basf2.register_module('ParticleGun')
        particlegun.param('pdgCodes', [int(particle['pdg'])])

        particlegun.param('momentumGeneration', 'fixed')
        particlegun.param('momentumParams', [particle['p']])

        particlegun.param('thetaGeneration', 'fixed')
        particlegun.param('thetaParams', [particle['theta']*180/np.pi])

        particlegun.param('phiGeneration', 'fixed')
        particlegun.param('phiParams', [particle['phi']*180/np.pi])

        main.add_module(particlegun)

        main.add_module('EventInfoSetter',
                        evtNumList=10)

        main.add_module('Gearbox')
        main.add_module('Geometry', useDB=True)

        add_simulation(path=main)

        add_reconstruction(path=main)
    
        ecl_output = getECLInfo(self.output_file(),
                                particle['pdg'],
                                particle['theta'],
                                particle['phi'],
                                particle['p'],
                                particle['px'],
                                particle['py'],
                                particle['pz'],
                                particle['pt'],
                                particle['E']
                                )

        main.add_module(ecl_output)

        return main

    def process(self):
        try:
            import basf2
            import ROOT
        except ImportError:
            raise ImportError("Cannot find basf2 or ROOT")

        path = self.create_path()

        path.add_module('Progress')
        #basf2.print_path(path)
        basf2.process(path)

        #print(basf2.statistics)
    
    @property
    def output_path(self):
        return os.path.join(base_path, "particle_gun")
    
    def output_file(self):
        return os.path.join(self.output_path, self.name, f"{self.index}")

    
    def output(self):
        output_key = f"{self.name}_{self.index}_particle_gun"
        return {output_key: luigi.LocalTarget(f"{self.output_file()}.csv")}


    def requires(self):
        yield FourVecGenTask(
            name = self.name
        )