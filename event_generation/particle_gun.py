import basf2
import b2luigi as luigi
from b2luigi.basf2_helper import Basf2PathTask
from ROOT import Belle2

import os
import numpy as np
import pandas as pd
from pandas import IndexSlice as idx

from fourvector_generation import FourVecGenTask

try:
    from simulation import add_simulation
    from reconstruction import add_reconstruction
    from mdst import add_mdst_output
except ImportError:
    # Output expected ImportErrors.
    print("Import Error")

class getECLInfo(basf2.Module):
    
    def __init__(self, output, pdg, theta, phi, p, px, py, pz, pt):
        super().__init__()
        self.output = output
        self.obj_eclgeometrypar = Belle2.ECL.ECLGeometryPar.Instance()
        self.barrel = np.arange(1153, 7777, dtype=int)
        self.col_names = [f'{i}' for i in self.barrel]
        print(pdg, theta, phi, p, px, py, pz, pt)
        self.pdg = pdg
        self.theta = theta
        self.phi = phi
        self.p = p
        self.px = px
        self.py = py
        self.pz = pz
        self.pt = pt

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
        print('event', ignore_event, 'barrel', energyinbarrel, 'correct pdg', correctpdg)

        self.index += 1

    def terminate(self):
        
        if len(self.tot_data>0):
            self.tot_data = self.tot_data[self.tot_data['pdg']!=0]
            self.tot_data.to_hdf(self.output+'h5', complevel=9, key='data')

class EvtGenTask(Basf2PathTask):
    base_path = luigi.Parameter()
    pdgs = luigi.Parameter()
    name = luigi.Parameter()
    index = luigi.Parameter()
    batch_system = "local"

    @property
    def input_dict(self):
        if self._input_dict is None:
            self._input_dict = {k: v.path for d in self.input() if isinstance(d, dict) for k, v in d.items()}
        return self._input_dict

    def create_path(self):
        main = basf2.create_path()
        particle = self.read_input().loc[:,idx[self.index,:]]

        particlegun = basf2.register_module('ParticleGun')
        particlegun.param('pdgCodes', [particle['pdg']])

        particlegun.param('momentumGeneration', 'fixed')
        particlegun.param('momentumParams', [particle['p']])

        particlegun.param('thetaGeneration', 'fixed')
        particlegun.param('thetaParams', [particle['theta']])

        particlegun.param('phiGeneration', 'fixed')
        particlegun.param('phiParams', [particle['phi']])

        main.add_module(particlegun)

        main.add_module('EventInfoSetter',
                        evtNumList=100)

        main.add_module('Gearbox')
        main.add_module('Geometry', useDB=True)

        add_simulation(path=main)

        add_reconstruction(path=main)

        add_mdst_output(path=main,
                        filename=self.output_file()+'root',
                        additionalBranches=['ECLCalDigits'])

        print(particle['theta'],
            particle['phi'],
            particle['p'],
            particle['px'],
            particle['py'],
            particle['pz'],
            particle['pt'])

        ecl_output = getECLInfo(self.output_file(),
                                particle['pdg'],
                                particle['theta'],
                                particle['phi'],
                                particle['p'],
                                particle['px'],
                                particle['py'],
                                particle['pz'],
                                particle['pt']
                                )

        main.add_module(ecl_output)

        return main

    def process(self):
        try:
            import basf2
            import ROOT
        except ImportError:
            raise ImportError("Cannot find basf2 or ROOT")

        particle_dict = self.read_fourvector()

        path = self.create_path(particle_dict)

        path.add_module('Progress')
        basf2.print_path(path)
        basf2.process(path)

        print(basf2.statistics)
    
    @property
    def output_path(self):
        return os.path.join(self.base_path, "particle_gun")
    
    def output_file(self):
        return os.path.join(self.output_path, f"{self.name}_{self.index}")

    
    def output(self):
        output_key = f"{self.name}_{self.index}_particle_gun"
        return {output_key: luigi.LocalTarget(f"{self.output_file}.h5")}


    def requires(self):
        yield FourVecGenTask(
            base_path = self.base_path,
            pdgs = self.pdgs, 
            name = self.name
        )