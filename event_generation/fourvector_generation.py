
from scipy.optimize import *
from scipy.optimize.nonlin import *
from pandas import IndexSlice as idx 
import b2luigi as luigi
import numpy as np
import pandas as pd
import os

masses_dict = {"11": 0.51998e-3, 
               "-11": 0.51998e-3, 
               "111": 134.9768e-3, 
               "211": 139.57039e-3, 
               "-211": 139.57039e-3,
              "22": 0,
              "13": 206.7682830e-3,
              "-13": 206.7682830e-3,
              "321": 493.677e-3,
              "-321": 493.677e-3,
              "310": 497.611e-3}


@pd.api.extensions.register_dataframe_accessor("v4")
class FourVecAccessor(object):
    def __init__(self, pandas_obj):
        # to distinguish between multiple particles or single particle
        # we only need to save the column information,
        self._obj_columns = pandas_obj.columns
        # to keep data consistent when appending columns unsing this accessor save indices to use them in returns
        self._obj_indices = pandas_obj.index
        # get the correct index level, 0 for single particle, 1 for multiple
        _vars = self._obj_columns.get_level_values(self._obj_columns.nlevels - 1)

        if 'M' in _vars and "px" in _vars:
            kin_vars = ["M", "px", "py", "pz"]
        elif 'mass' in _vars and 'px' in _vars:
            kin_vars = ["mass", "px", "py", "pz"]
        elif 'mass' in _vars and 'pt' in _vars:
            kin_vars = ["mass", "pt", "phi", "eta"]
        elif 'E' in _vars and "px" in _vars:
            kin_vars = ["E", "px", "py", "pz"]
        elif 'E' in _vars and 'pt' in _vars:
            kin_vars = ["E", "pt", "phi", "eta"]
        else:
            raise KeyError("No matching structure implemented for interpreting the data as a four "
                           "momentum!")

        # the following lines are where the magic happens

        # no multi-index, just on particle
        if self._obj_columns.nlevels == 1:
            # get the dtypes for the kinetic variables
            dtypes = pandas_obj.dtypes
            kin_view = list(map(lambda x: (x, dtypes[x]), kin_vars))

            # get the kinetic variables from the dataframe and convert it to a numpy array.
            # require it to be C_CONTIGUOUS, vector uses C-Style
            # This array can then be viewed as a vector object.
            # Every property not given is calculated on the fly by the vector object.
            # E.g. the mass is not stored but calculated when the energy is given and vice versa.
            self._v4 = np.require(pandas_obj[kin_vars].to_numpy(), requirements='C').view(
                kin_view).view(vec.MomentumNumpy4D)

        # multi-index, e.g. getting the four momentum for multiple particles
        elif self._obj_columns.nlevels == 2:
            # get the dtypes for the kinetic variables
            # assume the same dtypes for the other particles
            dtypes = pandas_obj[self._obj_columns.get_level_values(0).unique()[0]].dtypes
            kin_view = list(map(lambda x: (x, dtypes[x]), kin_vars))
            self._v4 = np.require(pandas_obj.loc[:, (self._obj_columns.get_level_values(0).unique(),
                                                     kin_vars)].to_numpy(),
                                  requirements='C').view(kin_view).view(vec.MomentumNumpy4D)

        else:
            raise IndexError("Expected a dataframe with a maximum of two multi-index levels.")

    def __getattribute__(self, item):
        """
        Attributes of this accessor are forwarded to the four vector.

        Returns either a pandas dataframe, if we have multiple particles
        or a pandas Series for a single particle.
        """
        try:
            return object.__getattribute__(self, item)
        except AttributeError:
            try:
                return pd.DataFrame(self._v4.__getattribute__(item),
                                    columns=pd.MultiIndex.from_product(
                                        [self._obj_columns.unique(0), [item]]),
                                    index=self._obj_indices)
            except ValueError:
                try:
                    return pd.Series(self._v4.__getattribute__(item).flatten(), name=item, index=self._obj_indices)
                except AttributeError as e:
                    if "'function' object has no attribute 'flatten'" in str(e):
                        raise AttributeError(
                            "Functions of the four vectors can NOT be called directly via the "
                            "accessor. Use the vector property instead! "
                            "Usage: 'df['particle'].v4.vector.desired_function()'")
                    raise e

    @property
    def vector(self):
        """The four vector object itself. It's required when using methods like boosting."""
        if self._obj_columns.nlevels == 1:
            return self._v4[:, 0]
        else:
            return self._v4


class FourVecGenTask(luigi.Task):
    base_path = luigi.Parameter()
    pdgs = luigi.Parameter()
    name = luigi.Parameter()
    batch_system = "local"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def generate_event(self):
        n = len(self.pdgs)
        masses = np.array([masses_dict[pdg] for pdg in self.pdgs])
        theta_in_barrel = False
        correct_masses = False
        while not (theta_in_barrel and correct_masses):
            #generate n-2 momenta
            part = (np.random.rand(n-2,3, )-0.5)*10
            #calculate n-2 Energies from masses
            part_E = np.sqrt(masses[:-2]**2 + (part**2).sum(1))
            event = pd.DataFrame(part, columns = ["px", "py", "pz"])
            event.loc[:,"E"] = part_E
            #generate px0 and py0
            px0 = (np.random.rand(1)-0.5)*10
            py0 = (np.random.rand(1)-0.5)*10
            # calculate px1 and py1 via momentum conservation
            py1 = - event.loc[:,"py"].sum() - py0
            px1 = - event.loc[:,"px"].sum() - px0
            # build nonlinear equation system for remaining values
            def non_lin_eq(x):
                E0, E1, pz0, pz1 = x
                M0 = masses[-2]
                M1 = masses[-1]
                Esum = event.loc[:,"E"].sum()
                pzsum = event.loc[:,"pz"].sum()
                eq1 = E0**2 - px0**2 - py0**2 - pz0**2 - M0**2,
                eq2 = E1**2 - px1**2 - py1**2 - pz1**2 - M1**2,
                eq3 = Esum + E0 + E1 - 10.580,
                eq4 = pzsum + pz0 + pz1
                eq1, eq2, eq3 = eq1[0][0], eq2[0][0], eq3[0]
                return [eq1, eq2, eq3, eq4]
            #solve it
            E0,E1,pz0,pz1 = self.solve_non_lin_eq(non_lin_eq=non_lin_eq)
            #complete event DataFrame
            event.loc[n-1, ["E","px","py","pz"]] = [E0,px0,py0,pz0]
            event.loc[n, ["E","px","py","pz"]] = [E1,px1,py1,pz1]
            event = event.astype("float")
            event.loc[:,"theta"] = event.v4.theta
            event.loc[:,"pt"] = event.v4.pt
            event.loc[:,"p"] = event.v4.p
            event.loc[:,"phi"] = event.v4.phi
            event.loc[:,"InvM"] = event.v4.M
            event.loc[:,"pdg"] = self.pdgs
            # calculate checks if event generated correctly
            theta_in_barrel = event.astype("float").v4.theta/(2*np.pi)*360
            correct_masses = (abs(event.loc[:,"InvM"] - masses) < 1e-3*masses).sum() == n
        
        return event
                    
    def solve_non_lin_eq(self, non_lin_eq):
        start_values = (0,0,0,0)
        E0,E1,pz0,pz1 = 0,0,0,0
        solver_trys = 0
        solver_index = 0
        solver_list = ["hybr", "lm", "broyden1", "broyden2", "anderson", "linearmixing", "diagbroyden", "excitingmixing", "krylov", "df-sane"]
        while (abs(np.array(func((E0,E1,pz0, pz1)))) < 1e-6).sum() != 4:  
            # solver might not converge
            solver_trys += 1
            try:
                E0, E1, pz0, pz1 = root(non_lin_eq, start_values, method = solver_list[solver_index], jac=False).x
            except NoConvergence as e:
                pass #retrying with different values when non convergent
            # if not coverging for 10 trys, try different solver
            if (solver_trys>10):
                solver_trys = 0
                solver_index+=1
                # if all solvers are tried, return latest results
                if solver_index>len(solver_list):
                    break
            start_values = ((np.random.rand(4)-0.5)*10).tolist()
    
    @property
    def output_path(self):
        return os.path.join(self.base_path, "fourvec_generation")
    
    def output(self):
        output_key = f"{self.name}_fourvec_generation"
        return {output_key: luigi.LocalTarget(os.path.join(self.output_path, f"{self.name}.csv"))}

    def run(self):
        event = self.generate_event().stack()
        event.to_csv(os.path.join(self.output_path, f"{self.name}.csv"))