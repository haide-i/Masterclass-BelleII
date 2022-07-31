
import pandas as pd
import numpy as np
import os
import b2luigi as luigi
from particle_gun import EvtGenTask

class ConcatTask(luigi.Task):
    base_path = luigi.Parameter()
    pdgs = luigi.Parameter()
    name = luigi.Parameter()

    @property
    def input_dict(self):
        if self._input_dict is None:
            self._input_dict = {k: v.path for d in self.input() if isinstance(d, dict) for k, v in d.items()}
        return self._input_dict

    @property
    def output_path(self):
        return os.path.join(self.base_path, "results")

    def run(self):
        dfs = []
        for i in range(len(self.pdgs)):
            input_key = f"{self.name}_{i}_particle_gun"
            dfs.append(pd.read_hdf(self.input_dict[input_key]))
        df = pd.concat(dfs)
        df.to_hdf(os.path.join(self.output_path, f"{self.name}.h5"))


    def requires(self):
        for i in range(len(self.pdgs)):
            yield EvtGenTask(
                base_path = self.base_path,
                pdgs = self.pdgs,
                name = self.name,
                index = i
            )