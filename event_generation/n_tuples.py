
import pandas as pd
import numpy as np
import os
import b2luigi as luigi
import copy
from particle_gun import EvtGenTask
from events import events_dict, base_path

class ConcatTask(luigi.Task):
    name = luigi.Parameter()

    batch_system = "local"


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pdgs = events_dict[self.name]

    @property
    def input_dict(self):
        self._input_dict = {}
        self._input_dict = {k: v.path for d in self.input() if isinstance(d, dict) for k, v in d.items()}
        return self._input_dict

    @property
    def output_path(self):
        return os.path.join(base_path, "results", self.name)
    
    def get_log_file_dir(self):
        path = os.path.join(base_path, "concatTask", "name")
        os.makedirs(path, exist_ok=True)
        return path

    def run(self):
        print(self.input_dict)
        dfs = []
        #load all particles
        for i in range(len(self.pdgs)):
            input_key = f"{self.name}_{i}_particle_gun"
            df = pd.read_csv(self.input_dict[input_key])   
            df = df.reset_index()
            df = df.loc[[0],:]
            dfs.append(df)
        df = pd.concat(dfs).reset_index()
        # concat particles
        old_cells = [str(i) for i in np.arange(1153, 7777, dtype=int)]
        cells = [str(i) for i in np.arange(0, 6624)]
        rename_cells_dict = {o:c for o,c in zip(old_cells, cells)}
        df = df.rename(rename_cells_dict, axis = "columns")
        # remove crystals, that are not in 9x9 around center
        center = df[cells].idxmax(axis=1).to_numpy(dtype='int')
        for i in range(len(df)):
            df_mask = (abs(int((center[i]/144.))%46 - df[cells].columns.astype(int)/144.%46) < 9) & (abs((center[i])%144 - (df[cells].columns.astype(int))%144) < 9)
            df.loc[i, cells] = df.loc[i, cells].where(df_mask, 0.0)
        #fancy code to put all energy into ecl crystalls (important for e.g. muons)
        df['energy_diff'] = df['energy'] - df[cells].sum(axis=1)
        new_df = copy.deepcopy(df)
        for k in range(len(df)):
            cell_vec = new_df[cells].iloc[k][new_df[cells].iloc[k]>0.0].to_numpy()
            norm_vec = np.exp(cell_vec)/sum(np.exp(cell_vec)) #softmax for proper scaling
            addede = norm_vec*new_df['energy_diff'].iloc[k].item()
            j=0
            for i in cells:
                if new_df.loc[k, str(i)] > 0.0:
                    value = new_df.loc[k, str(i)].item()
                    new_df.at[k, str(i)] = value + addede[j]
                    j+=1
        new_df.to_hdf(os.path.join(self.output_path, f"{self.name}.h5"), key = "event")

    def output(self):
        output_key = f"{self.name}_concated"
        return {output_key: luigi.LocalTarget(os.path.join(self.output_path, f"{self.name}.h5"))}

    def requires(self):
        for i in range(len(self.pdgs)):
            yield EvtGenTask(
                name = self.name,
                index = i
            )