  
from mesa.batchrunner import BatchRunner
from model1.model import *
from mesa.datacollection import DataCollector
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


fixed_params = {'num_agents': 50}

variable_params = {'percent_trolls': np.arange(0.0, 0.5, 0.02),
                   'percent_mods': np.arange(0.0, 0.5, 0.02),
                   }

batch_run = BatchRunner(TrollModNetwork,
                        variable_params,
                        fixed_params,
                        iterations=3,
                        max_steps=20,
                        model_reporters={"average_trolling": compute_avg_delta})


if __name__ == '__main__':
    
    batch_run.run_all()
    run_data = batch_run.get_model_vars_dataframe()
    plt.scatter(run_data.percent_trolls, run_data.percent_mods, c=run_data.average_trolling, cmap='nipy_spectral')
    plt.clim(0,3.5)
    cbar = plt.colorbar()
    cbar.set_label("Average delta trolled")
    plt.xlabel('Percent trolls')
    plt.ylabel('Percent mods')
    plt.show()
    