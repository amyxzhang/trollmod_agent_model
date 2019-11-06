from mesa.batchrunner import BatchRunner
from model2.model import *
from mesa.datacollection import DataCollector
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


fixed_params = {'num_agents': 50,
                'percent_trolls': 0.1}

variable_params = {'percent_mods': np.arange(0.0, 0.8, 0.02),
                   'mod_power': range(0,30,1),
                   }

batch_run = BatchRunner(TrollModNetwork,
                        variable_params,
                        fixed_params,
                        iterations=3,
                        max_steps=20,
                        model_reporters={"average_trolling_delta": compute_avg_delta})


if __name__ == '__main__':
    
    batch_run.run_all()
    run_data = batch_run.get_model_vars_dataframe()
    plt.scatter(run_data.mod_power, run_data.percent_mods, c=run_data.average_trolling_delta, cmap='nipy_spectral')
    plt.clim(0,1.75)
    cbar = plt.colorbar()
    cbar.set_label("Average trolling delta")
    plt.xlabel('Mod Power')
    plt.ylabel('Percent mods')
    plt.show()
    
    
#     model = TrollModNetwork(50,.1,.3)
#     for i in range(100):
#         model.step()
#     
#     trolling = [a.trolling_received for a in model.schedule.agents]
#     plt.hist(trolling)
#     plt.show()
#     
#     
#     trolling = model.datacollector.get_model_vars_dataframe()
#     trolling.plot()
#     plt.show()
