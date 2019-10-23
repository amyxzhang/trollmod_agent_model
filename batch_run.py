from mesa.batchrunner import BatchRunner
from agentmodel.model import *
from mesa.datacollection import DataCollector
import pandas as pd
import matplotlib.pyplot as plt


fixed_params = {'num_agents': 50}
variable_params = {'percent_trolls': [.1,.2,.3],
                   'percent_mods': [.1,.2,.3],
                   }

batch_run = BatchRunner(TrollModNetwork,
                        variable_params,
                        fixed_params,
                        iterations=5,
                        max_steps=100,
                        model_reporters={"average_trolling": compute_troll_avg})


if __name__ == '__main__':
    
    batch_run.run_all()
    run_data = batch_run.get_model_vars_dataframe()
    run_data.head()
    plt.scatter(run_data.percent_trolls, run_data.average_trolling)
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
