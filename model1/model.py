from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np
import math

from mesa.space import NetworkGrid


def compute_avg_delta(model):
    df = model.datacollector.get_model_vars_dataframe()
    return df['Average Delta Trolling'].mean()
    


def compute_troll_delta(model):
    # Trolling delta over the last 5 steps
    trolling_received_per_agent = []
    for agent in model.schedule.agents:
        if len(agent.trolling_snapshot) > 0:
            trolling_received_per_agent.append(float(agent.trolling_snapshot[0] - agent.trolling_snapshot[-1])/float(len(agent.trolling_snapshot)))
        else:
            trolling_received_per_agent.append(0)
    
    return np.average(trolling_received_per_agent)

class TrollModNetwork(Model):
    """A model with some number of trolls, mods, and regular users."""

    def __init__(self, num_agents=50, percent_trolls=.10, percent_mods=.20):

        self.num_agents = num_agents
        self.num_nodes = num_agents
        self.num_trolls = int(math.floor(float(num_agents) * percent_trolls))
        self.num_mods = int(math.floor(float(num_agents) * percent_mods))
        self.num_regular = self.num_agents - (self.num_trolls + self.num_mods)
        
        
#         self.G = nx.barabasi_albert_graph(self.num_nodes, int(round(float(self.num_nodes)*0.90)), seed=11)
#         self.G = nx.barabasi_albert_graph(self.num_nodes, int(round(float(self.num_nodes)*0.10)), seed=11)
        self.G = nx.powerlaw_cluster_graph(self.num_nodes, int(round(float(self.num_nodes)*0.1)), p=0.9, seed=11)
        self.grid = NetworkGrid(self.G)
        self.schedule = RandomActivation(self)
        self.running = True
        
        self.datacollector = DataCollector(
            model_reporters={"Average Delta Trolling": compute_troll_delta},
            agent_reporters={"Trolling Delta": lambda _: _.trolling_received_snapshot}
        )

        list_of_random_nodes = self.random.sample(self.G.nodes(), self.num_agents)

        # Create agents
        for i in range(self.num_trolls):
            a = TrollUser(i, self)
            self.schedule.add(a)
            # Add the agent to a random node
            self.grid.place_agent(a, list_of_random_nodes[i])
        
        for i in range(self.num_trolls, self.num_trolls + self.num_mods):
            a = ModUser(i, self)
            self.schedule.add(a)
            # Add the agent to a random node
            self.grid.place_agent(a, list_of_random_nodes[i])
        
        for i in range(self.num_trolls + self.num_mods, self.num_agents):
            a = RegularUser(i, self)
            self.schedule.add(a)
            # Add the agent to a random node
            self.grid.place_agent(a, list_of_random_nodes[i])

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

    def run_model(self, n):
        for i in range(n):
            self.step()



class RegularUser(Agent):
    """ An regular user in the network."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.trolling_received = 0
        self.trolling_received_snapshot = 0
        self.is_mod = False
        self.is_troll = False
        
        self.trolling_snapshot = []

    def step(self):
        self.trolling_snapshot.insert(0,self.trolling_received)
        if len(self.trolling_snapshot) > 5:
            self.trolling_snapshot = self.trolling_snapshot[0:5]
        self.trolling_received_snapshot = self.trolling_snapshot[0] - self.trolling_snapshot[-1]


class TrollUser(RegularUser):
    """ An regular user in the network."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.is_troll = True

    def send_trolling(self):
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        neighbors = self.model.grid.get_cell_list_contents(neighbors_nodes)
        for neighbor in neighbors:
            neighbor.trolling_received += 1

    def step(self):
        self.send_trolling()
        super().step()


class ModUser(RegularUser):
    """ An moderator user in the network."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.is_mod = True

    def block_trolling(self):
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        neighbors = self.model.grid.get_cell_list_contents(neighbors_nodes)
        for neighbor in neighbors:
            if neighbor.trolling_received > 0:
                neighbor.trolling_received -= 1
        if self.trolling_received > 0:
            self.trolling_received -= 1
                            
        

    def step(self):
        self.block_trolling()
        super().step()
        
