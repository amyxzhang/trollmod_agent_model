from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np
import math
import random

from mesa.space import NetworkGrid

def compute_avg_delta(model):
    df = model.datacollector.get_model_vars_dataframe()
    return df['Average Delta Misinfo'].mean()
    


def compute_misinfo_seen(model):
    misinfo_seen_avg = []
    for agent in model.schedule.agents:
        misinfo_seen_avg.append(len(agent.misinfo_seen))
    return np.average(misinfo_seen_avg)

def compute_misinfo_blocked(model):
    misinfo_blocked_avg = []
    for agent in model.schedule.agents:
        misinfo_blocked_avg.append(len(agent.misinfo_blocked))
    return np.average(misinfo_blocked_avg)


class MisinfoLabelingNetwork(Model):

    def __init__(self, num_agents=30, percent_misinformers=.10, percent_mods=.20, mod_work=10):

        self.num_agents = num_agents
        self.num_nodes = num_agents
        
        self.num_misinformers = int(math.floor(float(num_agents) * percent_misinformers))
        self.num_mods = int(math.floor(float(num_agents) * percent_mods))
        self.num_regular = self.num_agents - (self.num_misinformers + self.num_mods)
        self.mod_work = mod_work
        
#         self.G = nx.barabasi_albert_graph(self.num_nodes, int(round(float(self.num_nodes)*0.90)), seed=11)
        self.G = nx.barabasi_albert_graph(self.num_nodes, int(round(float(self.num_nodes)*0.10)), seed=11)
#         self.G = nx.powerlaw_cluster_graph(self.num_nodes, int(round(float(self.num_nodes)*0.10)), p=0.9, seed=11)
        self.grid = NetworkGrid(self.G)
        self.schedule = RandomActivation(self)
        self.running = True
        
        self.datacollector = DataCollector(
            model_reporters={"Avg Misinfo Seen": compute_misinfo_seen,
                             "Avg Misinfo Blocked": compute_misinfo_blocked},
            agent_reporters={"Misinfo Seen": lambda _: len(_.misinfo_seen),
                             "Misinfo Blocked": lambda _: len(_.misinfo_blocked)}
        )

        list_of_random_nodes = self.random.sample(self.G.nodes(), self.num_agents)

        # Create agents
        for i in range(self.num_misinformers):
            a = Misinformer(i, self)
            self.schedule.add(a)
            # Add the agent to a random node
            self.grid.place_agent(a, list_of_random_nodes[i])
        
        for i in range(self.num_misinformers, self.num_misinformers + self.num_mods):
            a = ModUser(i, self)
            self.schedule.add(a)
            # Add the agent to a random node
            self.grid.place_agent(a, list_of_random_nodes[i])
        
        for i in range(self.num_misinformers + self.num_mods, self.num_agents):
            a = RegularUser(i, self)
            self.schedule.add(a)
            # Add the agent to a random node
            self.grid.place_agent(a, list_of_random_nodes[i])

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        
        self.content_ids = []
        for _ in range(10):
            self.content_ids.append(random.randint(1,10000))
        
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

    def run_model(self, n):
        for i in range(n):
            self.step()



class RegularUser(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
        self.misinfo_received = []
        self.misinfo_blocked = []
        self.misinfo_seen = []
        
        self.is_mod = False
        self.is_misinformer = False
        
        
    def find_if_labeled(self, item, neighbors):
        if self.is_mod:
            for item2 in self.misinfo_labeled:
                if item == item2:
                    return True
                        
        for neighbor in neighbors:
            if neighbor.is_mod:
                for item2 in neighbor.misinfo_labeled:
                    if item == item2:
                        return True
        return False

    def step(self):
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        neighbors = self.model.grid.get_cell_list_contents(neighbors_nodes)
    
        for item in self.misinfo_received:
            if self.find_if_labeled(item, neighbors):
                self.misinfo_blocked.append(item)
            else:
                self.misinfo_seen.append(item)
        
        self.misinfo_received = []
        

class Misinformer(RegularUser):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.is_misinformer = True

    def post_misinfo(self):
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        neighbors = self.model.grid.get_cell_list_contents(neighbors_nodes)
        
        content_id = random.choice(self.model.content_ids)
        for neighbor in neighbors:
            if content_id not in neighbor.misinfo_received:
                neighbor.misinfo_received.append(content_id)

    def step(self):
        self.post_misinfo()
        super().step()


class ModUser(RegularUser):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
        self.is_mod = True
        self.misinfo_labeled = []

    def label_misinfo(self):
        count = self.model.mod_work
        random.shuffle(self.misinfo_received)
        for item in self.misinfo_received:
            if count == 0:
                break
            
            self.misinfo_labeled.append(item)
            count -= 1
        

    def step(self):
        self.label_misinfo()
        super().step()



