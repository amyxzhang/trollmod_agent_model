from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule
from mesa.visualization.modules import NetworkModule
from .model import TrollModNetwork


def compute_color(agent):
    if agent.is_mod:
        return "#0000FF" # blue
    elif agent.is_troll:
        return "#CC0000" # red
    elif agent.trolling_received_snapshot <= 0:
        return "#037f51" # green
    elif agent.trolling_received_snapshot < 2 :
        return "#FFFF00" # yellow
    else:
        return "#FFA500" # orange


def network_portrayal(G):
    # The model ensures there is 0 or 1 agent per node

    portrayal = dict()
    portrayal['nodes'] = [{'id': node_id,
                           'size': 2,
                           'color': compute_color(agents[0]),
                           'label': None if not agents else 'Agent:{} Trolling Delta:{} Total: {}'.format(agents[0].unique_id,
                                                                                        agents[0].trolling_received_snapshot, agents[0].trolling_received),
                           }
                          for (node_id, agents) in G.nodes.data('agent')]

    portrayal['edges'] = [{'id': edge_id,
                           'source': source,
                           'target': target,
                           'color': '#000000',
                           }
                          for edge_id, (source, target) in enumerate(G.edges)]

    return portrayal


grid = NetworkModule(network_portrayal, 500, 500, library='sigma')
chart = ChartModule([
    {"Label": "Average Delta Trolling", "Color": "Black"}],
    data_collector_name='datacollector'
)


model_params = {
    "num_agents": UserSettableParameter('slider', "Number of agents", 30, 0, 100, 10,
                                        description="Choose how many total agents to include in the model"),
    "percent_trolls": UserSettableParameter('slider', "Percent trolls", .1, 0, 1.0, 0.05,
                                       description="Choose what percent trolls to include in the model"),
    "percent_mods": UserSettableParameter('slider', "Percent mods", .2, 0, 1.0, 0.05,
                                       description="Choose what percent mods to include in the model"),
    "mod_power": UserSettableParameter('slider', "Mod power", 10, 0, 20, 2,
                                       description="How many items mods can remove per step")
}

server = ModularServer(TrollModNetwork, [grid, chart], "Troll Mod Model", model_params)
server.port = 8521