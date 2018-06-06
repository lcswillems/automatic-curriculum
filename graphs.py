import networkx as nx

def get_edgeless_graph(nodes):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    return G

def get_sequence_graph(nodes):
    G = nx.Graph()
    G.add_edges_from([(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)])
    return G

graphs = {}

graphs["SC-Edgeless"] = get_edgeless_graph([
    "SC-D1LnInBn-v0",
    "SC-D1LaInBn-v0",
    "SC-D1LnInBa-v0",
    "SC-D1LaInBa-v0",
    "SC-D1LaIaBn-v0",
    "SC-D2LnInBn-v0",
    "SC-D4LnInBn-v0",
    "SC-D4LuIuBu-v0"
])

G = nx.Graph()
G.add_edges_from([
    ("SC-D1LaInBn-v0", "SC-D1LaIaBn-v0"),
    ("SC-D1LaIaBn-v0", "SC-D1LaInBa-v0"),
    ("SC-D1LaInBa-v0", "SC-D4LuIuBu-v0"),
    ("SC-D1LnInBn-v0", "SC-D1LnInBa-v0"),
    ("SC-D1LnInBa-v0", "SC-D4LuIuBu-v0"),
    ("SC-D1LnInBn-v0", "SC-D2LnInBn-v0"),
    ("SC-D2LnInBn-v0", "SC-D4LnInBn-v0"),
    ("SC-D4LnInBn-v0", "SC-D4LuIuBu-v0")
])
graphs["SC-Normal"] = G

graphs["BabyAI-BlockedUnlockPickup"] = get_sequence_graph([
    "BabyAI-Unlock-v0",
    "BabyAI-UnlockPickup-v0",
    "BabyAI-BlockedUnlockPickup-v0"
])

graphs["BabyAI-KeyCorridor"] = get_sequence_graph([
    "BabyAI-KeyCorridorS3R1-v0",
    "BabyAI-KeyCorridorS3R2-v0",
    "BabyAI-KeyCorridorS3R3-v0",
    "BabyAI-KeyCorridorS4R3-v0",
    "BabyAI-KeyCorridorS5R3-v0",
    "BabyAI-KeyCorridorS6R3-v0"
])

graphs["BabyAI-FindObj"] = get_sequence_graph([
    "BabyAI-FindObjS5-v0",
    "BabyAI-FindObjS6-v0",
    "BabyAI-FindObjS7-v0"
])

graphs["BabyAI-FourObjs"] = get_sequence_graph([
    "BabyAI-FourObjsS5-v0",
    "BabyAI-FourObjsS6-v0",
    "BabyAI-FourObjsS7-v0"
])

graphs["BabyAI-UnlockPickupDist"] = get_sequence_graph([
    "BabyAI-Unlock-v0",
    "BabyAI-UnlockPickup-v0",
    "BabyAI-UnlockPickupDist-v0"
])