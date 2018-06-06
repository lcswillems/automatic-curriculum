import networkx as nx

graphs = {}

G = nx.Graph()
G.add_nodes_from([
    "SC-D1LnInBn-v0",
    "SC-D1LaInBn-v0",
    "SC-D1LnInBa-v0",
    "SC-D1LaInBa-v0",
    "SC-D1LaIaBn-v0",
    "SC-D2LnInBn-v0",
    "SC-D4LnInBn-v0",
    "SC-D4LuIuBu-v0"
])
graphs["SC-Edgeless"] = G

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

G = nx.Graph()
G.add_edges_from([
    ("BabyAI-Unlock-v0", "BabyAI-UnlockPickup-v0"),
    ("BabyAI-UnlockPickup-v0", "BabyAI-BlockedUnlockPickup-v0")
])
graphs["BabyAI-BlockedUnlockPickup"] = G

G = nx.Graph()
G.add_edges_from([
    ("BabyAI-KeyCorridorS3R1-v0", "BabyAI-KeyCorridorS3R2-v0"),
    ("BabyAI-KeyCorridorS3R2-v0", "BabyAI-KeyCorridorS3R3-v0"),
    ("BabyAI-KeyCorridorS3R3-v0", "BabyAI-KeyCorridorS4R3-v0"),
    ("BabyAI-KeyCorridorS4R3-v0", "BabyAI-KeyCorridorS5R3-v0"),
    ("BabyAI-KeyCorridorS5R3-v0", "BabyAI-KeyCorridorS6R3-v0")
])
graphs["BabyAI-KeyCorridor"] = G

G = nx.Graph()
G.add_edges_from([
    ("BabyAI-FindObjS5-v0", "BabyAI-FindObjS6-v0"),
    ("BabyAI-FindObjS6-v0", "BabyAI-FindObjS7-v0")
])
graphs["BabyAI-FindObj"] = G

G = nx.Graph()
G.add_edges_from([
    ("BabyAI-FourObjsS5-v0", "BabyAI-FourObjsS6-v0"),
    ("BabyAI-FourObjsS6-v0", "BabyAI-FourObjsS7-v0")
])
graphs["BabyAI-FourObjs"] = G