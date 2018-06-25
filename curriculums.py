import networkx as nx

def get_edgeless_graph(nodes):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    return G

def get_sequence_graph(nodes):
    G = nx.DiGraph()
    G.add_edges_from([(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)])
    return G

curriculums = {}

G = nx.DiGraph()
G.add_edges_from([
    ("SC-D1LaInBn-v0", "SC-D1LaInBa-v0"),
    ("SC-D1LaInBa-v0", "SC-D1LuIuBu-v0"),
    ("SC-D1LnInBn-v0", "SC-D1LnInBa-v0"),
    ("SC-D1LnInBa-v0", "SC-D1LuIuBu-v0")
])
curriculums["SC-Soft"] = G

G = nx.DiGraph()
G.add_edges_from([
    # Depth 1
    ("SC-Soft-Ld-v0", "SC-Soft-LdH-v0"),
    ("SC-Soft-Ld-v0", "SC-Ld-v0"),
    # Depth 2
    ("SC-Soft-LdH-v0", "SC-Soft-LdHB-v0"),
    ("SC-Soft-LdH-v0", "SC-LdH-v0"),
    ("SC-Ld-v0", "SC-LdH-v0"),
    # Depth 3
    ("SC-Soft-LdHB-v0", "SC-LdHB-v0"),
    ("SC-LdH-v0", "SC-LdHB-v0"),
    # Depth 4
    ("SC-LdHB-v0", "SC-1Q-v0"),
    # Depth 5
    ("SC-1Q-v0", "SC-2Q-v0"),
    # Depth 6
    ("SC-2Q-v0", "SC-Full-v0"),
])
curriculums["SC-Soft"] = G

curriculums["SC-Hard"] = get_sequence_graph([
    "SC-Ld-v0",
    "SC-LdH-v0",
    "SC-LdHB-v0",
    "SC-1Q-v0",
    "SC-Full-v0"
])

curriculums["BabyAI-BlockedUnlockPickup"] = get_sequence_graph([
    "BabyAI-Unlock-v0",
    "BabyAI-UnlockPickup-v0",
    "BabyAI-BlockedUnlockPickup-v0"
])

curriculums["BabyAI-KeyCorridor"] = get_sequence_graph([
    "BabyAI-KeyCorridorS3R1-v0",
    "BabyAI-KeyCorridorS3R2-v0",
    "BabyAI-KeyCorridorS3R3-v0",
    "BabyAI-KeyCorridorS4R3-v0",
    "BabyAI-KeyCorridorS5R3-v0",
    "BabyAI-KeyCorridorS6R3-v0"
])

curriculums["BabyAI-FindObj"] = get_sequence_graph([
    "BabyAI-FindObjS5-v0",
    "BabyAI-FindObjS6-v0",
    "BabyAI-FindObjS7-v0"
])

curriculums["BabyAI-1Room"] = get_sequence_graph([
    "BabyAI-1RoomS8-v0",
    "BabyAI-1RoomS12-v0",
    "BabyAI-1RoomS16-v0",
    "BabyAI-1RoomS20-v0"
])

curriculums["BabyAI-FourObjs"] = get_sequence_graph([
    "BabyAI-FourObjsS5-v0",
    "BabyAI-FourObjsS6-v0",
    "BabyAI-FourObjsS7-v0"
])

curriculums["BabyAI-UnlockPickupDist"] = get_sequence_graph([
    "BabyAI-UnlockPickup-v0",
    "BabyAI-UnlockPickupDist-v0"
])