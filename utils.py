import pickle
import networkx as nx
from itertools import product
import numpy as np
import matplotlib.pyplot as plt

# Load the graph and base nodes from files
def load_graph_and_bases(gpickle_path, bases_path):
    # Load the graph from a gpickle file
    with open(gpickle_path, "rb") as f:
        G = pickle.load(f)
    # Load the base nodes from a text file
    with open(bases_path, "r") as f:
        base_nodes = [eval(line.strip()) for line in f.readlines()] 
    return G, base_nodes

# Build the service matrix a_ji
# a_ji[j][i] = 1 if base j can serve node i
# a_ji[j][i] = 0 otherwise
def build_service_matrix(G, base_nodes, weight_threshold=700):
    nodes = list(G.nodes())
    node_index = {node: idx for idx, node in enumerate(nodes)}
    a_ji = np.zeros((len(base_nodes), len(nodes)))

    # Convert graph to adjacency matrix with weights
    adj_matrix = nx.adjacency_matrix(G, weight='weight')
    weight = adj_matrix.toarray() # get the weighted adjacency matrix

    # Iterate over each base node and check which nodes it can serve
    for j, base in enumerate(base_nodes):
        base_idx = node_index[base] # Get the index of the base node
        for i, node in enumerate(nodes):
            if weight[base_idx][i] > 0 and weight[base_idx][i] <= weight_threshold:
                i = node_index[node]
                a_ji[j][i] = 1
# Optional: allow base to cover itself (if needed)
        a_ji[j][base_idx] = 1
    
    return a_ji, nodes

# Build the service matrix a_ji
# a_ji[j][i] = 1 if base j can serve node i
# a_ji[j][i] = 0 otherwise
def build_service_matrix_weight(G, base_nodes, weight_threshold=700):
    nodes = list(G.nodes())
    node_index = {node: idx for idx, node in enumerate(nodes)}
    a_ji = np.zeros((len(base_nodes), len(nodes)))

    # Convert graph to adjacency matrix with weights
    adj_matrix = nx.adjacency_matrix(G, weight='weight')
    weight = adj_matrix.toarray() # get the weighted adjacency matrix

    # Iterate over each base node and check which nodes it can serve
    for j, base in enumerate(base_nodes):
        base_idx = node_index[base] # Get the index of the base node
        for i, node in enumerate(nodes):
            dist = weight[base_idx][i]
            if dist > 0 and dist <= weight_threshold:
                i = node_index[node]
                a_ji[j][i] = 1.0 / (weight[base_idx][i] + 1e-6)

        a_ji[j][base_idx] = 1

    return a_ji, nodes

# Generate all configurations of ambulances across bases
def generate_configs(num_bases, num_ambulances):
    configs = []
    # Generate all combinations of 0s and 1s for the number of bases
    for config in product([0, 1], repeat=num_bases):
        # Count the number of ambulances in the configuration
        if sum(config) == num_ambulances:
            configs.append(list(config))
    return configs

def generate_configs_weight(num_bases, num_ambulances):
    configs = []
    # Generate all combinations of ambulances across bases
    for config in product(range(num_ambulances + 1), repeat=num_bases):
        if sum(config) == num_ambulances: # Ensure total ambulances equals num_ambulances
            configs.append(list(config))
    return configs

# Compute the coverage matrix r_ij
# r_ij[i][j] = 1 if ambulance j can serve node i
# r_ij[i][j] = 0 otherwise
def compute_rij(configs, a_ji):
    n = len(a_ji[0]) # Number of nodes
    K = len(configs) # Number of configurations
    r_ij = np.zeros((n, K)) # Initialize the coverage matrix
    for j, config in enumerate(configs):
        coverage = np.zeros(n)
        # For each configuration, check which nodes can be served
        for b_idx, put in enumerate(config):
            if put > 0: # put > 0 means there are ambulances in this base
                coverage = np.maximum(coverage, a_ji[b_idx]) # Update coverage for this base
        r_ij[:, j] = coverage # Store the coverage for this configuration
    # print(r_ij) # the rij of the first config
    return r_ij

def compute_rij_weight(configs, a_ji):
    configs_np = np.array(configs)
    r_ij = configs_np @ a_ji
    return r_ij.T