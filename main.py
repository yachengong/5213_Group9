from utils import load_graph_and_bases, build_service_matrix, generate_configs, compute_rij, compute_rij_weight, build_service_matrix_weight, generate_configs_weight
from gurobi_solver import optimize_q
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd

instances = [
    ("data/50-4606-6-7-35.gpickle", "data/50-4606-6-7-35.bases", "50zones"),
    ("data/100-15960-6-7-35.gpickle", "data/100-15960-6-7-35.bases", "100zones"),
    ("data/200-210828-15-20-35.gpickle", "data/200-210828-15-20-35.bases", "200zones")
]

T_list = [1, 5, 15, 30, 60, 90, 120]
ambulance_list = [1, 2, 3, 4, 6, 7]

os.makedirs("all_outputs", exist_ok=True)

for gpickle_path, base_path, label in instances:
    G, base_nodes = load_graph_and_bases(gpickle_path, base_path)
    a_ji, nodes = build_service_matrix(G, base_nodes)

    max_ambulances = len(base_nodes)
    valid_ambulance_list = [a for a in ambulance_list if a <= max_ambulances]

    gap_matrix = np.zeros((len(valid_ambulance_list), len(T_list)))

    for i, num_ambulances in enumerate(valid_ambulance_list):
        for j, T in enumerate(T_list):
            print(f"\nRunning for T = {T}, ambulances = {num_ambulances}")
            configs = generate_configs(len(base_nodes), num_ambulances)
            r_ij = compute_rij(configs, a_ji)
            result = optimize_q(r_ij, T)
            q_opt, y_opt, g, f, gap = result
            print("Optimal q:", q_opt)
            print("y_i (avg service):", y_opt)
            print(f"g = {g:.3f}, f = {f:.3f}, g - f = {gap:.3f}")
            gap_matrix[i, j] = gap

    plt.figure(figsize=(8, 5))
    sns.heatmap(gap_matrix, annot=True, fmt=".3f", xticklabels=T_list, yticklabels=valid_ambulance_list, cmap="YlOrRd")
    plt.xlabel("Total Time T")
    plt.ylabel("Number of Ambulances")
    plt.title(f"Fairness Gap (g - f) — {label}")
    plt.tight_layout()
    plt.savefig(f"all_outputs/gap_heatmap_{label}.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    for i, a in enumerate(valid_ambulance_list):
        plt.plot(T_list, gap_matrix[i], marker='o', label=f"{a} ambulances")
    plt.xlabel("Total Time T")
    plt.ylabel("Fairness Gap (g - f)")
    plt.title(f"Gap vs T for Different Ambulance Counts — {label}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"all_outputs/lineplot_{label}.png")
    plt.show()

    T_grid, A_grid = np.meshgrid(T_list, valid_ambulance_list)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T_grid, A_grid, gap_matrix, cmap='viridis')
    ax.set_xlabel("Total Time T")
    ax.set_ylabel("Number of Ambulances")
    ax.set_zlabel("Fairness Gap (g - f)")
    ax.set_title(f"3D Surface Plot — {label}")
    plt.tight_layout()
    plt.savefig(f"all_outputs/surface_{label}.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    contour = plt.contourf(T_grid, A_grid, gap_matrix, levels=20, cmap="YlGnBu")
    plt.colorbar(contour, label="Fairness Gap (g - f)")
    plt.xlabel("Total Time T")
    plt.ylabel("Number of Ambulances")
    plt.title(f"Contour of Fairness Gap — {label}")
    plt.tight_layout()
    plt.savefig(f"all_outputs/contour_{label}.png")
    plt.show()

T_list = [1, 5, 15, 30, 60]
ambulance_list = [2, 3, 4, 6, 7]

gap_matrix = np.zeros((len(ambulance_list), len(T_list)))

G, base_nodes = load_graph_and_bases("data/50-3004-6-7-35.gpickle", "data/50-3004-6-7-35.bases")
a_ji, nodes = build_service_matrix_weight(G, base_nodes)
for i, num_ambulances in enumerate(ambulance_list):
    for j, T in enumerate(T_list):
        print(f"\nRunning for T = {T}, ambulances = {num_ambulances}")
        configs = generate_configs_weight(len(base_nodes), num_ambulances)
        r_ij = compute_rij_weight(configs, a_ji)
        result = optimize_q(r_ij, T)
        q_opt, y_opt, g, f, gap = result
        print("Optimal q:", q_opt)
        print("y_i (avg service):", y_opt)
        print(f"g = {g:.3f}, f = {f:.3f}, g - f = {gap:.3f}")
        gap_matrix[i, j] = gap

plt.figure(figsize=(8, 5))
sns.heatmap(gap_matrix, annot=True, fmt=".3f",
            xticklabels=T_list, yticklabels=ambulance_list, cmap="YlOrRd")

plt.xlabel("Total Time T")
plt.ylabel("Number of Ambulances")
plt.title("Fairness Gap (g - f)")
plt.tight_layout()
plt.show()