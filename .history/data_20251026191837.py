import numpy as np
import torch
from utils import black_scholes_solution

def generate_terminal_data(config):
    S = np.random.uniform(config["min_S"], config["max_S"], (config["N_data"], 1))
    t = np.full_like(S, config["T"])  
    C = np.maximum(S - config["K"], 0.0)
    C += np.random.normal(config["bias"], config["noise_variance"], size=C.shape)

    return (
        torch.tensor(S, dtype=torch.float32, requires_grad=True),
        torch.tensor(t, dtype=torch.float32, requires_grad=True),
        torch.tensor(C, dtype=torch.float32),
    )

def generate_boundary_data(config):
    S_min = np.full((config["N_data"], 1), config["min_S"], dtype=np.float64)
    S_max = np.full((config["N_data"], 1), config["max_S"]*2, dtype=np.float64)
    t_vals = np.random.uniform(0, config["T"], (config["N_data"], 1))
    # Boundary values
    C_min = np.zeros_like(S_min)  # C(min_S, t) = 0
    C_min += np.random.normal(config["bias"], config["noise_variance"], size=C_min.shape)
    # C_max = black_scholes_solution(S_max, config["K"], config["T"] - t_vals, config["r"], config["sigma"])
    # C_max += np.random.normal(config["bias"], config["noise_variance"], size=C_max.shape)
    
    # Concatenate both boundaries
    S_all = np.vstack([S_min, S_max])
    t_all = np.vstack([t_vals, t_vals])
    C_all = np.vstack([C_min, np.zeros_like(S_max)])

    return (
        torch.tensor(S_all, dtype=torch.float32, requires_grad=True),
        torch.tensor(t_all, dtype=torch.float32, requires_grad=True),
        torch.tensor(C_all, dtype=torch.float32),
    )


def generate_collocation_points(config):
    S = torch.tensor(np.random.uniform(config["min_S"], config["max_S"], (config["N_data"], 1)), dtype=torch.float32, requires_grad=True)
    t = torch.tensor(np.random.uniform(0, config["T"], (config["N_data"], 1)), dtype=torch.float32, requires_grad=True)
    return S, t
