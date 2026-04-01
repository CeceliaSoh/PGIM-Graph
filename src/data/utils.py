import torch

def build_row_normalized_adj(graph_data, add_self_loop=True):
    edge_index = graph_data["edge_index"]   # [2, E]
    num_nodes = len(graph_data["node_names"])

    # Step 1: build adjacency
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    # make undirected
    adj[edge_index[0], edge_index[1]] = 1.0
    adj[edge_index[1], edge_index[0]] = 1.0

    # Step 2: add self-loops (recommended)
    if add_self_loop:
        adj.fill_diagonal_(1.0)

    # Step 3: compute degree
    deg = adj.sum(dim=1)  # [N]

    # Step 4: row normalization: D^{-1} A
    deg_inv = deg.clamp(min=1e-8).reciprocal()  # avoid divide by 0
    adj_norm = deg_inv.unsqueeze(1) * adj       # [N, N]

    return adj_norm