from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


PROJECT_NODE_FILE = "project_level_node.csv"
SIZE_NODE_FILE = "size_level_node.csv"
SIZE_PROJECT_EDGE_FILE = "size_project.csv"


def import_dgl():
    try:
        import dgl
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This demo needs DGL. Install it in your environment, then rerun the script."
        ) from exc
    return dgl


def import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This demo needs PyTorch. Install it in your environment, then rerun the script."
        ) from exc
    return torch


def relation_name(edge_path: Path) -> str:
    return edge_path.stem.replace("-", "_").replace(" ", "_")


def load_unique_nodes(path: Path) -> pd.DataFrame:
    nodes = pd.read_csv(path)
    if "node_id" not in nodes.columns:
        raise ValueError(f"Expected node_id column in {path}")
    return nodes.drop_duplicates("node_id").reset_index(drop=True)


def project_edge_files(edge_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in edge_dir.glob("*.csv")
        if path.name != SIZE_PROJECT_EDGE_FILE
    )


def sample_project_nodes(project_nodes: pd.DataFrame, edge_dir: Path, sample_projects: int, seed: int) -> pd.DataFrame:
    if sample_projects >= len(project_nodes):
        return project_nodes.copy()

    anchors = project_nodes.sample(n=max(1, sample_projects // 3), random_state=seed)
    selected_ids = set(anchors["node_id"].astype(int))

    for edge_path in project_edge_files(edge_dir):
        edges = pd.read_csv(edge_path, usecols=["node_id", "neig_node_id"])
        neighbor_edges = edges[
            edges["node_id"].isin(selected_ids)
            | edges["neig_node_id"].isin(selected_ids)
        ]
        for row in neighbor_edges.sample(frac=1, random_state=seed).itertuples(index=False):
            selected_ids.add(int(row.node_id))
            selected_ids.add(int(row.neig_node_id))
            if len(selected_ids) >= sample_projects:
                break
        if len(selected_ids) >= sample_projects:
            break

    if len(selected_ids) < sample_projects:
        remaining = project_nodes[~project_nodes["node_id"].isin(selected_ids)]
        fill = remaining.sample(n=sample_projects - len(selected_ids), random_state=seed)
        selected_ids.update(fill["node_id"].astype(int))

    selected_ids = set(sorted(selected_ids)[:sample_projects])
    return project_nodes[project_nodes["node_id"].isin(selected_ids)].sort_values("node_id")


def local_id_map(node_ids) -> dict[int, int]:
    return {int(node_id): idx for idx, node_id in enumerate(sorted(node_ids))}


def edge_tensors(edges: pd.DataFrame, src_map: dict[int, int], dst_map: dict[int, int]):
    torch = import_torch()
    src = torch.tensor(edges["node_id"].map(src_map).to_list(), dtype=torch.int64)
    dst = torch.tensor(edges["neig_node_id"].map(dst_map).to_list(), dtype=torch.int64)
    return src, dst


def add_project_relations(edge_dir: Path, project_map: dict[int, int], max_edges_per_relation: int, seed: int):
    data_dict = {}
    selected_project_ids = set(project_map)
    for offset, edge_path in enumerate(project_edge_files(edge_dir)):
        edges = pd.read_csv(edge_path)
        required_cols = {"node_id", "neig_node_id"}
        if not required_cols.issubset(edges.columns):
            print(f"Skipping {edge_path.name}: missing {sorted(required_cols)}")
            continue

        edges = edges[
            edges["node_id"].isin(selected_project_ids)
            & edges["neig_node_id"].isin(selected_project_ids)
        ].drop_duplicates(["node_id", "neig_node_id"])
        if edges.empty:
            continue
        if len(edges) > max_edges_per_relation:
            edges = edges.sample(n=max_edges_per_relation, random_state=seed + offset)

        rel = relation_name(edge_path)
        data_dict[("project", rel, "project")] = edge_tensors(edges, project_map, project_map)

    return data_dict


def add_size_project_relations(edge_dir: Path, size_map: dict[int, int], project_map: dict[int, int]):
    torch = import_torch()
    edge_path = edge_dir / SIZE_PROJECT_EDGE_FILE
    if not edge_path.exists():
        return {}

    edges = pd.read_csv(edge_path)
    edges = edges[
        edges["node_id"].isin(size_map)
        & edges["neig_node_id"].isin(project_map)
    ].drop_duplicates(["node_id", "neig_node_id"])
    if edges.empty:
        return {}

    size_src = torch.tensor(edges["node_id"].map(size_map).to_list(), dtype=torch.int64)
    project_dst = torch.tensor(edges["neig_node_id"].map(project_map).to_list(), dtype=torch.int64)
    return {
        ("size", "belongs_to", "project"): (size_src, project_dst),
        ("project", "has_size", "size"): (project_dst, size_src),
    }


def build_sample_heterograph(node_dir: Path, edge_dir: Path, sample_projects: int, max_edges_per_relation: int, seed: int):
    dgl = import_dgl()
    torch = import_torch()

    project_nodes = load_unique_nodes(node_dir / PROJECT_NODE_FILE)
    size_nodes = load_unique_nodes(node_dir / SIZE_NODE_FILE)
    sampled_projects = sample_project_nodes(project_nodes, edge_dir, sample_projects, seed)

    project_ids = set(sampled_projects["node_id"].astype(int))
    size_project_edges = pd.read_csv(edge_dir / SIZE_PROJECT_EDGE_FILE)
    sampled_size_ids = set(
        size_project_edges.loc[
            size_project_edges["neig_node_id"].isin(project_ids),
            "node_id",
        ].astype(int)
    )
    sampled_sizes = size_nodes[size_nodes["node_id"].isin(sampled_size_ids)].copy()

    project_map = local_id_map(project_ids)
    size_map = local_id_map(sampled_size_ids)

    data_dict = {}
    data_dict.update(add_project_relations(edge_dir, project_map, max_edges_per_relation, seed))
    data_dict.update(add_size_project_relations(edge_dir, size_map, project_map))

    graph = dgl.heterograph(
        data_dict,
        num_nodes_dict={
            "project": len(project_map),
            "size": len(size_map),
        },
    )

    graph.nodes["project"].data["raw_node_id"] = torch.tensor(sorted(project_map), dtype=torch.int64)
    graph.nodes["size"].data["raw_node_id"] = torch.tensor(sorted(size_map), dtype=torch.int64)
    return graph, sampled_projects, sampled_sizes


def print_heterograph(graph, sampled_projects: pd.DataFrame, sampled_sizes: pd.DataFrame) -> None:
    print("Heterogeneous DGL graph")
    print(f"  graph: {graph}")
    print(f"  node types: {graph.ntypes}")
    print(f"  edge types: {graph.etypes}")
    print()

    for ntype in graph.ntypes:
        raw_ids = graph.nodes[ntype].data["raw_node_id"].tolist()
        print(f"{ntype} nodes")
        print(f"  count: {graph.num_nodes(ntype)}")
        print(f"  raw node ids: {raw_ids}")
        print()

    print("Relations")
    for canonical_etype in graph.canonical_etypes:
        src_type, rel, dst_type = canonical_etype
        src, dst = graph.edges(etype=canonical_etype)
        print(f"  ({src_type}) -[{rel}]-> ({dst_type})")
        print(f"    edges: {graph.num_edges(canonical_etype)}")
        print(f"    local src: {src.tolist()}")
        print(f"    local dst: {dst.tolist()}")

    if "project_id" in sampled_projects.columns:
        print()
        print("Sampled project table columns to inspect")
        cols = [col for col in ["node_id", "project_id", "latitude", "longitude", "rent_per_sqft"] if col in sampled_projects.columns]
        print(sampled_projects[cols].head(10).to_string(index=False))

    if "project_id" in sampled_sizes.columns:
        print()
        print("Sampled size table columns to inspect")
        cols = [col for col in ["node_id", "project_id", "rent_per_sqft", "rent_per_sqft_imp"] if col in sampled_sizes.columns]
        print(sampled_sizes[cols].head(10).to_string(index=False))


def save_heterograph_plot(graph, output: Path) -> None:
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    import networkx as nx

    relation_colors = {
        "belongs_to": "#0f766e",
        "has_size": "#14b8a6",
        "dist_250": "#2563eb",
        "same_age": "#9333ea",
        "same_mrt_250": "#dc2626",
        "same_mrt_dist_eps_5": "#ea580c",
        "same_planning_area": "#16a34a",
        "same_school_dist_eps_0p01": "#ca8a04",
    }
    fallback_colors = ["#475569", "#be123c", "#7c3aed", "#0891b2", "#4d7c0f"]

    nx_graph = nx.MultiDiGraph()
    positions = {}
    labels = {}

    project_raw_ids = graph.nodes["project"].data["raw_node_id"].tolist()
    size_raw_ids = graph.nodes["size"].data["raw_node_id"].tolist()
    project_count = max(1, len(project_raw_ids))
    size_count = max(1, len(size_raw_ids))

    for idx, raw_id in enumerate(project_raw_ids):
        angle = 2 * math.pi * idx / project_count
        node_key = ("project", idx)
        nx_graph.add_node(node_key, ntype="project")
        positions[node_key] = (math.cos(angle), math.sin(angle))
        labels[node_key] = f"p:{raw_id}"

    for idx, raw_id in enumerate(size_raw_ids):
        x = -1.3 + 2.6 * idx / max(1, size_count - 1)
        node_key = ("size", idx)
        nx_graph.add_node(node_key, ntype="size")
        positions[node_key] = (x, -1.7)
        labels[node_key] = f"s:{raw_id}"

    used_relations = []
    for canonical_etype in graph.canonical_etypes:
        src_type, rel, dst_type = canonical_etype
        src, dst = graph.edges(etype=canonical_etype)
        used_relations.append(rel)
        for src_id, dst_id in zip(src.tolist(), dst.tolist()):
            nx_graph.add_edge((src_type, src_id), (dst_type, dst_id), relation=rel)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 9))

    project_nodes = [node for node, data in nx_graph.nodes(data=True) if data["ntype"] == "project"]
    size_nodes = [node for node, data in nx_graph.nodes(data=True) if data["ntype"] == "size"]
    nx.draw_networkx_nodes(
        nx_graph,
        positions,
        nodelist=project_nodes,
        node_shape="o",
        node_color="#dbeafe",
        edgecolors="#1d4ed8",
        linewidths=1.5,
        node_size=900,
    )
    nx.draw_networkx_nodes(
        nx_graph,
        positions,
        nodelist=size_nodes,
        node_shape="s",
        node_color="#ccfbf1",
        edgecolors="#0f766e",
        linewidths=1.5,
        node_size=650,
    )
    nx.draw_networkx_labels(nx_graph, positions, labels=labels, font_size=8)

    for idx, rel in enumerate(sorted(set(used_relations))):
        color = relation_colors.get(rel, fallback_colors[idx % len(fallback_colors)])
        edges = [
            (src, dst)
            for src, dst, data in nx_graph.edges(data=True)
            if data["relation"] == rel
        ]
        nx.draw_networkx_edges(
            nx_graph,
            positions,
            edgelist=edges,
            edge_color=color,
            alpha=0.65,
            width=1.4,
            arrows=True,
            arrowsize=12,
            connectionstyle=f"arc3,rad={0.05 + 0.02 * (idx % 6)}",
        )

    legend_handles = [
        mlines.Line2D([], [], color=relation_colors.get(rel, fallback_colors[idx % len(fallback_colors)]), label=rel)
        for idx, rel in enumerate(sorted(set(used_relations)))
    ]
    plt.legend(handles=legend_handles, loc="upper left", fontsize=8, frameon=False)
    plt.title("Sampled PGIM DGL Heterograph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()
    print(f"\nSaved heterograph visualization to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample PGIM node/edge CSVs into a small DGL heterograph.")
    parser.add_argument("--node-dir", default="dataset/database_260519/nodes", help="Folder containing node CSVs.")
    parser.add_argument("--edge-dir", default="dataset/database_260519/edges", help="Folder containing edge CSVs.")
    parser.add_argument("--sample-projects", type=int, default=8, help="Number of project nodes to sample.")
    parser.add_argument("--max-edges-per-relation", type=int, default=8, help="Maximum project-project edges per relation.")
    parser.add_argument("--output", default="sample_heterograph.png", help="PNG path for the visualization.")
    parser.add_argument("--no-plot", action="store_true", help="Print the heterograph only; do not save a PNG.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph, sampled_projects, sampled_sizes = build_sample_heterograph(
        node_dir=Path(args.node_dir),
        edge_dir=Path(args.edge_dir),
        sample_projects=args.sample_projects,
        max_edges_per_relation=args.max_edges_per_relation,
        seed=args.seed,
    )
    print_heterograph(graph, sampled_projects, sampled_sizes)
    if not args.no_plot:
        save_heterograph_plot(graph, Path(args.output))


if __name__ == "__main__":
    main()
