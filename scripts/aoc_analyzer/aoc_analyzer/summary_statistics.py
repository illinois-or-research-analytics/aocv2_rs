from __future__ import annotations
from dataclasses import dataclass
from typing import List
import networkit as nk
import typer
import os.path
import pandas as pd
import numpy as np
from structlog import get_logger

def swap_second_suffix(p : str, new_name: str) -> str:
    return ".".join(p.split(".")[:-2] + [new_name, "stats", "csv"])

def is_integer(x):
    return isinstance(x, int) or x.is_integer()

def median(l):
    # returns an integer if possible
    med = np.median(l)
    return int(med) if is_integer(med) else med

def summarize_dist(l):
    minimum = min(l)
    maximum = max(l)
    med = median(l)
    # keep only two decimal places if not integer
    med = int(med) if is_integer(med) else round(med, 2)
    minimum = int(minimum) if is_integer(minimum) else round(minimum, 2)
    maximum = int(maximum) if is_integer(maximum) else round(maximum, 2)
    return f"{minimum}-{med}-{maximum}"

@dataclass
class ClusteringStats:
    num_clusters: int
    node_coverage: float
    edge_coverage: float
    cpm: List[float]
    modularity: List[float]
    mcd: List[int]
    cluster_sizes: List[int]
    conductance: List[float]
    edge_density: List[float]
    mean_degree: List[float]

    def from_df(graph: nk.Graph, df: pd.DataFrame) -> ClusteringStats:
        total_nodes = df['n'].sum()
        total_edges = df['m'].sum()
        num_clusters = len(df)
        node_coverage = total_nodes / graph.numberOfNodes()
        edge_coverage = total_edges / graph.numberOfEdges()
        cpm = df['cpm'].tolist()
        modularity = df['modularity'].tolist()
        mcd = df['mcd'].tolist()
        cluster_sizes = df['n'].tolist()
        conductance = df['conductance'].tolist()
        edge_density = [m / n ** 2 for n, m in zip(df['n'], df['m'])]
        mean_degree = [2 * m / n for n, m in zip(df['n'], df['m'])]
        return ClusteringStats(
            num_clusters=num_clusters,
            node_coverage=node_coverage,
            edge_coverage=edge_coverage,
            cpm=cpm,
            modularity=modularity,
            mcd=mcd,
            cluster_sizes=cluster_sizes,
            conductance=conductance,
            edge_density=edge_density,
            mean_degree=mean_degree,    
        )
    
    def to_stats(self) -> pd.DataFrame:
        return pd.DataFrame({
            'num_clusters': self.num_clusters,
            'node_coverage': self.node_coverage,
            'edge_coverage': self.edge_coverage,
            'cpm': summarize_dist(self.cpm),
            'modularity': summarize_dist(self.modularity),
            'mcd': summarize_dist(self.mcd),
            'cluster_sizes': summarize_dist(self.cluster_sizes),
            'conductance': summarize_dist(self.conductance),
            'edge_density': summarize_dist(self.edge_density),
            'mean_degree': summarize_dist(self.mean_degree),
        })

def main(
    input: str = typer.Option(..., "--input", "-i"),
    graph_path: str = typer.Option(..., "--graph", "-g"),
    prefixes : List[str] = typer.Option(..., "--prefixes", "-p"),
):
    log = get_logger()
    filemaps = {}
    filemaps["base"] = input
    for p in prefixes:
        filemaps[p] = swap_second_suffix(input, p)
    for _, path in filemaps.items():
        assert os.path.exists(path), f"File {path} does not exist"
    edgelist_reader = nk.graphio.EdgeListReader("\t", 0)
    graph = edgelist_reader.read(graph_path)
    n = graph.numberOfNodes()
    m = graph.numberOfEdges()
    log.info("graph loaded", graph=graph_path, n=n, m=m)
    df_maps = {}
    for name, path in filemaps.items():
        df_maps[name] = pd.read_csv(path)
    log.info("dataframes loaded")
    stats = {}
    for name, df in df_maps.items():
        stats[name] = ClusteringStats.from_df(graph, df)
    log.info("stats computed")
    # compute one DF with a column indicating the origin
    df = pd.concat([df.to_stats().assign(type=name) for name, df in stats.items()])
    # to LaTeX
    print(df.to_latex(index=False))

def entry_point():
    typer.run(main)

if __name__ == "__main__":
    entry_point()
