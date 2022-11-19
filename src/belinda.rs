use std::{
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};

use itertools::Itertools;
use roaring::{MultiOps, RoaringBitmap, RoaringTreemap};

use crate::{Clustering, DefaultGraph};

#[derive(Debug, Clone)]
pub enum ClusteringSource {
    Unknown,
    Cpm(f64),
    Modularity(f64),
}

impl Default for ClusteringSource {
    fn default() -> Self {
        ClusteringSource::Unknown
    }
}

pub struct EnrichedGraph {
    pub graph: DefaultGraph,
    acc_num_edges: Vec<u64>,
}

impl EnrichedGraph {
    pub fn from_graph(graph: DefaultGraph) -> EnrichedGraph {
        let mut acc = vec![0u64; graph.n() + 1];
        for i in 0..graph.n() {
            acc[i + 1] = acc[i] + graph.nodes[i].degree() as u64;
        }
        EnrichedGraph {
            graph,
            acc_num_edges: acc,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RichCluster {
    nodes: RoaringBitmap,
    n: u64,
    m: u64,
    c: u64,
    mcd: u64,
    vol: u64,
}

impl RichCluster {
    pub fn load_from_slice(g: &DefaultGraph, nodes: &[u32]) -> RichCluster {
        let n = nodes.len() as u64;
        let nodeset: RoaringBitmap = nodes.iter().cloned().map(|it| it as u32).collect();
        let mut m = 0;
        let mut c = 0;
        let mut vol = 0;
        let mut mcd = 0;
        for u in &nodeset {
            let adj = &g.nodes[u as usize].edges;
            vol += adj.len() as u64;
            if mcd == 0 && adj.len() > 0 {
                mcd = adj.len() as u64;
            } else {
                mcd = mcd.min(adj.len() as u64);
            }
            for &v in adj {
                if nodeset.contains(v as u32) {
                    m += 1;
                } else {
                    c += 1;
                }
            }
        }
        m /= 2;
        RichCluster {
            nodes: nodeset,
            n,
            m,
            c,
            mcd,
            vol,
        }
    }
}

pub struct RichClustering<const O: bool> {
    pub graph: Rc<EnrichedGraph>,
    pub clusters: BTreeMap<u64, RichCluster>,
    pub source: ClusteringSource,
}

pub struct ClusteringHandle<const O: bool> {
    pub graph: Rc<EnrichedGraph>,
    pub clustering: Rc<RichClustering<O>>,
    pub cluster_ids: BTreeSet<u64>,
}

impl<const O: bool> RichClustering<O> {
    pub fn universe_handle(clus: Rc<RichClustering<O>>) -> ClusteringHandle<O> {
        ClusteringHandle {
            graph: clus.graph.clone(),
            clustering: clus.clone(),
            cluster_ids: clus.clusters.keys().cloned().collect(),
        }
    }

    pub fn pack_from_clustering(graph: Rc<EnrichedGraph>, clus: Clustering) -> RichClustering<O> {
        let mut clusters = BTreeMap::new();
        for (i, c) in clus.clusters.into_iter() {
            let buf = &c
                .core_nodes
                .iter()
                .cloned()
                .map(|it| it as u32)
                .collect_vec();
            let c = RichCluster::load_from_slice(&graph.graph, buf);
            clusters.insert(i as u64, c);
        }
        RichClustering {
            graph: graph,
            clusters,
            source: ClusteringSource::Unknown,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GraphStats {
    covered_nodes: u32,
    covered_edges: u64,
}

impl ClusteringHandle<true> {
    pub fn stats(&self) -> GraphStats {
        let scoped_clusters = self
            .clustering
            .clusters
            .iter()
            .filter(|(k, v)| self.cluster_ids.contains(k))
            .map(|(k, v)| v)
            .collect_vec();
        let covered_nodes = scoped_clusters
            .iter()
            .map(|c| &c.nodes)
            .cloned()
            .collect_vec();
        let covered_nodes = covered_nodes.union().len() as u32;
        let mut covered_edges = RoaringTreemap::new();
        let graph = &self.graph.graph;
        let acc = &self.graph.acc_num_edges;
        for c in scoped_clusters.iter().cloned() {
            for u in c.nodes.iter() {
                let edges = &graph.nodes[u as usize].edges;
                let shift = acc[u as usize];
                for (offset, &v) in edges.into_iter().enumerate() {
                    if c.nodes.contains(v as u32) {
                        covered_edges.insert(shift + offset as u64);
                    }
                }
            }
        }
        let covered_edges = covered_edges.len() as u64;
        GraphStats {
            covered_nodes,
            covered_edges,
        }
    }
}
