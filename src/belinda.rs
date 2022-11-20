use std::{
    collections::{BTreeMap, BTreeSet},
    rc::Rc, sync::Arc,
};

use ahash::AHashMap;
use indicatif::ParallelProgressIterator;
use itertools::Itertools;
use rayon::prelude::{
    FromParallelIterator, IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    ParallelIterator,
};
use roaring::{MultiOps, RoaringBitmap, RoaringTreemap};
use tracing::debug;

use crate::{
    quality::DistributionSummary,
    utils::{calc_cpm_resolution, calc_modularity_resolution, choose2},
    Clustering, DefaultGraph,
};

#[derive(Debug, Clone)]
pub enum ClusteringSource {
    Unknown,
    Cpm(f64),
    Modularity(f64),
}

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
#[derive(strum_macros::Display)]
pub enum StatisticsType {
    Mcd,
    Cpm,
    Size,
    Modularity,
    Conductance,
    DeviationTreeness,
    Density,
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
    pub nodes: RoaringBitmap,
    pub n: u64,
    pub m: u64,
    pub c: u64,
    pub mcd: u64,
    pub vol: u64,
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
    pub graph: Arc<EnrichedGraph>,
    pub clusters: BTreeMap<u64, RichCluster>,
    pub source: ClusteringSource,
}

pub struct ClusteringHandle<const O: bool> {
    pub graph: Arc<EnrichedGraph>,
    pub clustering: Arc<RichClustering<O>>,
    pub cluster_ids: BTreeSet<u64>,
}

impl<const O: bool> RichClustering<O> {
    pub fn universe_handle(clus: Arc<RichClustering<O>>) -> ClusteringHandle<O> {
        ClusteringHandle {
            graph: clus.graph.clone(),
            clustering: clus.clone(),
            cluster_ids: clus.clusters.keys().cloned().collect(),
        }
    }

    pub fn pack_from_clustering(graph: Arc<EnrichedGraph>, clus: Clustering) -> RichClustering<O> {
        let k = clus.clusters.len();
        let raw_graph = &graph.graph;
        let mut clusters =
            BTreeMap::from_par_iter(clus.clusters.into_par_iter().progress_count(k as u64).map(
                |(k, c)| {
                    (
                        k as u64,
                        RichCluster::load_from_slice(
                            raw_graph,
                            &&c.core_nodes
                                .iter()
                                .cloned()
                                .map(|it| it as u32)
                                .collect_vec(),
                        ),
                    )
                },
            ));
        RichClustering {
            graph,
            clusters,
            source: ClusteringSource::Unknown,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SummarizedDistribution {
    pub percentiles: Box<[f64; 21]>,
}

impl SummarizedDistribution {
    pub fn minimum(&self) -> f64 {
        self.percentiles[0]
    }

    pub fn maximum(&self) -> f64 {
        self.percentiles[20]
    }

    pub fn median(&self) -> f64 {
        self.percentiles[10]
    }
}

impl FromIterator<f64> for SummarizedDistribution {
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        let summary: DistributionSummary<21> = iter.into_iter().collect();
        let percentiles = summary.values;
        SummarizedDistribution { percentiles }
    }
}

#[derive(Debug, Clone)]
pub struct GraphStats {
    pub covered_nodes: u32,
    pub covered_edges: u64,
    pub total_nodes: u32,
    pub total_edges: u64,
    pub statistics: AHashMap<StatisticsType, SummarizedDistribution>,
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
        let k = scoped_clusters.len();
        let covered_nodes = scoped_clusters
            .iter()
            .map(|c| &c.nodes)
            .cloned()
            .collect_vec();
        let covered_nodes = covered_nodes.union().len() as u32;
        debug!("covered nodes: {}", covered_nodes);
        let graph = &self.graph.graph;
        let acc = &self.graph.acc_num_edges;
        let unioned_edges: Vec<RoaringTreemap> = scoped_clusters
            .par_iter()
            .progress_count(k as u64)
            .map(|c| {
                let tm = RoaringTreemap::from_sorted_iter(c.nodes.iter().flat_map(|u| {
                    let edges = &graph.nodes[u as usize].edges;
                    let shift = acc[u as usize];
                    edges
                        .into_iter()
                        .enumerate()
                        .filter_map(move |(offset, &v)| {
                            if c.nodes.contains(v as u32) {
                                Some(shift + offset as u64)
                            } else {
                                None
                            }
                        })
                }))
                .unwrap();
                tm
            })
            .collect();
        let covered_edges = unioned_edges.union().len() as u64;
        let mut statistics_type = vec![
            StatisticsType::Mcd,
            StatisticsType::Size,
            StatisticsType::Modularity,
            StatisticsType::DeviationTreeness,
            StatisticsType::Density,
        ];
        let mut cpm_val = 0.0;
        let mut mod_val = 1.0;
        match self.clustering.source {
            ClusteringSource::Unknown => {}
            ClusteringSource::Cpm(r) => {
                statistics_type.push(StatisticsType::Cpm);
                cpm_val = r;
            }
            ClusteringSource::Modularity(r) => {
                mod_val = r;
            }
        }
        let statistics = AHashMap::from_iter(statistics_type.into_iter().map(|t| {
            (t.clone(), {
                let c: Vec<f64> = scoped_clusters
                    .par_iter()
                    .map(|c| match t.clone() {
                        StatisticsType::Mcd => c.mcd as f64,
                        StatisticsType::Cpm => {
                            calc_cpm_resolution(c.m as usize, c.n as usize, cpm_val)
                        }
                        StatisticsType::Size => c.n as f64,
                        StatisticsType::Modularity => calc_modularity_resolution(
                            c.m as usize,
                            (c.m * 2 + c.c) as usize,
                            graph.m(),
                            mod_val,
                        ),
                        StatisticsType::Conductance => {
                            c.c as f64 / c.vol.min(c.m * 2 - c.vol) as f64
                        }
                        StatisticsType::DeviationTreeness => (c.m - c.n + 1) as f64 / c.n as f64,
                        StatisticsType::Density => c.m as f64 / choose2(c.n as usize) as f64,
                    })
                    .collect();
                c.into_iter().collect()
            })
        }));
        GraphStats {
            covered_nodes,
            covered_edges,
            total_nodes: graph.n() as u32,
            total_edges: graph.m() as u64,
            statistics,
        }
    }
}
