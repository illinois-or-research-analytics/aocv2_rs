use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    rc::Rc,
    sync::Arc,
};

use ahash::AHashMap;
use indicatif::ParallelProgressIterator;
use itertools::Itertools;
use rayon::prelude::{
    FromParallelIterator, IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    ParallelBridge, ParallelIterator,
};
use roaring::{MultiOps, RoaringBitmap, RoaringTreemap};
use tracing::debug;

use crate::{
    quality::DistributionSummary,
    utils::{calc_cpm_resolution, calc_modularity_resolution, choose2, self},
    Clustering, DefaultGraph, PackedClustering,
};

#[derive(Debug, Clone)]
pub enum ClusteringSource {
    Unknown,
    Cpm(f64),
    Modularity(f64),
}

#[derive(Debug, Clone, PartialEq, Hash, Eq, strum_macros::Display)]
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
    pub fn load_from_bitmap(g: &DefaultGraph, nodes: RoaringBitmap) -> Self {
        let n = nodes.len() as u64;
        let nodeset: RoaringBitmap = nodes;
        let mut m = 0;
        let mut c = 0;
        let mut vol = 0;
        let mut mcd = (g.m() + 1) as u64;
        for u in &nodeset {
            let adj = &g.nodes[u as usize].edges;
            vol += adj.len() as u64;
            let mut inside_connectivity = 0;
            for &v in adj {
                if nodeset.contains(v as u32) {
                    m += 1;
                    inside_connectivity += 1;
                } else {
                    c += 1;
                }
            }
            mcd = mcd.min(inside_connectivity);
        }
        if mcd == (g.m() + 1) as u64 {
            mcd = 0;
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
    pub fn load_from_slice(g: &DefaultGraph, nodes: &[u32]) -> RichCluster {
        let nodeset: RoaringBitmap = nodes.iter().copied().collect();
        RichCluster::load_from_bitmap(g, nodeset)
    }
}

pub struct RichClustering<const O: bool> {
    pub graph: Arc<EnrichedGraph>,
    pub clusters: BTreeMap<u64, RichCluster>,
    pub source: ClusteringSource,
    pub node_covers: Vec<RoaringBitmap>,
}

pub struct ClusteringHandle<const O: bool> {
    pub graph: Arc<EnrichedGraph>,
    pub clustering: Arc<RichClustering<O>>,
    pub cluster_ids: BTreeSet<u64>,
    pub covered_nodes: RoaringBitmap,
    pub node_multiplicity: Vec<u32>,
}

impl ClusteringHandle<true> {
    pub fn size_diff(&self, rhs: &RichClustering<true>) -> (u32, SummarizedDistribution) {
        let mut dist = vec![];
        for (id, cluster) in &self.clustering.clusters {
            if self.cluster_ids.contains(id) {
                dist.push((rhs.clusters[id].n as f64 - cluster.n as f64) / cluster.n as f64);
            }
        }
        let cnt = dist.iter().filter(|&&it| it > 0.0).count();
        (cnt as u32, dist.into_iter().collect())
    }
}

impl RichClustering<true> {
    pub fn universe_handle(clus: Arc<RichClustering<true>>) -> ClusteringHandle<true> {
        let mut cluster_ids = BTreeSet::new();
        for (id, _) in &clus.clusters {
            cluster_ids.insert(*id);
        }
        ClusteringHandle::<true>::new(clus, cluster_ids)
    }
}

impl<const O: bool> RichClustering<O> {
    pub fn pack_from_file<P>(graph: Arc<EnrichedGraph>, p: P) -> anyhow::Result<Self>
    where
        P: AsRef<std::path::Path>,
    {
        let path = PathBuf::new().join(p);
        let raw_graph = &graph.graph;
        if path.extension().unwrap() == "lz4" {
            let clus : PackedClustering = utils::read_compressed_bincode(&path)?;
            let k = clus.clusters.len();
            let clusters = BTreeMap::from_par_iter(
                clus.clusters
                    .into_par_iter()
                    .progress_count(k as u64)
                    .map(|(k, c)| {
                        (
                            k as u64,
                            RichCluster::load_from_bitmap(
                                raw_graph,
                                c,
                            ),
                        )
                    }),
            );
            let mut node_covers = vec![RoaringBitmap::new(); graph.graph.n()];
            for (cid, k) in clusters.iter() {
                for n in k.nodes.iter() {
                    node_covers[n as usize].insert(*cid as u32);
                }
            }
            Ok(RichClustering {
                graph,
                clusters,
                source: ClusteringSource::Unknown,
                node_covers,
            })
        } else {
            Ok(Self::pack_from_clustering(graph.clone(), Clustering::parse_from_file(raw_graph, &path, false)?))
        }
    }

    pub fn pack_from_clustering(graph: Arc<EnrichedGraph>, clus: Clustering) -> RichClustering<O> {
        let k = clus.clusters.len();
        let raw_graph = &graph.graph;
        let clusters =
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
        let mut node_covers = vec![RoaringBitmap::new(); graph.graph.n()];
        for (cid, k) in clusters.iter() {
            for n in k.nodes.iter() {
                node_covers[n as usize].insert(*cid as u32);
            }
        }
        RichClustering {
            graph,
            clusters,
            source: ClusteringSource::Unknown,
            node_covers,
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
    pub num_clusters: u32,
    pub covered_nodes: u32,
    pub covered_edges: u64,
    pub total_nodes: u32,
    pub total_edges: u64,
    pub statistics: AHashMap<StatisticsType, SummarizedDistribution>,
}

impl ClusteringHandle<true> {
    pub fn new(clus: Arc<RichClustering<true>>, cluster_ids: BTreeSet<u64>) -> Self {
        let covered_nodes: Vec<_> = cluster_ids
            .iter()
            .map(|&it| clus.clusters[&it].nodes.clone())
            .collect();
        let covered_nodes = covered_nodes.union();
        let mut node_multiplicity = vec![0u32; clus.graph.graph.n()];
        for cid in cluster_ids.iter().map(|it| clus.clusters.get(it).unwrap()) {
            for n in cid.nodes.iter() {
                node_multiplicity[n as usize] += 1;
            }
        }
        for i in 0..node_multiplicity.len() {
            if node_multiplicity[i] == 0 {
                node_multiplicity[i] = 1;
            }
        }
        ClusteringHandle {
            graph: clus.graph.clone(),
            clustering: clus,
            cluster_ids,
            covered_nodes,
            node_multiplicity,
        }
    }
    pub fn stats(&self) -> GraphStats {
        let clusters = &self.clustering.clusters;
        let graph = &self.graph.graph;
        let labels = &self.clustering.node_covers;
        let clusters_map: RoaringBitmap = self.cluster_ids.iter().map(|it| *it as u32).collect();
        let scoped_clusters = self
            .cluster_ids
            .iter()
            .map(|k| clusters.get(k).unwrap())
            .collect_vec();
        let k = scoped_clusters.len();
        let covered_nodes = self.covered_nodes.len() as u32;
        debug!("covered nodes: {}", covered_nodes);
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
            num_clusters: k as u32,
            covered_nodes,
            covered_edges: covered_edges as u64,
            total_nodes: graph.n() as u32,
            total_edges: graph.m() as u64,
            statistics,
        }
    }
}