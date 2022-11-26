use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    sync::{atomic::AtomicU32, Arc, Mutex},
};

use ahash::AHashMap;
use itertools::Itertools;
use once_cell::sync::OnceCell;
use rayon::prelude::{
    FromParallelIterator, IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    ParallelBridge, ParallelIterator,
};
use roaring::{MultiOps, RoaringBitmap, RoaringTreemap};

use crate::{
    quality::DistributionSummary,
    utils::{self, calc_cpm_resolution, calc_modularity_resolution, choose2},
    Clustering, DefaultGraph, Node, PackedClustering,
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
    pub acc_num_edges: Vec<u64>,
    pub singletons: RoaringBitmap,
}

impl EnrichedGraph {
    pub fn from_graph(graph: DefaultGraph) -> EnrichedGraph {
        let mut acc = vec![0u64; graph.n() + 1];
        for i in 0..graph.n() {
            acc[i + 1] = acc[i] + graph.nodes[i].degree() as u64;
        }
        let bitset = (0..graph.n() as u32)
            .into_iter()
            .filter(|it| graph.nodes[*it as usize].degree() == 1)
            .collect();
        EnrichedGraph {
            graph,
            acc_num_edges: acc,
            singletons: bitset,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RichCluster {
    pub nodes: RoaringBitmap,
    pub edges: OnceCell<RoaringTreemap>,
    pub n: u64,
    pub m: u64,
    pub c: u64,
    pub mcd: u64,
    pub vol: u64,
}

impl RichCluster {
    pub fn from_single_node(graph: &DefaultGraph, node_id: usize) -> RichCluster {
        let node = &graph.nodes[node_id];
        RichCluster {
            nodes: [node_id as u32].into_iter().collect(),
            edges: OnceCell::new(),
            n: 1,
            m: 0,
            c: node.degree() as u64,
            mcd: 0,
            vol: node.degree() as u64,
        }
    }
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
            edges: OnceCell::new(),
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
    pub clusters: BTreeMap<u32, RichCluster>,
    pub singleton_clusters: Vec<RichCluster>,
    pub cover: RoaringBitmap,
    pub singleton_mask: RoaringBitmap,
    pub source: ClusteringSource,
    pub edge_coverage_cache: Arc<Mutex<AHashMap<BTreeSet<u32>, u64>>>,
}

pub struct ClusteringHandle<const O: bool> {
    pub graph: Arc<EnrichedGraph>,
    pub clustering: Arc<RichClustering<O>>,
    pub cluster_ids: RoaringBitmap,
    pub covered_nodes: RoaringBitmap,
    pub node_multiplicity: Vec<u32>,
    pub is_overlapping: bool,
    pub has_singletons: bool,
    pub num_covered_edges: OnceCell<u64>,
}

impl ClusteringHandle<true> {
    pub fn size_diff(&self, rhs: &RichClustering<true>) -> (u32, SummarizedDistribution) {
        let mut dist = vec![];
        for (id, cluster) in &self.clustering.clusters {
            if self.cluster_ids.contains(*id as u32) {
                dist.push((rhs.clusters[id].n as f64 - cluster.n as f64) / cluster.n as f64);
            }
        }
        let cnt = dist.iter().filter(|&&it| it > 0.0).count();
        (cnt as u32, dist.into_iter().collect())
    }
}

impl RichClustering<true> {
    pub fn universe_handle(clus: Arc<RichClustering<true>>) -> ClusteringHandle<true> {
        let mut cluster_ids = RoaringBitmap::new();
        for (id, _) in &clus.clusters {
            cluster_ids.insert(*id as u32);
        }
        ClusteringHandle::<true>::new(clus, cluster_ids, true)
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
            let clus: PackedClustering = utils::read_compressed_bincode(&path)?;
            let k = clus.clusters.len();
            let clusters = BTreeMap::from_par_iter(
                clus.clusters
                    .into_par_iter()
                    .map(|(k, c)| (k as u32, RichCluster::load_from_bitmap(raw_graph, c))),
            );
            Ok(Self::from_graph_and_clustering(graph, clusters))
        } else {
            Ok(Self::pack_from_clustering(
                graph.clone(),
                Clustering::parse_from_file(raw_graph, &path, false)?,
            ))
        }
    }

    pub fn from_graph_and_clustering(
        graph: Arc<EnrichedGraph>,
        clusters: BTreeMap<u32, RichCluster>,
    ) -> RichClustering<O> {
        let raw_graph = &graph.graph;
        let cover = clusters
            .values()
            .map(|it| it.nodes.clone())
            .collect::<Vec<_>>()
            .union();
        let mut singleton_mask = RoaringBitmap::new();
        singleton_mask.insert_range(0..(raw_graph.n() as u32));
        singleton_mask -= &cover;
        RichClustering {
            graph: graph.clone(),
            clusters,
            source: ClusteringSource::Unknown,
            singleton_clusters: singleton_mask
                .iter()
                .map(|it| RichCluster::from_single_node(raw_graph, it as usize))
                .collect(),
            cover,
            singleton_mask,
            edge_coverage_cache: Arc::new(Mutex::new(AHashMap::new())),
        }
    }

    pub fn pack_from_clustering(graph: Arc<EnrichedGraph>, clus: Clustering) -> RichClustering<O> {
        let k = clus.clusters.len();
        let raw_graph = &graph.graph;
        let clusters = BTreeMap::from_par_iter(clus.clusters.into_par_iter().map(|(k, c)| {
            (
                k as u32,
                RichCluster::load_from_slice(
                    raw_graph,
                    &c.core_nodes
                        .iter()
                        .cloned()
                        .map(|it| it as u32)
                        .collect_vec(),
                ),
            )
        }));
        Self::from_graph_and_clustering(graph, clusters)
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
    pub fn new(
        clus: Arc<RichClustering<true>>,
        cluster_ids: RoaringBitmap,
        has_singletons: bool,
    ) -> Self {
        let covered_nodes: Vec<_> = cluster_ids
            .iter()
            .map(|it| clus.clusters[&it].nodes.clone())
            .collect();
        let covered_nodes = covered_nodes.union();
        let node_multiplicity = (0..clus.graph.graph.n())
            .map(|_| AtomicU32::new(0))
            .collect_vec();
        cluster_ids
            .iter()
            .par_bridge()
            .map(|it| clus.clusters.get(&it).unwrap())
            .for_each(|cid| {
                for n in cid.nodes.iter() {
                    node_multiplicity[n as usize]
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            });
        let num_covered = covered_nodes.len();
        let sum_cluster_length = cluster_ids
            .iter()
            .map(|it| clus.clusters[&it].nodes.len())
            .sum::<u64>();
        ClusteringHandle {
            graph: clus.graph.clone(),
            clustering: clus,
            cluster_ids,
            covered_nodes,
            node_multiplicity: node_multiplicity
                .into_iter()
                .map(|it| it.into_inner())
                .collect(),
            is_overlapping: num_covered < sum_cluster_length,
            has_singletons,
            num_covered_edges: OnceCell::new(),
        }
    }

    pub fn get_covered_edges(&self) -> u64 {
        let covered_edges = self.num_covered_edges.get_or_init(|| {
            let clusters = &self.clustering.clusters;
            let scoped_clusters = self
                .cluster_ids
                .iter()
                .map(|k| clusters.get(&k).unwrap())
                .collect_vec();
            let graph = &self.graph.graph;
            if self.is_overlapping {
                let mut hm = self.clustering.edge_coverage_cache.lock().unwrap();
                hm.entry(self.cluster_ids.iter().collect())
                    .or_insert_with(|| {
                        let acc = &self.graph.acc_num_edges;
                        let unioned_edges: Vec<&RoaringTreemap> = scoped_clusters
                            .par_iter()
                            .map(|c| {
                                let r = c.edges.get_or_init(|| {
                                    let tm = RoaringTreemap::from_sorted_iter(c.nodes.iter().flat_map(
                                        |u| {
                                            let edges = &graph.nodes[u as usize].edges;
                                            let shift = acc[u as usize];
                                            edges.iter().enumerate().filter_map(move |(offset, &v)| {
                                                if c.nodes.contains(v as u32) {
                                                    Some(shift + offset as u64)
                                                } else {
                                                    None
                                                }
                                            })
                                        },
                                    ))
                                    .unwrap();
                                    tm
                                });
                                r
                            })
                            .collect();
                        (unioned_edges.union().len() / 2) as u64
                    })
                    .clone()
            } else {
                scoped_clusters.iter().map(|it| it.m).sum()
            }
        });
        *covered_edges
    }

    pub fn get_covered_nodes(&self) -> u32 {
        self.covered_nodes.len() as u32
            + if self.has_singletons {
                self.clustering.singleton_clusters.len() as u32
            } else {
                0
            }
    }

    pub fn stats(&self) -> GraphStats {
        let clusters = &self.clustering.clusters;
        let graph = &self.graph.graph;
        let scoped_clusters = self
            .cluster_ids
            .iter()
            .map(|k| clusters.get(&k).unwrap())
            .collect_vec();
        let k = scoped_clusters.len();
        let covered_edges = self.get_covered_edges();
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
                    .clone()
                    .into_par_iter()
                    .chain(if self.has_singletons {
                        self.clustering.singleton_clusters.par_iter()
                    } else {
                        [].par_iter()
                    })
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
            covered_nodes: self.get_covered_nodes(),
            covered_edges: covered_edges as u64,
            total_nodes: graph.n() as u32,
            total_edges: graph.m() as u64,
            statistics,
        }
    }
}
