//! Data science utilities. I don't have to write anything cleanly at this point
use core::num;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

use itertools::Itertools;
use rayon::prelude::IntoParallelRefIterator;
use rayon::prelude::ParallelIterator;
use serde::Deserialize;
use serde::Serialize;
use serde_with::serde_as;
use tracing::debug;

use crate::io::FilesSpecifier;
use crate::quality::files_and_labels;
use crate::quality::DistributionSummary;
use crate::AbstractSubset;
use crate::Cluster;
use crate::Clustering;
use crate::DefaultGraph;

#[derive(Hash, PartialEq, Serialize, Deserialize, Clone)]
pub enum ClusteringFilter {
    None(),
    NotTree(),
    OnlyTree(),
    SizeLowerBound(usize),
}

#[serde_as]
#[derive(Deserialize, Serialize)]
pub struct VerboseGlobalStatistics<const N: usize> {
    num_clusters: usize,
    node_coverage: f64,
    edge_coverage: f64,
    cluster_size: DistributionSummary<N>,
    modularity_score: DistributionSummary<N>,
    cpm_score: DistributionSummary<N>,
    mcd_score: DistributionSummary<N>,
    density_score: DistributionSummary<N>,
    treeness_score: DistributionSummary<N>,
    num_clusters_per_node: f64,
}

type StatisticsReport = BTreeMap<String, Vec<(Vec<ClusteringFilter>, VerboseGlobalStatistics<5>)>>;

pub struct ClusteringWithFilter<'a> {
    clustering: &'a Clustering,
    filter: Vec<ClusteringFilter>,
}

impl<'a> ClusteringWithFilter<'a> {
    fn new(clustering: &'a Clustering, filter: Vec<ClusteringFilter>) -> Self {
        Self { clustering, filter }
    }

    fn clusters(&'a self, g: &'a DefaultGraph) -> impl Iterator<Item = &Cluster> + 'a {
        self.clustering.clusters.values().filter(move |c| {
            self.filter.iter().all(|f| match f {
                ClusteringFilter::None() => true,
                ClusteringFilter::NotTree() => {
                    let n = c.size();
                    let m = g.num_edges_inside(&c.core());
                    m > n - 1
                }
                ClusteringFilter::SizeLowerBound(bound) => c.size() >= *bound,
                ClusteringFilter::OnlyTree() => {
                    let n = c.size();
                    let m = g.num_edges_inside(&c.core());
                    m == n - 1
                }
            })
        })
    }
}

impl VerboseGlobalStatistics<5> {
    pub fn from_clustering_with_filter(
        g: &DefaultGraph,
        clustering: &ClusteringWithFilter,
        resolution: f64,
    ) -> Self {
        let num_c = clustering.clusters(g).count();
        let num_nodes_covered = BTreeSet::<usize>::from_iter(
            clustering
                .clusters(g)
                .flat_map(|x| x.core().each_node_id().copied().collect_vec()),
        )
        .len();
        let node_coverage = num_nodes_covered as f64 / g.n() as f64;
        let edge_coverage = BTreeSet::<(usize, usize)>::from_iter(
            clustering
                .clusters(g)
                .flat_map(|x| g.each_edge_inside(&x.core()).collect_vec()),
        )
        .len() as f64
            / g.m() as f64;
        let cluster_size = clustering.clusters(g).map(|x| x.size()).collect();
        let modularity_score = clustering
            .clusters(g)
            .map(|x| g.modularity_of(&x.core(), 1.0))
            .collect();
        let cpm_score = clustering
            .clusters(g)
            .map(|x| g.cpm_of(&x.core(), resolution))
            .collect();
        let mcd_score = clustering
            .clusters(g)
            .map(|x| g.mcd_of(&x.core()).unwrap())
            .collect();
        let density_score = clustering
            .clusters(g)
            .map(|x| g.edge_density_inside(&x.core()))
            .collect();
        let treeness_score = clustering
            .clusters(g)
            .map(|x| g.treeness_of(&x.core()))
            .collect();
        let num_clusters_per_node = num_c as f64 / g.n() as f64;
        Self {
            num_clusters: num_c,
            node_coverage,
            edge_coverage,
            cluster_size,
            modularity_score,
            cpm_score,
            mcd_score,
            density_score,
            treeness_score,
            num_clusters_per_node,
        }
    }
}

pub fn calculate_statistics(
    configs: Vec<Vec<ClusteringFilter>>,
    files: &FilesSpecifier,
    resolution: f64,
    g: &DefaultGraph,
) -> anyhow::Result<StatisticsReport> {
    let mut report =
        BTreeMap::<String, Vec<(Vec<ClusteringFilter>, VerboseGlobalStatistics<5>)>>::new();
    for (filepath, label) in files_and_labels(files) {
        debug!("Processing {}", filepath);
        let label = label.unwrap_or_else(|| &filepath);
        let clus = Clustering::parse_from_file(g, &filepath, false)?;
        let stats = configs
            .par_iter()
            .map(|config| {
                let filtered_clustering = ClusteringWithFilter::new(&clus, config.clone());
                let stats = VerboseGlobalStatistics::from_clustering_with_filter(
                    g,
                    &filtered_clustering,
                    resolution,
                );
                (config.clone(), stats)
            })
            .collect();
        report.insert(label.to_string(), stats);
    }
    Ok(report)
}
