use std::path::Path;

use anyhow::bail;
use rayon::prelude::FromParallelIterator;
use rayon::prelude::IntoParallelIterator;
use rayon::prelude::IntoParallelRefIterator;
use rayon::prelude::ParallelBridge;
use rayon::prelude::ParallelIterator;
use serde::Deserialize;
use serde::Serialize;
use serde_with::serde_as;

use crate::aoc::AocConfig;
use crate::base::Graph;
use crate::io::FilesSpecifier;
use crate::utils::calc_cpm_resolution;
use crate::utils::calc_modularity;
use crate::AbstractSubset;
use crate::DefaultGraph;

use crate::utils::calc_modularity_resolution;
use crate::Clustering;
use crate::Node;
pub fn modularity<'a, X>(g: &'a DefaultGraph, c: &'a X) -> f64
where
    X: AbstractSubset<'a>,
{
    let big_l = g.m();
    let ls = g.num_edges_inside(c);
    let ds = c
        .each_node_id()
        .map(|&n| g.nodes[n].degree())
        .sum::<usize>();
    calc_modularity(ls, ds, big_l)
}

pub fn mcd<'a, X>(g: &'a DefaultGraph, c: &'a X) -> usize
where
    X: AbstractSubset<'a>,
{
    return g.degrees_inside(c).min().unwrap_or(0);
}

#[derive(Serialize, Deserialize)]
pub struct ClusterInformation {
    pub variant: Option<String>,
    cid: Option<usize>,
    n: usize,
    m: usize,
    mcd: usize,
    modularity: f64,
    cpm: f64,
    conductance: f64,
}

impl ClusterInformation {
    pub fn from_single_cluster<'a, X>(g: &'a DefaultGraph, c: &'a X, quality: &AocConfig) -> Self
    where
        X: AbstractSubset<'a>,
    {
        let resolution = match quality {
            AocConfig::Mod(r) => *r,
            AocConfig::Cpm(r) => *r,
            _ => 1.0,
        };
        let n = c.each_node_id().count();
        let m = g.num_edges_inside(c);
        let mcd = mcd(g, c);

        let big_l = g.m();
        let ls = g.num_edges_inside(c);
        let ds = c
            .each_node_id()
            .map(|&n| g.nodes[n].degree())
            .sum::<usize>();
        let modularity = calc_modularity_resolution(ls, ds, big_l, 1.0);
        let cpm = calc_cpm_resolution(ls, n, resolution);
        let conductance = g.conductance_of(c);
        Self {
            variant: None,
            cid: None,
            n,
            m,
            mcd,
            modularity,
            cpm,
            conductance,
        }
    }

    // TODO: see if it is right to set mcd only among the core nodes
    pub fn vec_from_clustering(
        g: &DefaultGraph,
        clus: &Clustering,
        quality: &AocConfig,
    ) -> Vec<Self> {
        clus.clusters
            .par_iter()
            .map(|(&k, v)| {
                let mut record = Self::from_single_cluster(g, &v.core(), quality);
                record.cid = Some(k);
                record
            })
            .collect()
    }
}

#[serde_as]
#[derive(Deserialize, Serialize)]
pub struct DistributionSummary<const N: usize> {
    #[serde_as(as = "Box<[_; N]>")]
    pub values: Box<[f64; N]>,
}

impl<const N: usize> FromIterator<f64> for DistributionSummary<N> {
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        let mut stats = inc_stats::Percentiles::new();
        for v in iter {
            stats.add(v);
        }
        let mut sample_points: Vec<f64> = vec![0.0];
        let step = 1.0 / (N - 1) as f64;
        for i in 0..(N - 2) {
            sample_points.push((i + 1) as f64 * step);
        }
        sample_points.push(1.0);
        let ans = match stats.percentiles(&sample_points) {
            Err(_) | Ok(None) => vec![0.0; N],
            Ok(Some(v)) => v,
        };
        let mut values = [0.0; N];
        values.copy_from_slice(&ans);
        Self {
            values: Box::new(values),
        }
    }
}

impl <const N: usize> FromParallelIterator<f64> for DistributionSummary<N> {
    fn from_par_iter<T: IntoParallelIterator<Item = f64>>(iter: T) -> Self {
        iter.into_par_iter().collect()
    }
}

impl<const N: usize> FromIterator<usize> for DistributionSummary<N> {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        iter.into_iter().map(|x| x as f64).collect()
    }
}

#[serde_as]
#[derive(Deserialize, Serialize)]
pub struct GlobalStatistics<const N: usize> {
    pub node_coverage: f64,
    pub edge_coverage: f64,
    pub modularity: f64,
    pub num_clusters: usize,
    pub cluster_size: DistributionSummary<N>,
    pub cluster_mcd: DistributionSummary<N>,
}

impl<const N: usize> GlobalStatistics<N> {
    pub fn from_clustering(g: &DefaultGraph, clus: &Clustering) -> Self {
        let num_clusters = clus.clusters.len();
        let node_coverage =
            clus.clusters.values().map(|x| x.size()).sum::<usize>() as f64 / g.n() as f64;
        let edge_coverage = clus
            .clusters
            .values()
            .par_bridge()
            .map(|x| g.num_edges_inside(&x.core()))
            .sum::<usize>() as f64
            / g.m() as f64;
        let modularity = clus
            .clusters
            .values()
            .par_bridge()
            .map(|x| modularity(g, &x.core()))
            .sum::<f64>();
        let cluster_size = clus.clusters.iter().map(|(_, v)| v.size()).collect();
        let cluster_mcd_buf: Vec<_> = clus
            .clusters
            .par_iter()
            .map(|(_, v)| mcd(g, &v.core()))
            .collect();
        let cluster_mcd = cluster_mcd_buf.into_iter().collect();
        Self {
            node_coverage,
            edge_coverage,
            modularity,
            num_clusters,
            cluster_size,
            cluster_mcd,
        }
    }

    pub fn from_clustering_with_local(
        g: &DefaultGraph,
        clus: &Clustering,
        local_info: &[ClusterInformation],
    ) -> Self {
        let num_clusters = clus.clusters.len();
        let node_coverage = local_info.iter().map(|it| it.n).sum::<usize>() as f64 / g.n() as f64;
        let edge_coverage = local_info.iter().map(|it| it.m).sum::<usize>() as f64 / g.m() as f64;
        let modularity = clus
            .clusters
            .values()
            .par_bridge()
            .map(|x| modularity(g, &x.core()))
            .sum::<f64>();
        let cluster_size = clus.clusters.iter().map(|(_, v)| v.size()).collect();
        let cluster_mcd = local_info.iter().map(|it| it.mcd).collect();
        Self {
            node_coverage,
            edge_coverage,
            modularity,
            num_clusters,
            cluster_size,
            cluster_mcd,
        }
    }
}

pub fn files_and_labels<'a>(spec: &'a FilesSpecifier) -> Vec<(String, Option<&'a str>)> {
    match spec {
        FilesSpecifier::SingleFile(f) => vec![(f.clone(), None)],
        FilesSpecifier::FileFamily(f, l) => l
            .iter()
            .map(|x| (f.replace("{}", x), Some(x.as_str())))
            .collect(),
    }
}

pub fn ensure_files_and_labels_exists(
    files_labels: &[(String, Option<&str>)],
) -> anyhow::Result<()> {
    for (f, _) in files_labels {
        if !Path::new(f).exists() {
            bail!("File {} does not exist", f);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::DistributionSummary;
    use std::iter::FromIterator;
    #[test]
    pub fn test_distribution_summary_three() {
        let summary = DistributionSummary::<3>::from_iter(1..=100);
        assert_eq!(1.0, summary.values[0]);
        assert_eq!(50.5, summary.values[1]);
        assert_eq!(100.0, summary.values[2]);
    }

    #[test]
    pub fn test_distribution_summary_five() {
        let summary = DistributionSummary::<5>::from_iter(1..=100);
        assert_eq!(1.0, summary.values[0]);
        assert_eq!(25.75, summary.values[1]);
        assert_eq!(50.5, summary.values[2]);
        assert_eq!(75.25, summary.values[3]);
        assert_eq!(100.0, summary.values[4]);
    }
}
