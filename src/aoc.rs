use crate::{utils, Clustering, Graph, Node};
use itertools::Itertools;
use std::cmp::Reverse;
// use rayon::prelude::ParallelBridge;
use rayon::prelude::*;
use tracing::debug;

pub enum AocConfig {
    AocM(),
    AocK(usize),
}

pub fn augment_clusters(
    bg: &Graph<Node>,
    clustering: &mut Clustering,
    candidate_ids: &mut [usize],
    config: AocConfig,
) {
    candidate_ids.sort_by_key(|&it| Reverse(bg.nodes[it].degree()));
    clustering.clusters.par_iter_mut().for_each(|(_, cluster)| {
        if cluster.is_singleton() {
            return;
        }
        let mcd = cluster.mcd(bg).unwrap();
        let threshold = match config {
            AocConfig::AocM() => mcd,
            AocConfig::AocK(k) => k,
        };
        let total_l = bg.m();
        let mut ls = bg.num_edges_inside(&cluster.core());
        let mut ds = cluster
            .core_nodes
            .iter()
            .map(|&n| bg.nodes[n].degree())
            .sum::<usize>();
        let cluster_core = &cluster.core_nodes;
        let viable_candidates = candidate_ids
            .iter()
            .filter(|it| !cluster_core.contains(*it))
            .map(|&it| &bg.nodes[it])
            .collect_vec();
        for cand in viable_candidates {
            let cluster_core = cluster.core();
            let cluster_all = cluster.all();
            let num_core_neighbors = cand.edges_inside(&cluster_core).count();
            if num_core_neighbors < threshold {
                continue;
            }
            let ls_delta = cand.edges_inside(&cluster_all).count() + ls;
            let ds_delta = cand.degree() + ds;
            let new_modularity = utils::calc_modularity(ls_delta, ds_delta, total_l);
            if new_modularity <= 0.0 {
                continue;
            }
            ls = ls_delta;
            ds = ds_delta;
            cluster.add_periphery(cand.id);
        }
    });
}
