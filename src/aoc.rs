use crate::io::*;
use crate::utils::choose2;
use crate::{utils, Cluster, Clustering, Graph, Node};
use itertools::Itertools;
use nom::{
    branch::alt,
    sequence::{tuple}, Parser,
};
use rayon::prelude::*;
use std::cmp::Reverse;
use tracing::debug;

pub trait AugmentingConfig: Sized {
    type Augmenter: Augmenter<Self>;
    fn augmenter(&self, bg: &Graph<Node>, c: &Cluster) -> Self::Augmenter;
}

#[derive(Clone)]
struct AugmentByMcd {
    // purposefully left empty as there are no parameters to specify
}

impl AugmentingConfig for AugmentByMcd {
    type Augmenter = McdKAugmenter;

    fn augmenter(&self, bg: &Graph<Node>, c: &Cluster) -> Self::Augmenter {
        McdKAugmenter::new(c.mcd(bg).unwrap_or_default(), bg, c)
    }
}

#[derive(Clone)]
struct AugmentByK {
    pub k: usize,
}

impl AugmentingConfig for AugmentByK {
    type Augmenter = McdKAugmenter;

    fn augmenter(&self, bg: &Graph<Node>, c: &Cluster) -> Self::Augmenter {
        McdKAugmenter::new(self.k, bg, c)
    }
}

#[derive(Clone)]
struct AugmentByMod {
    pub resolution: f64,
}

#[derive(Clone)]
struct AugmentByCpm {
    pub resolution: f64,
}

impl AugmentingConfig for AugmentByCpm {
    type Augmenter = CpmAugmenter;

    fn augmenter(&self, bg: &Graph<Node>, c: &Cluster) -> Self::Augmenter {
        let ls = bg.num_edges_inside(&c.core());
        let total_nodes = bg.n();
        CpmAugmenter {
            resolution: self.resolution,
            total_nodes,
            ls,
            cpm: ls as f64 - choose2(total_nodes) as f64 * self.resolution,
        }
    }
}

impl AugmentingConfig for AugmentByMod {
    type Augmenter = ModularityAugmenter;

    fn augmenter(&self, bg: &Graph<Node>, c: &Cluster) -> Self::Augmenter {
        let total_l = bg.m();
        let ls = bg.num_edges_inside(&c.core());
        let ds = c
            .core_nodes
            .iter()
            .map(|&n| bg.nodes[n].degree())
            .sum::<usize>();
        let modularity = utils::calc_modularity_resolution(ls, ds, total_l, self.resolution);
        ModularityAugmenter {
            resolution: self.resolution,
            modularity,
            total_l,
            ls,
            ds,
        }
    }
}

pub trait Augmenter<T: AugmentingConfig> {
    fn construct(t: &T, bg: &Graph<Node>, c: &Cluster) -> T::Augmenter {
        t.augmenter(bg, c)
    }
    fn query(&mut self, bg: &Graph<Node>, c: &Cluster, node: &Node) -> bool;
}

#[derive(Debug, Clone)]
pub enum AocConfig {
    Mcd(),
    K(usize),
    Mod(f64),
    Cpm(f64),
}

pub fn parse_aoc_config(s: &str) -> Result<AocConfig, String> {
    let mut pc = alt((
        tuple((token("cpm"), nom::number::complete::double)).map(|(_, k)| AocConfig::Cpm(k as f64)),
        tuple((token("mod"), nom::number::complete::double)).map(|(_, k)| AocConfig::Mod(k as f64)),
        alt((token("m"), token("mcd"))).map(|_| AocConfig::Mcd()),
        tuple((token("k"), decimal)).map(|(_, k)| AocConfig::K(k.parse::<usize>().unwrap())),
    ));
    let (rest, config) = pc(s).map_err(|e| format!("{:?}", e))?;
    if rest.is_empty() {
        Ok(config)
    } else {
        Err(format!("Could not parse config: {}", s))
    }
}

#[derive(Debug, Clone)]
struct McdKAugmenter {
    threshold: usize,
    total_l: usize,
    ls: usize,
    ds: usize,
}

impl McdKAugmenter {
    fn query(&mut self, _bg: &Graph<Node>, c: &Cluster, node: &Node) -> bool {
        let cluster_core = c.core();
        let cluster_all = c.all();
        let num_core_neighbors = node.edges_inside(&cluster_core).count();
        if num_core_neighbors < self.threshold {
            return false;
        }
        let ls_delta = node.edges_inside(&cluster_all).count() + self.ls;
        let ds_delta = node.degree() + self.ds;
        let new_modularity = utils::calc_modularity(ls_delta, ds_delta, self.total_l);
        if new_modularity <= 0.0 {
            return false;
        }
        self.ls = ls_delta;
        self.ds = ds_delta;
        true
    }

    fn new(threshold: usize, bg: &Graph<Node>, c: &Cluster) -> Self {
        McdKAugmenter {
            threshold,
            total_l: bg.m(),
            ls: bg.num_edges_inside(&c.core()),
            ds: c
                .core_nodes
                .iter()
                .map(|&n| bg.nodes[n].degree())
                .sum::<usize>(),
        }
    }
}

impl Augmenter<AugmentByMcd> for McdKAugmenter {
    fn query(&mut self, bg: &Graph<Node>, c: &Cluster, node: &Node) -> bool {
        McdKAugmenter::query(self, bg, c, node)
    }
}

impl Augmenter<AugmentByK> for McdKAugmenter {
    fn query(&mut self, bg: &Graph<Node>, c: &Cluster, node: &Node) -> bool {
        McdKAugmenter::query(self, bg, c, node)
    }
}

struct CpmAugmenter {
    resolution: f64,
    cpm: f64,
    total_nodes: usize,
    ls: usize,
}

impl Augmenter<AugmentByCpm> for CpmAugmenter {
    fn query(&mut self, _bg: &Graph<Node>, c: &Cluster, node: &Node) -> bool {
        let cluster_all = c.all();
        let ls_prime = node.edges_inside(&cluster_all).count() + self.ls;
        let cpm_prime = ls_prime as f64 - choose2(self.total_nodes + 1) as f64 * self.resolution;
        if cpm_prime < self.cpm {
            return false;
        }
        self.ls = ls_prime;
        self.cpm = cpm_prime;
        self.total_nodes += 1;
        true
    }
}

struct ModularityAugmenter {
    resolution: f64,
    modularity: f64,
    total_l: usize,
    ls: usize,
    ds: usize,
}

impl Augmenter<AugmentByMod> for ModularityAugmenter {
    fn query(&mut self, _bg: &Graph<Node>, c: &Cluster, node: &Node) -> bool {
        let cluster_all = c.all();
        let ls_prime = node.edges_inside(&cluster_all).count() + self.ls;
        let ds_prime = node.degree() + self.ds;
        let new_modularity =
            utils::calc_modularity_resolution(ls_prime, ds_prime, self.total_l, self.resolution);
        if new_modularity <= self.modularity {
            return false;
        }
        self.ls = ls_prime;
        self.ds = ds_prime;
        self.modularity = new_modularity;
        true
    }
}

pub fn augment_clusters_from_cli_config(
    bg: &Graph<Node>,
    clustering: &mut Clustering,
    candidate_ids: &mut [usize],
    config: &AocConfig,
) {
    match config {
        AocConfig::Mcd() => {
            let augmenter = AugmentByMcd {};
            augment_clusters(bg, clustering, candidate_ids, &augmenter);
        }
        AocConfig::K(k) => {
            let augmenter = AugmentByK { k: *k };
            augment_clusters(bg, clustering, candidate_ids, &augmenter);
        }
        AocConfig::Mod(resolution) => {
            let augmenter = AugmentByMod {
                resolution: *resolution,
            };
            augment_clusters(bg, clustering, candidate_ids, &augmenter);
        }
        AocConfig::Cpm(resolution) => {
            let augmenter = AugmentByCpm {
                resolution: *resolution,
            };
            augment_clusters(bg, clustering, candidate_ids, &augmenter);
        }
    }
}

pub fn augment_clusters<X: AugmentingConfig + Clone + Sync>(
    bg: &Graph<Node>,
    clustering: &mut Clustering,
    candidate_ids: &mut [usize],
    augmenting_config: &X,
) {
    candidate_ids.sort_by_key(|&it| Reverse(bg.nodes[it].degree()));
    clustering
        .clusters
        .par_iter_mut()
        .for_each(|(&cid, cluster)| {
            debug!("Augmenting cluster {} with size {}", cid, cluster.size());
            let mut augmenter = X::augmenter(augmenting_config, bg, cluster);
            if cluster.is_singleton() {
                return;
            }
            let cluster_core = &cluster.core_nodes;
            let viable_candidates = candidate_ids
                .iter()
                .filter(|it| !cluster_core.contains(*it))
                .map(|&it| &bg.nodes[it])
                .collect_vec();
            for cand in viable_candidates {
                if augmenter.query(bg, cluster, cand) {
                    cluster.add_periphery(cand.id);
                }
            }
        });
}
