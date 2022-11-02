use crate::misc::OnlineConductance;
use crate::utils::{choose2, NeighborhoodFilter};
use crate::{io::*, AbstractSubset};
use crate::{utils, Cluster, Clustering, Graph, Node};
use ahash::{AHashMap, AHashSet};
use indicatif::ParallelProgressIterator;
use itertools::Itertools;
use nom::{branch::alt, sequence::tuple, Parser};
use priority_queue::PriorityQueue;
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
        let total_nodes = c.size();
        let cpm = utils::calc_cpm_resolution(ls, total_nodes, self.resolution);
        CpmAugmenter {
            original_cpm: cpm,
            resolution: self.resolution,
            total_nodes,
            ls,
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
            original_modularity: modularity,
            resolution: self.resolution,
            total_l,
            ls,
            ds,
        }
    }
}

#[derive(Clone)]
struct AugmentByDensityThreshold {
    pub threshold: Option<f64>,
}

impl AugmentingConfig for AugmentByDensityThreshold {
    type Augmenter = DensityThresholdAugmenter;

    fn augmenter(&self, bg: &Graph<Node>, c: &Cluster) -> Self::Augmenter {
        let (n, m) = bg.count_n_m(&c.core());
        DensityThresholdAugmenter {
            threshold: self
                .threshold
                .unwrap_or_else(|| m as f64 / choose2(n) as f64),
            m,
        }
    }
}

pub trait Augmenter<T: AugmentingConfig> {
    fn construct(t: &T, bg: &Graph<Node>, c: &Cluster) -> T::Augmenter {
        t.augmenter(bg, c)
    }
    fn query(&mut self, bg: &Graph<Node>, c: &Cluster, node: &Node) -> bool;
    fn query_and_admit(&mut self, bg: &Graph<Node>, c: &mut Cluster, node: &Node) -> bool {
        let ans = self.query(bg, c, node);
        if ans {
            c.add_periphery(node.id);
        }
        ans
    }
    fn allows_earlystopping() -> bool {
        false
    }
}

#[derive(Clone)]
struct AugmentByMeanDegree {}

#[derive(Clone)]
struct AugmentByConductance {}

#[derive(Clone)]
struct MeanDegreeAugmenter {
    threshold: f64,
    ls: usize,
}

#[derive(Clone)]
struct ConductanceAugmenter {
    threshold: f64,
    online_conductance: OnlineConductance,
}

impl AugmentingConfig for AugmentByMeanDegree {
    type Augmenter = MeanDegreeAugmenter;
    fn augmenter(&self, bg: &Graph<Node>, c: &Cluster) -> MeanDegreeAugmenter {
        let ls = bg.num_edges_inside(&c.core());
        let threshold = ls as f64 / c.core_nodes.len() as f64;
        MeanDegreeAugmenter { threshold, ls }
    }
}

impl AugmentingConfig for AugmentByConductance {
    type Augmenter = ConductanceAugmenter;
    fn augmenter(&self, bg: &Graph<Node>, c: &Cluster) -> ConductanceAugmenter {
        let subset = c.core();
        let online_conductance = OnlineConductance::new(bg, &subset);
        ConductanceAugmenter {
            threshold: online_conductance.conductance(),
            online_conductance,
        }
    }
}

impl Augmenter<AugmentByMeanDegree> for MeanDegreeAugmenter {
    fn query(&mut self, _bg: &Graph<Node>, c: &Cluster, node: &Node) -> bool {
        let d = node.degree_inside(&c.core());
        if d <= 0 {
            return false;
        }
        let ls_delta = self.ls + d;
        let n_delta = c.size() + 1;
        if ls_delta as f64 / n_delta as f64 >= self.threshold {
            self.ls = ls_delta;
            true
        } else {
            false
        }
    }

    fn allows_earlystopping() -> bool {
        true
    }
}

impl Augmenter<AugmentByConductance> for ConductanceAugmenter {
    fn query(&mut self, bg: &Graph<Node>, c: &Cluster, node: &Node) -> bool {
        let threshold = self.threshold;
        let oc = &mut self.online_conductance;
        let (_, success) = oc.update_conductance_if(bg, node, &c.core(), |c| c <= threshold);
        success
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AocConfig {
    Mcd(),
    K(usize),
    Mod(f64),
    Cpm(f64),
    EdgeDensity(f64),
    Denser(),
    MeanDegree(),
    Conductance(),
}

pub fn parse_aoc_config(s: &str) -> Result<AocConfig, String> {
    let mut pc = alt((
        tuple((token("cpm"), nom::number::complete::double)).map(|(_, k)| AocConfig::Cpm(k as f64)),
        tuple((token("mod"), nom::number::complete::double)).map(|(_, k)| AocConfig::Mod(k as f64)),
        tuple((token("density"), nom::number::complete::double))
            .map(|(_, k)| AocConfig::EdgeDensity(k as f64)),
        token("denser").map(|_| AocConfig::Denser()),
        token("mean-degree").map(|_| AocConfig::MeanDegree()),
        token("conductance").map(|_| AocConfig::Conductance()),
        alt((token("mcd"), token("m"))).map(|_| AocConfig::Mcd()),
        tuple((token("k"), decimal)).map(|(_, k)| AocConfig::K(k.parse::<usize>().unwrap())),
    ));
    let (rest, config) = pc(s).map_err(|e| format!("{:?}", e))?;
    if rest.is_empty() {
        Ok(config)
    } else {
        Err(format!(
            "Could not parse config specifying candidate addition criterion: {}",
            s
        ))
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

    fn allows_earlystopping() -> bool {
        true
    }
}

impl Augmenter<AugmentByK> for McdKAugmenter {
    fn query(&mut self, bg: &Graph<Node>, c: &Cluster, node: &Node) -> bool {
        McdKAugmenter::query(self, bg, c, node)
    }

    fn allows_earlystopping() -> bool {
        true
    }
}

#[derive(Debug, Clone)]
struct DensityThresholdAugmenter {
    threshold: f64,
    m: usize,
}

impl Augmenter<AugmentByDensityThreshold> for DensityThresholdAugmenter {
    fn query(&mut self, _bg: &Graph<Node>, c: &Cluster, node: &Node) -> bool {
        let cluster_all = c.all();
        let n_prime = c.size() + 1;
        let d = node.degree_inside(&cluster_all);
        if d <= 0 {
            return false;
        }
        let m_prime = self.m + d;
        let density_prime = m_prime as f64 / choose2(n_prime) as f64;
        if density_prime >= self.threshold {
            self.m = m_prime;
            true
        } else {
            false
        }
    }

    fn allows_earlystopping() -> bool {
        true
    }
}

struct CpmAugmenter {
    original_cpm: f64,
    resolution: f64,
    total_nodes: usize,
    ls: usize,
}

impl CpmAugmenter {
    pub fn cpm(&self) -> f64 {
        utils::calc_cpm_resolution(self.ls, self.total_nodes, self.resolution)
    }
}

impl Augmenter<AugmentByCpm> for CpmAugmenter {
    fn query(&mut self, _bg: &Graph<Node>, c: &Cluster, node: &Node) -> bool {
        let cluster_all = c.all();
        let n_prime = c.size() + 1;
        let d = node.degree_inside(&cluster_all);
        if d <= 0 {
            return false;
        }
        let m_prime = self.ls + d;
        let cpm_prime = m_prime as f64 - choose2(n_prime) as f64 * self.resolution;
        if cpm_prime < self.original_cpm {
            return false;
        }
        self.total_nodes += 1;
        self.ls += d;
        true
    }

    fn allows_earlystopping() -> bool {
        true
    }
}

struct ModularityAugmenter {
    original_modularity: f64,
    resolution: f64,
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
        if new_modularity <= self.original_modularity {
            return false;
        }
        self.ls = ls_prime;
        self.ds = ds_prime;
        true
    }
}

pub fn augment_clusters_from_cli_config<'a, X: ExpandStrategy, S: AbstractSubset<'a> + Sync>(
    bg: &'a Graph<Node>,
    clustering: &mut Clustering,
    candidates: &'a S,
    config: &AocConfig,
) {
    match config {
        AocConfig::Mcd() => {
            let augmenter = AugmentByMcd {};
            X::expand(bg, clustering, candidates, &augmenter);
        }
        AocConfig::K(k) => {
            let augmenter = AugmentByK { k: *k };
            X::expand(bg, clustering, candidates, &augmenter);
        }
        AocConfig::Mod(resolution) => {
            let augmenter = AugmentByMod {
                resolution: *resolution,
            };
            X::expand(bg, clustering, candidates, &augmenter);
        }
        AocConfig::Cpm(resolution) => {
            let augmenter = AugmentByCpm {
                resolution: *resolution,
            };
            X::expand(bg, clustering, candidates, &augmenter);
        }
        AocConfig::EdgeDensity(gamma) => {
            let augmenter = AugmentByDensityThreshold {
                threshold: Some(*gamma),
            };
            X::expand(bg, clustering, candidates, &augmenter);
        }
        AocConfig::Denser() => {
            let augmenter = AugmentByDensityThreshold { threshold: None };
            X::expand(bg, clustering, candidates, &augmenter);
        }
        AocConfig::MeanDegree() => {
            let augmenter = AugmentByMeanDegree {};
            X::expand(bg, clustering, candidates, &augmenter);
        }
        AocConfig::Conductance() => {
            let augmenter = AugmentByConductance {};
            X::expand(bg, clustering, candidates, &augmenter);
        }
    }
}

// FIXME: this is duplicate code with the normal `augment_clusters` function
pub fn augment_clusters_local_expand<
    'a,
    X: AugmentingConfig + Clone + Sync,
    S: AbstractSubset<'a> + Sync,
>(
    bg: &'a Graph<Node>,
    clustering: &mut Clustering,
    candidates: &S,
    augmenting_config: &X,
) {
    let n_clusters = clustering.clusters.len() as u64;
    // let viable_candidates: AHashSet<usize> = AHashSet::from_iter(candidate_ids.iter().cloned());
    clustering
        .clusters
        .par_iter_mut()
        .progress_count(n_clusters)
        .for_each(|(&cid, cluster)| {
            debug!("Augmenting cluster {} with size {}", cid, cluster.size());
            let mut neighborhood_multiplicities: AHashMap<usize, usize> = AHashMap::new();
            let mut augmenter = X::augmenter(augmenting_config, bg, cluster);
            cluster
                .core_nodes
                .iter()
                .flat_map(|n| bg.nodes[*n].edges.iter())
                .for_each(|n| {
                    if !cluster.core_nodes.contains(n) {
                        *neighborhood_multiplicities.entry(*n).or_insert(0) += 1;
                    }
                });
            let mut pq: PriorityQueue<usize, Reverse<usize>> = PriorityQueue::new();
            neighborhood_multiplicities.iter().for_each(|(n, m)| {
                if candidates.contains(n) {
                    pq.push(*n, Reverse(*m));
                }
            });
            // let mut considered: AHashSet<usize> = AHashSet::new();
            let mut graveyard: AHashMap<usize, usize> = AHashMap::new();
            let mut stopping_criterion = 0usize;
            while let Some((n, Reverse(m))) = pq.pop() {
                let cand = &bg.nodes[n];
                if m <= stopping_criterion {
                    break;
                }
                if augmenter.query_and_admit(bg, cluster, cand) {
                    cand.edges
                        .iter()
                        .filter(|it| !cluster.contains(*it))
                        .for_each(|it| {
                            // if is already in the queue, update the priority
                            if candidates.contains(it) {
                                if let Some(Reverse(m)) = pq.get_priority(it) {
                                    pq.change_priority(it, Reverse(m + 1));
                                } else {
                                    let old_degree = graveyard.get(it).copied().unwrap_or(0usize);
                                    pq.push(*it, Reverse(old_degree + 1));
                                }
                            }
                        });
                } else {
                    graveyard.insert(n, m);
                    let can_earlystop = X::Augmenter::allows_earlystopping();
                    if can_earlystop {
                        break;
                    }
                }
            }
        })
}

pub fn augment_clusters<'a, X: AugmentingConfig + Clone + Sync, S: AbstractSubset<'a> + Sync>(
    bg: &'a Graph<Node>,
    clustering: &mut Clustering,
    candidates: &'a S,
    augmenting_config: &X,
) {
    let mut candidate_ids = candidates.each_node_id().copied().collect_vec();
    candidate_ids.sort_by_key(|&it| Reverse(bg.nodes[it].degree()));
    let n_clusters = clustering.clusters.len() as u64;
    clustering
        .clusters
        .par_iter_mut()
        .progress_count(n_clusters)
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
            let mut filter = NeighborhoodFilter::new(bg, &cluster.core());
            for cand in viable_candidates {
                if filter.is_relevant(&cand.id) && augmenter.query_and_admit(bg, cluster, cand) {
                    filter.add_neighbors_of(bg, cand.id);
                }
            }
        });
}

pub trait ExpandStrategy {
    fn expand<'a, X: AugmentingConfig + Clone + Sync, S: AbstractSubset<'a> + Sync>(
        bg: &'a Graph<Node>,
        clustering: &mut Clustering,
        candidate_ids: &'a S,
        augmenting_config: &X,
    ) -> ();
}

pub struct LegacyExpandStrategy {}
pub struct LocalExpandStrategy {}

impl ExpandStrategy for LegacyExpandStrategy {
    fn expand<'a, X: AugmentingConfig + Clone + Sync, S: AbstractSubset<'a> + Sync>(
        bg: &'a Graph<Node>,
        clustering: &mut Clustering,
        candidate_ids: &'a S,
        augmenting_config: &X,
    ) -> () {
        augment_clusters(bg, clustering, candidate_ids, augmenting_config);
    }
}

impl ExpandStrategy for LocalExpandStrategy {
    fn expand<'a, X: AugmentingConfig + Clone + Sync, S: AbstractSubset<'a> + Sync>(
        bg: &'a Graph<Node>,
        clustering: &mut Clustering,
        candidate_ids: &'a S,
        augmenting_config: &X,
    ) -> () {
        augment_clusters_local_expand(bg, clustering, candidate_ids, augmenting_config);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        parse_aoc_config, AugmentByCpm, AugmentByDensityThreshold, Augmenter, AugmentingConfig,
    };
    use crate::{aoc::AocConfig, Cluster, DefaultGraph, Graph};

    #[test]
    pub fn edge_density_augmenter_makes_sense() -> anyhow::Result<()> {
        let augment_config = AugmentByDensityThreshold {
            threshold: Some(0.5),
        };
        let g = Graph::parse_edgelist_from_str("0 1\n1 2\n2 3\n 3 4")?;
        let mut c = Cluster::from_iter(vec![g.retrieve("3").unwrap(), g.retrieve("4").unwrap()]);
        let mut augmenter = augment_config.augmenter(&g, &c);
        assert_eq!(2, c.size());
        assert!(augmenter.query_and_admit(&g, &mut c, g.node_from_label("2")));
        assert_eq!(3, c.size());
        assert!(augmenter.query_and_admit(&g, &mut c, g.node_from_label("1")));
        assert_eq!(4, c.size());
        assert!(!augmenter.query_and_admit(&g, &mut c, g.node_from_label("0")));
        assert_eq!(4, c.size());
        Ok(())
    }

    #[test]
    pub fn cpm_augmenter_makes_sense() -> anyhow::Result<()> {
        let r = 0.5;
        let augment_config = AugmentByCpm { resolution: r };
        let g: DefaultGraph = vec![(0, 1), (1, 2), (2, 0), (3, 4), (5, 6)]
            .into_iter()
            .collect();
        let mut c = Cluster::from_iter(vec![g.retrieve("0").unwrap(), g.retrieve("1").unwrap()]);
        let mut augmenter = augment_config.augmenter(&g, &c);
        assert_eq!(1, augmenter.ls);
        assert_eq!(g.cpm_of(&c.core(), r), augmenter.original_cpm);
        assert_eq!(g.cpm_of(&c.all(), r), augmenter.cpm());
        assert_eq!(0.5, augmenter.cpm());
        assert_eq!(2, c.size());
        assert!(augmenter.query_and_admit(&g, &mut c, g.node_from_label("2")));
        assert_eq!(
            c.periphery_nodes.iter().copied().collect::<Vec<usize>>(),
            vec![g.node_from_label("2").id]
        );
        assert_eq!(g.cpm_of(&c.all(), r), augmenter.cpm());
        assert_eq!(3, c.size());
        assert!(!augmenter.query_and_admit(&g, &mut c, g.node_from_label("3")));
        assert_eq!(3, c.size());
        assert_eq!(g.cpm_of(&c.all(), r), augmenter.cpm());
        Ok(())
    }

    #[test]
    pub fn can_parse_quality_specifier() -> anyhow::Result<()> {
        assert_eq!(AocConfig::Mod(0.5), parse_aoc_config("mod0.5").unwrap());
        assert_eq!(AocConfig::Cpm(0.5), parse_aoc_config("cpm0.5").unwrap());
        assert_eq!(AocConfig::K(3), parse_aoc_config("k3").unwrap());
        assert_eq!(AocConfig::Mcd(), parse_aoc_config("mcd").unwrap());
        assert_eq!(AocConfig::Mcd(), parse_aoc_config("m").unwrap());
        assert_eq!(
            AocConfig::EdgeDensity(0.5),
            parse_aoc_config("density0.5").unwrap()
        );
        assert_eq!(AocConfig::Denser(), parse_aoc_config("denser").unwrap());
        assert_eq!(
            AocConfig::MeanDegree(),
            parse_aoc_config("mean-degree").unwrap()
        );
        assert_eq!(
            AocConfig::Conductance(),
            parse_aoc_config("conductance").unwrap()
        );
        Ok(())
    }
}
