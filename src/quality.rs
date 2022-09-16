use serde::Deserialize;
use serde::Serialize;

use crate::aoc::AocConfig;
use crate::base::Graph;
use crate::utils::calc_cpm_resolution;
use crate::utils::calc_modularity;
use crate::AbstractSubset;

use crate::Clustering;
use crate::Node;
use crate::utils::calc_modularity_resolution;
pub fn modularity<'a, X>(g: &'a Graph<Node>, c: &'a X) -> f64
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

pub fn mcd<'a, X>(g: &'a Graph<Node>, c: &'a X) -> usize
where
    X: AbstractSubset<'a>,
{
    return g.degrees_inside(c).min().unwrap_or(0);
}

#[derive(Serialize, Deserialize)]
pub struct ClusterInformation {
    cid: Option<usize>,
    n: usize,
    m: usize,
    mcd: usize,
    modularity: f64,
    cpm : f64,
}

impl ClusterInformation {
    pub fn from_single_cluster<'a, X>(g: &'a Graph<Node>, c: &'a X, quality : &AocConfig) -> Self
    where
        X: AbstractSubset<'a>,
    {
        let resolution = match  quality
         {
            AocConfig::Mcd() => 1.0,
            AocConfig::K(_) => 1.0,
            AocConfig::Mod(r) => *r,
            AocConfig::Cpm(r) => *r,
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
        let modularity = calc_modularity_resolution(ls, ds, big_l, resolution);
        let cpm = calc_cpm_resolution(ls, n, resolution);
        Self {
            cid: None,
            n,
            m,
            mcd,
            modularity,
            cpm,
        }
    }

    // TODO: see if it is right to set mcd only among the core nodes
    pub fn vec_from_clustering(g: &Graph<Node>, clus: &Clustering, quality : &AocConfig) -> Vec<Self> {
        let mut records = Vec::new();
        for (&k, v) in clus.clusters.iter() {
            let mut record = Self::from_single_cluster(g, &v.core(), quality);
            record.cid = Some(k);
            records.push(record);
        }
        records
    }
}
