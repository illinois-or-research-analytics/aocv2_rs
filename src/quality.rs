use serde::Deserialize;
use serde::Serialize;

use crate::base::Graph;
use crate::utils::calc_modularity;
use crate::AbstractSubset;

use crate::Clustering;
use crate::Node;
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
}

impl ClusterInformation {
    pub fn from_single_cluster<'a, X>(g: &'a Graph<Node>, c: &'a X) -> Self
    where
        X: AbstractSubset<'a>,
    {
        let n = c.each_node_id().count();
        let m = g.num_edges_inside(c);
        let mcd = mcd(g, c);
        let modularity = modularity(g, c);
        Self {
            cid: None,
            n,
            m,
            mcd,
            modularity,
        }
    }

    // TODO: see if it is right to set mcd only among the core nodes
    pub fn vec_from_clustering(g: &Graph<Node>, clus: &Clustering) -> Vec<Self> {
        let mut records = Vec::new();
        for (&k, v) in clus.clusters.iter() {
            let n = v.size();
            let m = g.num_edges_inside(&v.core());
            let mcd = mcd(g, &v.core());
            let modularity = modularity(g, &v.all());
            records.push(Self {
                cid: Some(k),
                n,
                m,
                mcd,
                modularity,
            });
        }
        records
    }
}
