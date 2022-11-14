use std::io::BufRead;

use ahash::AHashSet;

use crate::{AbstractSubset, Cluster, DefaultGraph, Graph, Node};

/// A wrapper to a list of node ids, designed for easy construction
/// from files (w.r.t. a graph).
pub struct NodeList {
    pub node_ids: Vec<usize>,
}

/// A mathematical subset of the graph, represented entirely
/// as a struct including a vector of node ids (for fast iteration)
/// and a set of node ids (for fast membership queries).
/// Use [`Self::new()`] to construct a subset from a vector of node ids.
pub struct OwnedSubset {
    pub node_ids: Vec<usize>,
    pub node_inclusion: AHashSet<usize>,
}

impl OwnedSubset {
    pub fn new(node_ids: Vec<usize>) -> Self {
        let node_inclusion = node_ids.iter().copied().collect();
        Self {
            node_ids,
            node_inclusion,
        }
    }

    pub fn insert(&mut self, node_id: usize) {
        if self.node_inclusion.insert(node_id) {
            self.node_ids.push(node_id);
        }
    }
}

/// A ``subset'' of nodes containing everything
pub struct UniverseSet {
    nodes: Vec<usize>, // FIXME: definitely do not need to store this
}

impl UniverseSet {
    pub fn new(max_nodes: usize) -> Self {
        Self {
            nodes: (0..max_nodes).collect(),
        }
    }

    pub fn new_from_graph(g: &DefaultGraph) -> Self {
        Self::new(g.n())
    }
}

impl<'a> AbstractSubset<'a> for UniverseSet {
    fn contains(&self, _node_id: &usize) -> bool {
        true
    }

    fn each_node_id(&'a self) -> Self::NodeIterator {
        self.nodes.iter()
    }

    type NodeIterator = std::slice::Iter<'a, usize>;
}

impl FromIterator<usize> for OwnedSubset {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        let node_ids = iter.into_iter().collect();
        Self::new(node_ids)
    }
}

impl<'a> AbstractSubset<'a> for OwnedSubset {
    fn contains(&self, node_id: &usize) -> bool {
        self.node_inclusion.contains(node_id)
    }

    fn each_node_id(&'a self) -> Self::NodeIterator {
        self.node_ids.iter()
    }

    type NodeIterator = std::slice::Iter<'a, usize>;
}

impl NodeList {
    pub fn from_raw_file<P>(g: &DefaultGraph, p: P) -> anyhow::Result<Self>
    where
        P: AsRef<std::path::Path>,
    {
        let f = std::fs::File::open(p)?;
        let reader = std::io::BufReader::new(f);
        let mut node_ids = Vec::new();
        for l in reader.lines() {
            let l = l?;
            let node_id = g.retrieve(l.trim().parse()?).unwrap();
            node_ids.push(node_id);
        }
        Ok(Self { node_ids })
    }

    pub fn into_cluster(self) -> Cluster {
        Cluster::from_iter(self.node_ids.into_iter())
    }

    pub fn into_owned_subset(self) -> OwnedSubset {
        OwnedSubset::new(self.node_ids)
    }
}

/// A helper structure to calculate conductance of growing clusters online
#[derive(Clone, Debug)]
pub struct OnlineConductance {
    cut: usize,
    vol: usize,
    total_degree: usize,
}

impl OnlineConductance {
    pub fn new<'a, X>(graph: &'a DefaultGraph, view: &'a X) -> Self
    where
        X: AbstractSubset<'a>,
    {
        let cut = graph.cut_of(view);
        let vol = graph.volume_inside(view);
        Self {
            cut,
            vol,
            total_degree: graph.total_degree(),
        }
    }

    pub fn conductance(&self) -> f64 {
        self.cut as f64 / self.vol.min(self.total_degree - self.vol) as f64
    }

    pub fn update_conductance_if<'a, X>(
        &mut self,
        _graph: &'a DefaultGraph,
        node: &'a Node,
        view: &'a X,
        f: impl Fn(f64) -> bool,
    ) -> (f64, bool)
    where
        X: AbstractSubset<'a>,
    {
        let d = node.degree_inside(view);
        let cut_prime = self.cut - d + node.degree_outside(view);
        let vol_prime = self.vol + node.degree();
        let conductance_prime =
            cut_prime as f64 / vol_prime.min(self.total_degree - vol_prime) as f64;
        let satisfied = f(conductance_prime) && d > 0;
        if satisfied {
            self.cut = cut_prime;
            self.vol = vol_prime;
        }
        (conductance_prime, satisfied)
    }

    pub fn query_conductance<'a, X>(
        &self,
        _graph: &'a DefaultGraph,
        node: &'a Node,
        view: &'a X,
    ) -> f64
    where
        X: AbstractSubset<'a>,
    {
        let cut_prime = self.cut - node.degree_inside(view) + node.degree_outside(view);
        let vol_prime = self.vol + node.degree();

        cut_prime as f64 / vol_prime.min(self.total_degree - vol_prime) as f64
    }

    pub fn add_node<'a, X>(&mut self, graph: &'a DefaultGraph, node: &'a Node, view: &'a X) -> f64
    where
        X: AbstractSubset<'a>,
    {
        self.update_conductance_if(graph, node, view, |_| true).0
    }
}
