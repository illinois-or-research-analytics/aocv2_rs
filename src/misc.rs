use std::io::BufRead;

use ahash::AHashSet;

use crate::{AbstractSubset, Cluster, Graph, Node};

pub struct NodeList {
    pub node_ids: Vec<usize>,
}

pub struct NodeListAsSubset {
    pub node_ids: Vec<usize>,
    pub node_inclusion: AHashSet<usize>,
}

impl NodeListAsSubset {
    pub fn new(node_ids: Vec<usize>) -> Self {
        let node_inclusion = node_ids.iter().copied().collect();
        Self {
            node_ids,
            node_inclusion,
        }
    }
}

impl<'a> AbstractSubset<'a> for NodeListAsSubset {
    fn contains(&self, node_id: &usize) -> bool {
        self.node_inclusion.contains(node_id)
    }

    fn each_node_id(&'a self) -> Self::NodeIterator {
        self.node_ids.iter()
    }

    type NodeIterator = std::slice::Iter<'a, usize>;
}

impl NodeList {
    pub fn from_raw_file<P>(g: &Graph<Node>, p: P) -> anyhow::Result<Self>
    where
        P: AsRef<std::path::Path>,
    {
        let f = std::fs::File::open(p)?;
        let reader = std::io::BufReader::new(f);
        let mut node_ids = Vec::new();
        for l in reader.lines() {
            let l = l?;
            let node_id = g.retrieve(l.trim()).unwrap();
            node_ids.push(node_id);
        }
        Ok(Self { node_ids })
    }

    pub fn into_cluster(self) -> Cluster {
        Cluster::from_iter(self.node_ids.into_iter())
    }

    pub fn into_owned_subset(self) -> NodeListAsSubset {
        NodeListAsSubset::new(self.node_ids)
    }
}
