use ahash::AHashMap;

use lz4::EncoderBuilder;
use probabilistic_collections::bloom::BloomFilter;
use serde::{Deserialize, Serialize};

use crate::{AbstractSubset, DefaultGraph};

/// Internal API used for allocating new internal ids for nodes
/// when seeing external ids.
#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct NameSet {
    pub next_id: usize,
    pub forward: AHashMap<usize, usize>,
    pub rev: Vec<usize>,
}

impl NameSet {
    /// Given an external id, retrieve the internal id
    pub fn retrieve(&self, name: usize) -> Option<usize> {
        self.forward.get(&name).copied()
    }

    /// Given an internal id, retrieve the external id
    pub fn rev(&self, id: usize) -> Option<usize> {
        self.rev.get(id).copied()
    }

    /// Given an internal id and external id pair, insert them into the mapping
    pub fn bi_insert(&mut self, name: usize, id: usize) {
        self.forward.insert(name, id);
        // make sure self.rev is long enough
        if self.rev.len() <= id {
            self.rev.resize(id + 1, 0);
        }
        self.rev[id] = name;
    }
}

/// Binomial coefficient (n choose 2)
pub fn choose2(n: usize) -> usize {
    if n == 0 {
        0
    } else {
        n * (n - 1) / 2
    }
}

pub fn calc_modularity(ls: usize, ds: usize, big_l: usize) -> f64 {
    let (ls, ds, big_l) = (ls as f64, ds as f64, big_l as f64);
    (ls / big_l) - (ds / (2.0 * big_l)).powi(2)
}

pub fn calc_modularity_resolution(ls: usize, ds: usize, big_l: usize, resolution: f64) -> f64 {
    let (ls, ds, big_l) = (ls as f64, ds as f64, big_l as f64);
    (ls / big_l) - resolution * (ds / (2.0 * big_l)).powi(2)
}

pub fn calc_cpm_resolution(ls: usize, n: usize, resolution: f64) -> f64 {
    ls as f64 - resolution * choose2(n) as f64
}

pub fn write_compressed_bincode<S, P>(path: P, data: &S) -> anyhow::Result<()>
where
    S: Serialize,
    P: AsRef<std::path::Path>,
{
    let mut file = std::fs::File::create(path)?;
    let mut encoder = EncoderBuilder::new().level(4).build(&mut file)?;
    bincode::serialize_into(&mut encoder, data)?;
    let (_, res) = encoder.finish();
    res?;
    Ok(())
}

pub fn read_compressed_bincode<S, P>(path: P) -> anyhow::Result<S>
where
    S: for<'de> Deserialize<'de>,
    P: AsRef<std::path::Path>,
{
    let mut file = std::fs::File::open(path)?;
    let mut decoder = lz4::Decoder::new(&mut file)?;
    let data = bincode::deserialize_from(&mut decoder)?;
    Ok(data)
}

/// Internal API used for the legacy strategy of expansion
/// to quickly rule out nodes that are disconnected to a cluster.
/// Internally uses a bloom filter.
#[derive(Debug)]
pub struct NeighborhoodFilter {
    filter: BloomFilter<usize>,
}

impl NeighborhoodFilter {
    pub fn new<'a, X>(bg: &DefaultGraph, view: &'a X) -> Self
    where
        X: AbstractSubset<'a>,
    {
        let n = view.num_nodes();
        let size = bg.n().min(n * 10);
        let mut filter = BloomFilter::<usize>::new(size, 0.01);
        for node_id in view.each_node_id() {
            let node = &bg.nodes[*node_id];
            for neighbor_id in &node.edges {
                filter.insert(neighbor_id);
            }
        }
        Self { filter }
    }

    pub fn add_neighbors_of(&mut self, bg: &DefaultGraph, node_id: usize) {
        let node = &bg.nodes[node_id];
        for neighbor_id in &node.edges {
            self.filter.insert(neighbor_id);
        }
    }

    pub fn is_relevant(&self, node_id: &usize) -> bool {
        self.filter.contains(node_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_choose2() {
        assert_eq!(choose2(0), 0);
        assert_eq!(choose2(1), 0);
        assert_eq!(choose2(2), 1);
        assert_eq!(choose2(3), 3);
        assert_eq!(choose2(4), 6);
        assert_eq!(choose2(5), 10);
    }
}
