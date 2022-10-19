use bimap::BiMap;
use lz4::EncoderBuilder;
use probabilistic_collections::bloom::BloomFilter;
use serde::{Deserialize, Serialize};

use crate::{AbstractSubset, Graph, Node};

#[derive(Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct NameSet {
    pub next_id: usize,
    pub bimap: BiMap<String, usize>,
}

impl NameSet {
    pub fn retrieve(&self, name: &str) -> Option<usize> {
        self.bimap.get_by_left(name).copied()
    }

    pub fn rev(&self, id: usize) -> Option<&str> {
        self.bimap.get_by_right(&id).map(|it| it.as_str())
    }
}

pub fn choose2(n: usize) -> usize {
    n * (n - 1) / 2
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

#[derive(Debug)]
pub struct NeighborhoodFilter {
    filter: BloomFilter<usize>,
}

impl NeighborhoodFilter {
    pub fn new<'a, X>(bg: &Graph<Node>, view: &'a X) -> Self
    where
        X: AbstractSubset<'a>,
    {
        let n = view.num_nodes();
        let size = bg.n().min(n * 10);
        let mut filter = BloomFilter::<usize>::new(size, 0.01);
        for node_id in view.each_node_id() {
            let node = &bg.nodes[*node_id];
            for neighbor_id in &node.out_edges {
                filter.insert(neighbor_id);
            }
        }
        Self { filter }
    }

    pub fn add_neighbors_of(&mut self, bg: &Graph<Node>, node_id: usize) {
        let node = &bg.nodes[node_id];
        for neighbor_id in &node.out_edges {
            self.filter.insert(neighbor_id);
        }
    }

    pub fn is_relevant(&self, node_id: &usize) -> bool {
        self.filter.contains(node_id)
    }
}
