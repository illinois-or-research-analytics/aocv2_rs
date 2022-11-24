use rand::rngs::SmallRng;
use rand::seq::IteratorRandom;
use rand::{Rng, SeedableRng};

use crate::DefaultGraph;

pub trait GraphGenerator {
    fn gen(&mut self) -> DefaultGraph;
}

pub struct ErdosRenyiGenerator {
    pub rng: SmallRng,
    pub n: usize,
    pub p: f64,
}

impl ErdosRenyiGenerator {
    pub fn new(n: usize, p: f64) -> ErdosRenyiGenerator {
        ErdosRenyiGenerator {
            rng: SmallRng::from_entropy(),
            n,
            p,
        }
    }
}

impl GraphGenerator for ErdosRenyiGenerator {
    fn gen(&mut self) -> DefaultGraph {
        let mut edgelist = vec![];
        for i in 0..(self.n - 1) {
            for j in (i + 1)..self.n {
                if self.rng.gen::<f64>() < self.p {
                    edgelist.push((i, j));
                }
            }
        }
        edgelist.into_iter().collect()
    }
}

pub struct FixedNumEdgesGenerator {
    pub rng: SmallRng,
    pub n: usize,
    pub m: usize,
}

impl FixedNumEdgesGenerator {
    pub fn new(n: usize, m: usize) -> FixedNumEdgesGenerator {
        FixedNumEdgesGenerator {
            rng: SmallRng::from_entropy(),
            n,
            m,
        }
    }
}

impl GraphGenerator for FixedNumEdgesGenerator {
    fn gen(&mut self) -> DefaultGraph {
        let mut edgelist = vec![];
        for i in 0..(self.n - 1) {
            for j in (i + 1)..self.n {
                edgelist.push((i, j));
            }
        }
        let it = edgelist.into_iter().choose_multiple(&mut self.rng, self.m);
        it.into_iter().collect()
    }
}

pub struct LinkedListGraphGenerator {
    pub n: usize,
}

impl LinkedListGraphGenerator {
    pub fn new(n: usize) -> LinkedListGraphGenerator {
        LinkedListGraphGenerator { n }
    }
}

impl GraphGenerator for LinkedListGraphGenerator {
    fn gen(&mut self) -> DefaultGraph {
        let mut edgelist = vec![];
        for i in 0..(self.n - 1) {
            edgelist.push((i, i + 1));
        }
        edgelist.into_iter().collect()
    }
}
