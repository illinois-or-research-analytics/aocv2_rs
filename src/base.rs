use crate::utils::{self, NameSet};
use ahash::AHashSet;
use anyhow::{bail, Ok};
use itertools::Itertools;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{hash_set, BTreeMap, BTreeSet},
    ffi::OsStr,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Read, Write},
    iter::FromIterator,
    path::Path,
};
use tracing::debug;

pub trait AbstractNode {
    //! A trait for nodes in a graph.
    fn assign_id(&mut self, id: usize);
    fn add_out_edge(&mut self, target: usize);
    fn add_in_edge(&mut self, from: usize);
}

/// A trait for a specific subset of a graph.
pub trait AbstractSubset<'a> {
    fn contains(&self, node_id: &usize) -> bool;
    type NodeIterator: Iterator<Item = &'a usize>;
    fn each_node_id(&'a self) -> Self::NodeIterator;
}

#[derive(Default, Debug)]
pub struct TransientNode {
    id: usize,
    in_edges: BTreeSet<usize>,
    out_edges: BTreeSet<usize>,
    edges: BTreeSet<usize>,
}

impl AbstractNode for TransientNode {
    fn add_out_edge(&mut self, target: usize) {
        self.edges.insert(target);
        self.out_edges.insert(target);
    }

    fn add_in_edge(&mut self, from: usize) {
        self.edges.insert(from);
        self.in_edges.insert(from);
    }

    fn assign_id(&mut self, id: usize) {
        self.id = id;
    }
}

/// The default node type, contains information  both the directed and the undirected topology
#[derive(Default, Debug, Serialize, Deserialize)]
pub struct Node {
    pub id: usize,
    pub in_edges: Vec<usize>,
    pub out_edges: Vec<usize>,
    pub edges: Vec<usize>,
}

impl Node {
    pub fn edges_inside<'a, X>(&'a self, c: &'a X) -> impl Iterator<Item = &'a usize> + 'a
    where
        X: AbstractSubset<'a>,
    {
        self.edges.iter().filter(move |&e| c.contains(e))
    }

    pub fn degree(&self) -> usize {
        self.edges.len()
    }

    pub fn indegree(&self) -> usize {
        self.in_edges.len()
    }

    pub fn outdegree(&self) -> usize {
        self.out_edges.len()
    }

    pub fn total_degree(&self) -> usize {
        self.indegree() + self.outdegree()
    }
}

impl AbstractNode for Node {
    fn add_out_edge(&mut self, target: usize) {
        self.edges.push(target);
        self.out_edges.push(target);
    }

    fn add_in_edge(&mut self, from: usize) {
        self.edges.push(from);
        self.in_edges.push(from);
    }

    fn assign_id(&mut self, id: usize) {
        self.id = id;
    }
}

impl TransientNode {
    pub fn into_permanent(self) -> Node {
        Node {
            id: self.id,
            in_edges: self.in_edges.into_iter().collect_vec(),
            out_edges: self.out_edges.into_iter().collect_vec(),
            edges: self.edges.into_iter().collect_vec(),
        }
    }
}

#[derive(Default, Debug, Serialize, Deserialize)]
pub struct Graph<NodeT>
where
    NodeT: Default + AbstractNode,
{
    name_set: NameSet,
    pub nodes: Vec<NodeT>,
    m_cache: usize,
}

impl<'a, NodeT> Graph<NodeT>
where
    NodeT: Default + AbstractNode,
{
    pub fn request(&mut self, s: &str) -> usize {
        return self
            .name_set
            .bimap
            .get_by_left(s)
            .copied()
            .unwrap_or_else(|| {
                let id = self.name_set.next_id;
                self.name_set.next_id += 1;
                self.name_set.bimap.insert(s.to_string(), id);
                let mut node = NodeT::default();
                node.assign_id(id);
                self.nodes.push(node);
                id
            });
    }

    pub fn retrieve(&self, s: &str) -> Option<usize> {
        self.name_set.retrieve(s)
    }

    pub fn n(&self) -> usize {
        self.nodes.len()
    }
    pub fn m(&self) -> usize {
        self.m_cache
    }
}

impl Graph<Node> {
    pub fn parse_edgelist_from_reader<R: BufRead + Read>(reader: R) -> anyhow::Result<Graph<Node>> {
        let mut graph = Graph::<TransientNode>::default();
        // let mut progress = 0;
        for line in reader.lines() {
            let line = line?;
            let mut parts = line.split_whitespace();
            let from = parts
                .next()
                .ok_or_else(|| anyhow::anyhow!("missing from"))?;
            let to = parts.next().ok_or_else(|| anyhow::anyhow!("missing to"))?;
            let from_id = graph.request(from);
            let to_id = graph.request(to);
            graph.nodes[from_id].add_out_edge(to_id);
            graph.nodes[to_id].add_in_edge(from_id);
            // progress += 1;
            // if progress % 100000 == 0 {
            //     debug!("progress: {}", progress);
            // }
        }
        let (permanent_nodes, name_set) = (
            graph
                .nodes
                .into_iter()
                .map(|node| node.into_permanent())
                .collect_vec(),
            graph.name_set,
        );
        let num_edges = permanent_nodes.iter().map(|n| n.edges.len()).sum::<usize>() / 2;
        Ok(Graph {
            name_set,
            nodes: permanent_nodes,
            m_cache: num_edges,
        })
    }

    pub fn parse_from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Graph<Node>> {
        if path.as_ref().extension() == Some(OsStr::new("lz4")) {
            debug!("loading from bincode.lz4");
            Self::parse_compressed_bincode(path)
        } else {
            debug!("loading from edgelist");
            Self::parse_edgelist(path)
        }
    }

    pub fn parse_compressed_bincode<P: AsRef<Path>>(path: P) -> anyhow::Result<Graph<Node>> {
        utils::read_compressed_bincode(path)
    }

    pub fn parse_edgelist<P>(path: P) -> anyhow::Result<Graph<Node>>
    where
        P: AsRef<Path>,
    {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Self::parse_edgelist_from_reader(reader)
    }

    pub fn each_edge_inside<'a, X>(
        &'a self,
        view: &'a X,
    ) -> impl Iterator<Item = (usize, usize)> + 'a
    where
        X: AbstractSubset<'a>,
    {
        view.each_node_id()
            .map(|it| &self.nodes[*it])
            .flat_map(|it| it.edges_inside(view).map(|e| (it.id, *e)))
    }

    pub fn degrees_inside<'a, X>(&'a self, view: &'a X) -> impl Iterator<Item = usize> + 'a
    where
        X: AbstractSubset<'a>,
    {
        view.each_node_id()
            .map(|it| &self.nodes[*it])
            .map(|it| it.edges_inside(view).count())
    }

    pub fn num_edges_inside<'a, X>(&'a self, view: &'a X) -> usize
    where
        X: AbstractSubset<'a>,
    {
        self.degrees_inside(view).sum::<usize>() / 2
    }
}

#[derive(Default)]
pub struct Cluster {
    pub core_nodes: AHashSet<usize>,
    pub periphery_nodes: AHashSet<usize>,
}

impl Cluster {
    pub fn add_core(&mut self, node: usize) {
        self.core_nodes.insert(node);
    }

    pub fn add_periphery(&mut self, node: usize) {
        self.periphery_nodes.insert(node);
    }

    pub fn mcd(&self, bg: &Graph<Node>) -> Option<usize> {
        bg.degrees_inside(&self.core()).min()
    }

    pub fn size(&self) -> usize {
        self.core_nodes.len() + self.periphery_nodes.len()
    }

    pub fn is_singleton(&self) -> bool {
        self.size() == 1
    }

    pub fn is_non_trivial(&self) -> bool {
        self.size() > 1
    }
}

impl FromIterator<usize> for Cluster {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        let mut cluster = Cluster::default();
        for node in iter {
            cluster.add_core(node);
        }
        cluster
    }
}

pub enum ClusterViewType {
    Core,
    Periphery,
}

pub struct ClusterView<'a> {
    pub cluster: &'a Cluster,
    pub view_type: ClusterViewType,
}

pub struct ClusterEntireView<'a> {
    pub cluster: &'a Cluster,
}

impl<'a> AbstractSubset<'a> for ClusterEntireView<'a> {
    fn contains(&self, node_id: &usize) -> bool {
        self.cluster.core_nodes.contains(node_id) || self.cluster.periphery_nodes.contains(node_id)
    }

    fn each_node_id(&'a self) -> Self::NodeIterator {
        self.cluster
            .core_nodes
            .iter()
            .chain(self.cluster.periphery_nodes.iter())
    }

    type NodeIterator = std::iter::Chain<
        std::collections::hash_set::Iter<'a, usize>,
        std::collections::hash_set::Iter<'a, usize>,
    >;
}

impl<'a> AbstractSubset<'a> for ClusterView<'a> {
    fn contains(&self, node: &usize) -> bool {
        match self.view_type {
            ClusterViewType::Core => self.cluster.core_nodes.contains(node),
            ClusterViewType::Periphery => self.cluster.periphery_nodes.contains(node),
        }
    }

    fn each_node_id(&'a self) -> Self::NodeIterator {
        match self.view_type {
            ClusterViewType::Core => self.cluster.core_nodes.iter(),
            ClusterViewType::Periphery => self.cluster.periphery_nodes.iter(),
        }
    }

    type NodeIterator = hash_set::Iter<'a, usize>;
}

impl<'a> Cluster {
    pub fn core(&'a self) -> ClusterView<'a> {
        ClusterView {
            cluster: self,
            view_type: ClusterViewType::Core,
        }
    }

    pub fn periphery(&'a self) -> ClusterView<'a> {
        ClusterView {
            cluster: self,
            view_type: ClusterViewType::Periphery,
        }
    }

    pub fn all(&'a self) -> ClusterEntireView<'a> {
        ClusterEntireView { cluster: self }
    }
}

pub struct Clustering {
    pub clusters: BTreeMap<usize, Cluster>,
}

impl Clustering {
    pub fn parse_from_reader<R: Read + BufRead>(
        bg: &Graph<Node>,
        reader: R,
        reverse: bool,
    ) -> anyhow::Result<Self> {
        let mut clusters: BTreeMap<usize, Cluster> = BTreeMap::default();
        let mut not_found_nodes: BTreeSet<usize> = BTreeSet::default();
        for line in reader.lines() {
            let line = line?;
            let mut parts = line.split_whitespace();
            let cluster_name = parts
                .next()
                .ok_or_else(|| anyhow::anyhow!("missing cluster_id"))?;
            let node_name = parts
                .next()
                .ok_or_else(|| anyhow::anyhow!("missing node_name"))?;
            let (cluster_name, node_name) = if reverse {
                (node_name, cluster_name)
            } else {
                (cluster_name, node_name)
            };
            let cluster_id = cluster_name
                .parse::<usize>()
                .map_err(|_| anyhow::anyhow!("invalid cluster_id"))?;
            let node_id = bg.retrieve(node_name);
            match node_id {
                Some(node_id) => {
                    clusters.entry(cluster_id).or_default().add_core(node_id);
                }
                None => {
                    if !not_found_nodes.contains(&cluster_id) {
                        not_found_nodes.insert(cluster_id);
                    } else {
                        bail!("node {} stipulated in cluster {} not found in graph, and is not singleton", node_name, cluster_id);
                    }
                }
            }
        }
        let singleton_keys = clusters
                .iter()
                .filter(|(_, cluster)| cluster.is_singleton())
                .map(|(k, _)| k)
                .copied()
                .collect_vec();
        for key in singleton_keys.iter() {
            clusters.remove(key);
        }
        Ok(Clustering { clusters })
    }

    pub fn parse_from_file<P>(bg: &Graph<Node>, path: P, reverse: bool) -> anyhow::Result<Self>
    where
        P: AsRef<Path>,
    {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Self::parse_from_reader(bg, reader, reverse)
    }

    pub fn write_raw<W>(&self, mut writer: W, graph: &Graph<Node>) -> anyhow::Result<()>
    where
        W: Write,
    {
        for (cluster_id, cluster) in self.clusters.iter() {
            for node_id in cluster
                .core_nodes
                .iter()
                .chain(cluster.periphery_nodes.iter())
            {
                writeln!(
                    writer,
                    "{} {}",
                    cluster_id,
                    graph.name_set.rev(*node_id).unwrap()
                )?;
            }
        }
        Ok(())
    }

    pub fn write_file<P>(&self, graph: &Graph<Node>, path: P) -> anyhow::Result<()>
    where
        P: AsRef<Path>,
    {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        self.write_raw(writer, graph)
    }
}

#[cfg(test)]
mod tests {
    use std::io::BufReader;

    use ahash::AHashSet;

    use crate::{AbstractSubset, Cluster, Graph};

    #[test]
    pub fn clustering_can_create_view() {
        let mut c: Cluster = Cluster {
            core_nodes: AHashSet::default(),
            periphery_nodes: AHashSet::default(),
        };
        c.core_nodes.extend([1, 2]);
        c.periphery_nodes.extend([3]);
        assert!(c.core().contains(&1));
        assert!(c.core().contains(&2));
        assert!(!c.periphery().contains(&1));
        assert!(c.periphery().contains(&3));
        for i in 1..4 {
            assert!(c.all().contains(&i));
        }
        for i in 10..20 {
            assert!(!c.all().contains(&i));
        }
    }

    #[test]
    pub fn can_parse_simple_graphs() {
        let edgelist = "0 1\n1 2\n2 3";
        let reader = BufReader::new(edgelist.as_bytes());
        let graph = Graph::parse_edgelist_from_reader(reader).unwrap();
        assert_eq!(4, graph.n());
        assert_eq!(3, graph.m());
    }
}
