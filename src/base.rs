use crate::{
    misc::OwnedSubset,
    utils::{self, choose2, NameSet},
};
use ahash::AHashSet;
use anyhow::{bail, Ok};
use itertools::Itertools;
use ordered_float::NotNan;
use rayon::prelude::*;
use roaring::RoaringBitmap;
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

/// A trait for nodes in a graph.
/// The (weak) reason this exists is to handle polymorphism in the underlying edge storage and edge topology.
/// See [TransientNode](crate::base::TransientNode) and [Node](crate::base::Node) for concrete implementations.
///
/// As always, the ids stored inside the nodes are the *internal* ids mapped from the original ids.
///
/// # Tradeoffs
///
/// Due to how Rust handles memory safety, the nodes canonically only store the node ids of its outgoing edges.
pub trait AbstractNode {
    fn assign_id(&mut self, id: usize);
    fn add_out_edge(&mut self, target: usize);
    fn add_in_edge(&mut self, from: usize);
}

/// An abstract representation for a specific subset of a graph, different
/// from a [Cluster](crate::base::Cluster).
/// This subset needs to support two operations.
/// First a fast query of a node existence (see [`Self::contains()`]), second a iterator over all node ids (see [`Self::each_node_id()`]).
/// See also [OwnedSubset](crate::misc::OwnedSubset), a direct owned realization of this trait.
pub trait AbstractSubset<'a> {
    fn contains(&self, node_id: &usize) -> bool;
    type NodeIterator: Iterator<Item = &'a usize>;
    fn each_node_id(&'a self) -> Self::NodeIterator;

    fn num_nodes(&'a self) -> usize {
        self.each_node_id().count()
    }
}

/// A node specialized to eliminate parallel edges during construction, using sets as the underlying storage of edges.
/// It is not designed to be the permanent representation of nodes and should be converted after building the graph.
/// The motivation is for robustness and reducing human errors. See [Node](crate::base::Node) for the permanent representation.
#[derive(Default, Debug, Clone)]
pub struct TransientNode {
    id: usize,
    // in_edges: BTreeSet<usize>,
    // out_edges: BTreeSet<usize>,
    edges: BTreeSet<usize>,
}

impl AbstractNode for TransientNode {
    fn add_out_edge(&mut self, target: usize) {
        self.edges.insert(target);
        // self.out_edges.insert(target);
    }

    fn add_in_edge(&mut self, from: usize) {
        self.edges.insert(from);
        // self.in_edges.insert(from);
    }

    fn assign_id(&mut self, id: usize) {
        self.id = id;
    }
}

/// The default node type, contains information only on the undirected topology
/// with storage of edges as a vector for maximum efficiency of iteration.
#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Node {
    pub id: usize,
    // pub in_edges: Vec<usize>,
    // pub out_edges: Vec<usize>,
    pub edges: Vec<usize>,
}

impl Node {
    /// Iterates the incident node ids inside the subset.
    pub fn edges_inside<'a, X>(&'a self, c: &'a X) -> impl Iterator<Item = &'a usize> + 'a
    where
        X: AbstractSubset<'a>,
    {
        self.edges.iter().filter(move |&e| c.contains(e))
    }

    /// Iterates the incident node ids outside the subset.
    pub fn edges_outside<'a, X>(&'a self, c: &'a X) -> impl Iterator<Item = &'a usize> + 'a
    where
        X: AbstractSubset<'a>,
    {
        self.edges.iter().filter(move |&e| !c.contains(e))
    }

    /// Counts the number of incident nodes inside the subset.
    pub fn degree_inside<'a, X>(&'a self, c: &'a X) -> usize
    where
        X: AbstractSubset<'a>,
    {
        self.edges_inside(c).count()
    }

    /// Counts the number of incident nodes outside the subset.
    pub fn degree_outside<'a, X>(&'a self, c: &'a X) -> usize
    where
        X: AbstractSubset<'a>,
    {
        self.edges_outside(c).count()
    }

    /// Checks if the node is relevant (i.e., has any connection) to the subset.
    pub fn is_relevant_to<'a, X>(&'a self, c: &'a X) -> bool
    where
        X: AbstractSubset<'a>,
    {
        self.degree_inside(c) > 0
    }

    /// The global degree of the node
    pub fn degree(&self) -> usize {
        self.edges.len()
    }

    /// Retrieves the indegree of the node. Due to being undirected, this is the same as the degree.
    pub fn indegree(&self) -> usize {
        self.degree()
    }

    /// Retrieves the outdegree of the node. Due to being undirected, this is the same as the degree.
    pub fn outdegree(&self) -> usize {
        self.degree()
    }

    /// The "total degree" of the node. This is currently double the degree.
    pub fn total_degree(&self) -> usize {
        self.indegree() + self.outdegree()
    }
}

impl AbstractNode for Node {
    fn add_out_edge(&mut self, target: usize) {
        self.edges.push(target);
        // self.out_edges.push(target);
    }

    fn add_in_edge(&mut self, from: usize) {
        self.edges.push(from);
        // self.in_edges.push(from);
    }

    fn assign_id(&mut self, id: usize) {
        self.id = id;
    }
}

impl TransientNode {
    /// Converts a transient node into a permanent node by flattening the edge-set into a vector.
    pub fn into_permanent(self) -> Node {
        Node {
            id: self.id,
            // in_edges: self.in_edges.into_iter().collect_vec(),
            // out_edges: self.out_edges.into_iter().collect_vec(),
            edges: self.edges.into_iter().collect_vec(),
        }
    }
}

/// A graph polymorphic over the underlying node type.
/// AOCluster uses a traditional architecture of storing its nodes inside vectors.
/// However, in order to support non-continuous (gapped) node ids from the input graph,
/// the *internal* ids differ from the external ids. Therefore AOCluster always
/// maps back the id when writing the output. See [NameSet](utils::NameSet) and the
/// correposponding field to understand the mapping.
#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Graph<NodeT>
where
    NodeT: Default + AbstractNode,
{
    pub name_set: NameSet,
    pub nodes: Vec<NodeT>,
    m_cache: usize,
}

/// The sane default undirected graph type
pub type DefaultGraph = Graph<Node>;

impl<'a, NodeT> Graph<NodeT>
where
    NodeT: Default + AbstractNode,
{
    /// Given an external node id, returns the internal node id.
    /// If the internal id is not allocated yet, a new one will be allocated.
    pub fn request(&mut self, s: usize) -> usize {
        return self.name_set.forward.get(&s).copied().unwrap_or_else(|| {
            let id = self.name_set.next_id;
            self.name_set.next_id += 1;
            self.name_set.forward.insert(s, id);
            self.name_set.rev.push(s);
            let mut node = NodeT::default();
            node.assign_id(id);
            self.nodes.push(node);
            id
        });
    }

    /// Retrieves the internal id of the node with the given external id.
    pub fn retrieve(&self, s: usize) -> Option<usize> {
        self.name_set.retrieve(s)
    }

    /// Returns the number of nodes in the graph.
    pub fn n(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of edges in the graph.
    pub fn m(&self) -> usize {
        self.m_cache
    }

    /// Returns the total degree of the graph.
    pub fn total_degree(&self) -> usize {
        self.m() * 2
    }

    /// Creates a subset of the graph using *internal* node ids.
    pub fn owned_subset(&self, nodes: Vec<usize>) -> OwnedSubset {
        OwnedSubset::new(nodes)
    }

    /// Retrieves the node using the external id.
    pub fn node_from_label(&'a self, label: usize) -> &'a NodeT {
        let nid = self.retrieve(label).unwrap();
        &self.nodes[nid]
    }
}

impl FromIterator<(usize, usize)> for Graph<Node> {
    fn from_iter<T: IntoIterator<Item = (usize, usize)>>(iter: T) -> Self {
        Graph::<Node>::from_integer_edges(iter.into_iter())
    }
}

impl Graph<Node> {
    /// Constructs a `DefaultGraph` from an iterator of integer edges.
    pub fn from_integer_edges(edges: impl Iterator<Item = (usize, usize)>) -> Self {
        let mut graph = Graph::<TransientNode>::default();
        for (from, to) in edges {
            let required_length = from.max(to) + 1;
            if graph.nodes.len() < required_length {
                graph.nodes.resize(required_length, Default::default());
            }
            graph.nodes[from].add_out_edge(to);
            graph.nodes[to].add_in_edge(from);
        }
        for (id, node) in graph.nodes.iter_mut().enumerate() {
            node.assign_id(id);
            graph.name_set.bi_insert(id, id);
        }
        graph.name_set.next_id = graph.nodes.len();
        let permanent_nodes = graph
            .nodes
            .into_iter()
            .map(|node| node.into_permanent())
            .collect_vec();
        let num_edges = permanent_nodes.iter().map(|n| n.edges.len()).sum::<usize>() / 2;
        Graph {
            name_set: graph.name_set,
            nodes: permanent_nodes,
            m_cache: num_edges,
        }
    }

    pub fn parse_edgelist_from_reader<R: BufRead + Read>(reader: R) -> anyhow::Result<Graph<Node>> {
        let mut graph = Graph::<TransientNode>::default();
        for line in reader.lines() {
            let line = line?;
            let mut parts = line.split_whitespace();
            let from = parts
                .next()
                .ok_or_else(|| anyhow::anyhow!("missing from"))?;
            let to = parts.next().ok_or_else(|| anyhow::anyhow!("missing to"))?;
            let from_id = graph.request(from.parse()?);
            let to_id = graph.request(to.parse()?);
            graph.nodes[from_id].add_out_edge(to_id);
            graph.nodes[to_id].add_in_edge(from_id);
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

    pub fn parse_edgelist_from_str(s: &str) -> anyhow::Result<Graph<Node>> {
        let reader = BufReader::new(s.as_bytes());
        Graph::parse_edgelist_from_reader(reader)
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
            .flat_map(|it| {
                it.edges_inside(view).map(|e| {
                    let (u, v) = (it.id, *e);
                    if u < v {
                        (u, v)
                    } else {
                        (v, u)
                    }
                })
            })
    }

    pub fn each_edge<'a>(&'a self) -> impl Iterator<Item = (usize, usize)> + 'a {
        self.nodes.iter().flat_map(|it| {
            it.edges.iter().filter_map(|e| {
                let (u, v) = (it.id, *e);
                if u < v {
                    Some((u, v))
                } else {
                    None
                }
            })
        })
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

    pub fn count_n_m<'a, X>(&'a self, view: &'a X) -> (usize, usize)
    where
        X: AbstractSubset<'a>,
    {
        let m = self.num_edges_inside(view);
        let n = view.num_nodes();
        (n, m)
    }

    pub fn edge_density_inside<'a, X>(&'a self, view: &'a X) -> f64
    where
        X: AbstractSubset<'a>,
    {
        let (n, m) = self.count_n_m(view);
        m as f64 / choose2(n) as f64
    }

    /// The degree volume of the subset `X` defined as the sum of the global degrees.
    pub fn volume_inside<'a, X>(&'a self, view: &'a X) -> usize
    where
        X: AbstractSubset<'a>,
    {
        view.each_node_id().map(|it| self.nodes[*it].degree()).sum()
    }

    /// The total number of out-going edges from a cluster
    pub fn cut_of<'a, X>(&'a self, view: &'a X) -> usize
    where
        X: AbstractSubset<'a>,
    {
        view.each_node_id()
            .map(|it| self.nodes[*it].degree_outside(view))
            .sum::<usize>()
    }

    /// The conductance of the cut induced by `X`
    pub fn conductance_of<'a, X>(&'a self, view: &'a X) -> f64
    where
        X: AbstractSubset<'a>,
    {
        let cut = self.cut_of(view);
        let total_degree = self.total_degree();
        let vol1 = self.volume_inside(view);
        let volume = vol1.min(total_degree - vol1);
        cut as f64 / volume as f64
    }

    /// The Constant Potts Model (CPM) value of a subset `X`
    pub fn cpm_of<'a, X>(&'a self, view: &'a X, resolution: f64) -> f64
    where
        X: AbstractSubset<'a>,
    {
        let ls = self.num_edges_inside(view);
        utils::calc_cpm_resolution(ls, view.num_nodes(), resolution)
    }

    pub fn modularity_of<'a, X>(&'a self, view: &'a X, resolution: f64) -> f64
    where
        X: AbstractSubset<'a>,
    {
        let total_l = self.m();
        let ls = self.num_edges_inside(view);
        let ds = view
            .each_node_id()
            .map(|&n| self.nodes[n].degree())
            .sum::<usize>();
        let modularity = utils::calc_modularity_resolution(ls, ds, total_l, resolution);
        modularity
    }

    pub fn treeness_of<'a, X>(&'a self, view: &'a X) -> f64
    where
        X: AbstractSubset<'a>,
    {
        let m = self.num_edges_inside(view) as f64;
        let n = view.num_nodes() as f64;
        return (m - n + 1.0) / n;
    }

    /// The minimum intra-cluster degree of a subset `X`
    pub fn mcd_of<'a, X>(&'a self, view: &'a X) -> Option<usize>
    where
        X: AbstractSubset<'a>,
    {
        self.degrees_inside(view).min()
    }
}

/// A "cluster" not in the mathematical sense but actually two sets
/// , a core and a periphery. See [AbstractSubset] for the actual mathematical subset.
///
/// To view different parts of the cluster as mathematical subsets
/// use [`Self::core()`], [`Self::periphery()`], [`Self::all()`].
///
/// # Design
/// Given input cluster `K`, intuitively AOC augments `K` greedily.
/// Intuitively, it only suffices to keep track of `K` (say, in a `HashSet`).
/// Unfortunately, the design of AOC with criteria `k` or `mcd` is
/// defined with respect to the "original" cluster, also known as the "core".
///
/// Therefore, it has been decided that a cluster has two components --
/// the "core", the original cluster, and also the "periphery",
/// containing all the nodes augmented to the cluster. One consequence
/// of this is that the entirety of the cluster (both the core and the periphery)
/// can be viewed only by [`Self::all()`].
#[derive(Default, Debug, Clone)]
pub struct Cluster {
    pub core_nodes: AHashSet<usize>,
    pub periphery_nodes: AHashSet<usize>,
}

impl Cluster {
    /// Add a node (using its node id) to the cluster.
    pub fn add_core(&mut self, node: usize) {
        self.core_nodes.insert(node);
    }

    /// Add a node (using its node id) to the periphery.
    pub fn add_periphery(&mut self, node: usize) {
        self.periphery_nodes.insert(node);
    }

    /// The minimum intra-cluster degree of the cluster at its core.
    /// A wrapper over [`Graph::mcd_of()`].
    pub fn core_mcd(&self, bg: &DefaultGraph) -> Option<usize> {
        bg.mcd_of(&self.core())
    }

    /// Number of nodes of the entirety (core + periphery)
    pub fn size(&self) -> usize {
        self.core_nodes.len() + self.periphery_nodes.len()
    }

    /// Is the `size` 1?
    pub fn is_singleton(&self) -> bool {
        self.size() == 1
    }

    /// Does the entirety of the cluster contain more than one node?
    pub fn is_non_trivial(&self) -> bool {
        self.size() > 1
    }

    /// Does either the core or the periphery contain the node?
    pub fn contains(&self, node: &usize) -> bool {
        self.core_nodes.contains(node) || self.periphery_nodes.contains(node)
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

/// Used internally in [ClusterView]
pub enum ClusterViewType {
    Core,
    Periphery,
}

/// Internally used for [`Cluster::core()`], [`Cluster::periphery()`]
pub struct ClusterView<'a> {
    pub cluster: &'a Cluster,
    pub view_type: ClusterViewType,
}

/// Internally used for [`Cluster::all()`]]
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
    /// Extract the subset of the graph associated with the core of the cluster.
    pub fn core(&'a self) -> ClusterView<'a> {
        ClusterView {
            cluster: self,
            view_type: ClusterViewType::Core,
        }
    }

    /// Extract the subset of the graph associated with the periphery of the cluster.
    pub fn periphery(&'a self) -> ClusterView<'a> {
        ClusterView {
            cluster: self,
            view_type: ClusterViewType::Periphery,
        }
    }

    /// Extract the subset of the graph associated with the entirety of the cluster.
    pub fn all(&'a self) -> ClusterEntireView<'a> {
        ClusterEntireView { cluster: self }
    }
}

/// A clustering, or a collection of labeled clusters.
/// The labelling is assumed to be integers.
pub struct Clustering {
    /// the raw data of the clusters, a map from the cluster id to the cluster
    pub clusters: BTreeMap<usize, Cluster>,
    /// "attention", defined as the minimum sized cluster that we want to focus/augment on
    pub attention: usize,
}

#[derive(Serialize, Deserialize)]
pub struct PackedClustering {
    pub clusters: BTreeMap<usize, RoaringBitmap>,
}

impl From<Clustering> for PackedClustering {
    fn from(clustering: Clustering) -> Self {
        let mut clusters = BTreeMap::new();
        for (cluster_id, cluster) in clustering.clusters {
            let bitmap = cluster
                .core_nodes
                .into_iter()
                .map(|it| it as u32)
                .collect::<RoaringBitmap>();
            clusters.insert(cluster_id, bitmap);
        }
        PackedClustering { clusters }
    }
}

impl From<PackedClustering> for Clustering {
    fn from(packed_clustering: PackedClustering) -> Self {
        let mut clusters = BTreeMap::new();
        for (cluster_id, bitmap) in packed_clustering.clusters {
            let cluster = Cluster {
                core_nodes: bitmap.into_iter().map(|it| it as usize).collect(),
                periphery_nodes: AHashSet::new(),
            };
            clusters.insert(cluster_id, cluster);
        }
        Clustering {
            clusters,
            attention: 0,
        }
    }
}

impl Clustering {
    /// Construct a clustering given a background graph and an existing
    /// `node_id cluster_id`-based clustering.
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
                (cluster_name, node_name)
            } else {
                (node_name, cluster_name)
            };
            let cluster_id = cluster_name
                .parse::<usize>()
                .map_err(|_| anyhow::anyhow!("invalid cluster_id"))?;
            let node_id = bg.retrieve(node_name.parse()?);
            match node_id {
                Some(node_id) => {
                    clusters.entry(cluster_id).or_default().add_core(node_id);
                }
                None => {
                    if !not_found_nodes.contains(&cluster_id) {
                        not_found_nodes.insert(cluster_id);
                    } else {
                        bail!("node {} requested in cluster {} not found in graph, and is not singleton", node_name, cluster_id);
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
        Ok(Clustering {
            clusters,
            attention: 0,
        })
    }

    /// Shorthand for `parse_from_reader`
    pub fn parse_from_file<P>(bg: &Graph<Node>, path: P, reverse: bool) -> anyhow::Result<Self>
    where
        P: AsRef<Path>,
    {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Self::parse_from_reader(bg, reader, reverse)
    }

    /// Write the clustering to a writer
    pub fn write_raw<W>(
        &self,
        mut writer: W,
        graph: &Graph<Node>,
        reverse_order: bool,
    ) -> anyhow::Result<()>
    where
        W: Write,
    {
        for (cluster_id, cluster) in self.clusters.iter() {
            for node_id in cluster
                .core_nodes
                .iter()
                .chain(cluster.periphery_nodes.iter())
            {
                let (cid, nid) = (cluster_id, graph.name_set.rev(*node_id).unwrap());
                if reverse_order {
                    writeln!(writer, "{}\t{}", cid, nid)?;
                } else {
                    writeln!(writer, "{}\t{}", nid, cid)?;
                }
            }
        }
        Ok(())
    }

    /// Shorthand over `write_raw`
    pub fn write_file<P>(
        &self,
        graph: &Graph<Node>,
        path: P,
        legacy_order: bool,
    ) -> anyhow::Result<()>
    where
        P: AsRef<Path>,
    {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        self.write_raw(writer, graph, legacy_order)
    }

    /// Only keep the clusters that satisfy `f`, analogous to `BTreeMap::retain`
    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&usize, &mut Cluster) -> bool,
    {
        self.clusters.retain(f);
    }

    /// The conductance of the clustering defined as the minimum over all cuts
    /// induced by the clusters.
    pub fn conductance(&self, graph: &Graph<Node>) -> Option<f64> {
        self.clusters
            .values()
            .map(|c| NotNan::new(graph.conductance_of(&c.all())).unwrap())
            .min()
            .map(|x| x.into_inner())
    }
}

impl<'a> Clustering {
    /// Mutable iterator over all clusters that are above the attention
    pub fn worthy_clusters(
        &'a mut self,
    ) -> impl ParallelIterator<Item = (&'a usize, &'a mut Cluster)> {
        self.clusters
            .par_iter_mut()
            .filter(|(_, cluster)| cluster.core_nodes.len() >= self.attention)
    }

    pub fn num_worthy_clusters(&self) -> usize {
        self.clusters
            .iter()
            .filter(|(_, cluster)| cluster.core_nodes.len() >= self.attention)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use ahash::AHashSet;
    use anyhow::Ok;
    use std::io::BufReader;

    use crate::{AbstractSubset, Cluster, Graph};

    use super::DefaultGraph;

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

    #[test]
    pub fn basic_graph_subset() -> anyhow::Result<()> {
        let edgelist = "0 1\n1 2\n2 3\n 3 4";
        let reader = BufReader::new(edgelist.as_bytes());
        let graph = Graph::parse_edgelist_from_reader(reader)?;
        let subset = graph.owned_subset(vec![1, 2, 3, 4]);
        let (n, m) = graph.count_n_m(&subset);
        assert_eq!(4, n);
        assert_eq!(3, m);
        assert_eq!(graph.edge_density_inside(&subset), 0.5);
        Ok(())
    }

    #[test]
    pub fn graph_from_edgelist() {
        let str_g = DefaultGraph::parse_edgelist_from_str("0 1\n1 2\n2 3\n 3 4").unwrap();
        let usize_g: DefaultGraph = [(0, 1), (1, 2), (2, 3), (3, 4)].into_iter().collect();
        assert_eq!(str_g, usize_g);
    }

    #[test]
    pub fn graph_from_iter() {
        let g: DefaultGraph = [
            (99999, 99999),
            (3, 4),
            (1, 2),
            (2, 3),
            (1033, 104),
            (104, 3),
        ]
        .into_iter()
        .collect();
        for (internal_id, &external_id) in g.name_set.rev.iter().enumerate() {
            assert_eq!(g.name_set.retrieve(external_id).unwrap(), internal_id);
            assert_eq!(g.name_set.rev(internal_id).unwrap(), external_id);
        }
    }
}
