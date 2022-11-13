AOCluster
===================

Generic overlapping cluster using greedy augmentation. Given a clustering of a network, this method enlarges the clusters by allowing nodes from the network to be added to the clusters. AOC performs this augmentation based on a user-selected cluster criterion score, such as modularity[^1], the Constant Potts Model[^2] (CPM) score, or the minimum intra-cluster degree[^3]. When considering whether a node can be added to a cluster, the addition is permitted only if the criterion score of the enlarged cluster does not drop below the criterion score of the original cluster.

[^1]: Newman, Mark EJ, and Michelle Girvan. "Finding and evaluating community structure in networks." *Physical review E* 69.2 (2004): 026113.

[^2]: Traag, Vincent A., Paul Van Dooren, and Yurii Nesterov. "Narrow scope for resolution-limit-free community detection." *Physical Review E* 84.1 (2011): 016114.

[^3]: Wedell, Eleanor, et al. "Center–periphery structure in research communities." *Quantitative Science Studies* 3.1 (2022): 289-314.

## Getting Started

First, install `aocluster` if not already installed. Either see Releases or see [installation instructions](#building) below.

Prepare an edgelist (undirected) graph consisting entirely of integers, such as the following graph:

```python
# graph.txt
# this is a comment. Comments are not supposed to be in the actual input
0 1
1 10
10 1 # duplicate edges / parallel edges will be ignored for robustness
# any non-newline whitespace separating node pairs can work
```

and a corresponding vertex membership file

```python
# clustering.txt
# this is a comment, not supposed to be in the actual input
# node_id, cluster_id
0 1
1 1
# singletons need not have a label
```

along with an [augmentation criterion](#quality-measures), e.g. `mcd`. Run the following command and see the new clustering at `output_clustering.txt`:

```bash
aocluster augment -c clustering.txt -g graph.txt -m mcd -o output_clustering.txt
```

See below for more options.

## Options

`aocluster augment` is the main sub-command.

Boldfaced parameters are required. 🌟 suggests an important algorithmic parameter -- the rest we tried to leave with reasonable defaults. Notation such as `{k:int}` denote a parameter that should be an integer, e.g., `k10` matches `k{k:int}`.

| Param | Description |
| --- | --- |
| **`-g {graphpath:string}`** 🌟 | Path to the background graph in edgelist/[packed](#graph-packing) format |
| **`-c {clusteringpath:string}`** 🌟 | Path to the clustering, default in `node_id cluster_id` format |
| **`-m {mode:enum(mode)}`** 🌟 | Augmenting criterion used for augmentation. See [here](#augmenting-criterion). |
| **`-o {outputpath:string}`** 🌟 | Path to the output clustering, will be in same format as the input clustering |
| `--augmentable-lb {a:int}` 🌟 | Defaults to `0`. Clusters below this size will not be augmented. Shorthand `-a {a:int}`. |
| `--candidates {c:candidate_spec}` | Specifies which nodes are eligible to be augmented to clusters. Defaults to `all`. See [here](#specifying-candidates). |
| `-s {strategy:enum(strategy)}` | What greedy heuristic to use to maximize the clusters. Defaults to `local`. See [here](#expansion-strategy). |
| `[--legacy-cid-nid-order]` | Is the input clustering in `cluster_id node_id` format (AOC legacy)? |

## Augmenting Criterion

Recall that $n = |V|$ and $m = |E|$ for a graph $G = (V, E)$. The current version of AOC is motivated by augmenting a cluster while not hurting its original "quality". Each of the following quality measures can be specified as an augmenting criterion, with a query $u$ added to $K_i$ if the augmented cluster has no worse quality than the input version of the cluster.

| Quality | Description |
| --- | --- |
| `cpm{r:double}` | CPM with resolution value $r$ |
| `mod{r:double}` | modularity with resolution value $r$. Use $r = 1$ for "vanilla" modularity. |
| `density` | $m / \binom{n}{2}$ |
| `mean-degree` | average degree inside the cluster |
| `conductance` | conductance of the cut induced by the cluster |

Minor note: `cpm{r:double}` is the obvious choice for CPM-based clustering, but `density` also makes sense, since CPM-based clustering guarantees that each cluster has density $\geq r$.

### Legacy Criteria

Legacy criteria are kept to preserve features from the previous version of AOC. Notably, the following two legacy criteria also consider the connectivity to the original not-augmented cluster. That is, let $O_i$ denote the original (unmodified) version of cluster $K_i$ (which might have already been augmented) -- the number of edges of a candidate node $u$ connected to $O_i$ will be considered. Also, the two criteria below also only augments $u$ to $K_i$ if $K_i \cup \{u\}$ has positive modularity. See the [original paper](#citations) for more information.

| Specifier | Description |
| --- | --- |
| `k{k:int}` | corresponds to `AOC_k` with $k$ |
| `mcd` (shorthand `m`) | corresponds to `AOC_m` |

## Specifying Candidates

| Specifier | Description |
| --- | --- |
| `cluster_size:{lb:int}` | All non-singleton cluster nodes with cluster size $\geq lb$ |
| `all` | All nodes |
| `degree:{lb:int}` | All nodes with global degree $\geq lb$ |
| `{filepath:string}` | All nodes from newline delimited file `filepath` |

By default, the candidate specifier is `all`.

## Expansion Strategy

| Strategy | Description |
| --- | --- |
| `legacy` | Candidate nodes are sorted by decreasing degree |
| `local` | The highest degree node to the current expanding cluster first |

## Examples

### Scenario 1: augmenting a Leiden-based clustering

Say that you have the following existing clustering that you want to augment into overlapping clusters

 - The background network is at `graph.txt`, in the edgelist/packed format
 - Leiden output (tab-delimited node id cluster id file) is at `clustering.txt`. The Leiden clustering was done specifying resolution $r = 0.1$ with CPM as the optimization criterion

What is left to be specified includes:
 - the output path (say you want it to be at `output.txt`)
 - the augmenting criterion. You want to choose either `density` or `cpm0.1`. Note that it is important to reuse the same $r$ value for `cpm{r:resolution}` here (i.e., $r = 0.1$)
 - `augmenting-lb`, in this case `0` seems like a reasonable value

The following command can be run:

```bash
aocluster augment -g graph.txt -c clustering.txt -m cpm0.1 --candidates all -o output.txt -a 0
```

### Scenario 2: emulating AOC legacy

Given the following configuration:

 - IKC[^3] was run with $k = 10$
 - Background graph located at `graph.txt`
 - IKC clustering (**converted** to the legacy-AOC expected cluster id node id format) located at `clustering.txt`
 - You aim to run AOC_k, so you specify the augmenting criterion as `k10` (also important to set this $k$ exactly the same as the above $k$, or in the case of AOC_m, specify this as `mcd`)
 - Use the `candidates.txt` file for the candidates
 - Output file intended to be at `output.txt` (same format as input)
 - `augmentable-lb` is 0 (the default)

```bash
aocluster augment -g graph.txt -c clustering.txt -q k10 --candidates candidates.txt --legacy-cid-nid-order --strategy legacy -o output.txt -a 0
```

## Other Features

### Command Documentation

Commands are somewhat self documenting. For example,
try `aocluster augment --help`. Or any command suffixed with `--help`.

### Progress Bar

`aocluster augment` when running under a terminal environment will display a progress bar of the algorithm.

### Clustering filtering

Sometimes a clustering method will produce a lot of small clusters. To filter these out, try `aocluster filter`:

```bash
# Filter out all size <= 10 clusters:
aocluster filter -g graph.txt -c clustering.txt --size-lower-bound 11 -o output_clustering.txt [--legacy-cid-nid-order]
```

### Graph Packing

```bash
aocluster pack -g graph.txt -o graph_packed.bincode.lz4
```

The `graph_packed.bincode.lz4` file can be used in lieu of `graph.txt`
for computational efficiency.

### Clustering Statistics

`aocluster` supports an efficient evaluation of clustering statistics.

```bash
aocluster stats -c clusters.txt -g graph.txt --global global_stats.json -o local_stats.csv [-q quality]
```

### Parallelism

`aocluster` by default uses `min(32, num_physical_cores)` number of threads. To specify the number of threads to use,
append `-t num_threads` after `aocluster` before any subcommands. For example,

```bash
aocluster -t 8 stats [...]
```

## Building

Install the [Rust toolchain](https://www.rust-lang.org/tools/install) and then compile the binary:

```bash
cargo build --release
```

To install the binary in path:

```bash
cargo install --path .
```

## Citations

See also the paper for the previous version of AOC:

> Jakatdar, A., Liu, B., Warnow, T., & Chacko, G. (2022). AOC; Assembling Overlapping Communities. arXiv preprint arXiv:2208.04842.

## Credits

This README is written by both Baqiao Liu and Tandy Warnow.