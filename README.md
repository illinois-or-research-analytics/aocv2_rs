AOCluster
===================

Generic overlapping cluster using greedy augmentation. In other words, **a**ssembling **o**verlapping **c**lusters using an existing clustering.

## Getting Started

First, install `aocluster` if not already installed. Either see Releases or see [installation instructions](#building) below.

Prepare an edgelist (undirected) graph consisting entirely of integers, such as the following graph:

```python
# graph.txt
# this is a comment. Comments are not supposed to be in the actual input
0 1
1 10
10 1 # duplicate edges / parallel edges will be ignored for robustness
# any whitespace separating nodes can work: tabs, spaces, etc.
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

Boldfaced parameters are required. Notation such as `{k:int}` denote a parameter that should be an integer, e.g., `k10` matches `k{k:int}`.

| Param | Description |
| --- | --- |
| **`-g {graphpath:string}`** | Path to the graph in edgelist/[packed](#graph-packing) format |
| **`-c {clusteringpath:string}`** | Path to the clustering, default in `node_id cid` format |
| **`-m {mode:enum(mode)}`** | Quality used for augmentation. See [here](#quality-measures). |
| **`-o {outputpath:string}`** | Path to the output clustering, same format as the input clustering |
| `--attention {a:int}` | Defaults to `11`. Clusters below this size will not be augmented. |
| `--candidates {c:candidate_spec}` | Defaults to `all`. See [here](#specifying-candidates). |
| `-s {strategy:enum(strategy)}` | What strategy to maximize the clusters. Defaults to `local`. See [here](#expansion-strategy). |
| `[--legacy-cid-nid-order]` | Is the input clustering in `cid nid` format (AOC legacy)? |

## Quality Measures

Recall that $n = |V|$ and $m = |E|$.

| Specifier | Description |
| --- | --- |
| `cpm{r:double}` | CPM with resolution value $r$ |
| `mod{r:double}` | modularity with resolution value $r$ |
| `density` | $m / \binom{n}{2}$ |
| `k{k:int}` | minimum intra-cluster degree must $\geq k$ |
| `mcd` (or `m`) | minimum intra-cluster degree |
| `mean-degree` | average degree inside the cluster |
| `conductance` | conductance of the cut induced by the cluster |

Examples include `cpm0.1` and `k100`. This parameter must be specified (no default value).

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

Given the following configuration:

 - $\gamma = 0.1$ for the resolution parameter
 - Existing graph located at `graph.txt`
 - Leiden clustering (tab-delimited node id cluster id file) at `clustering.txt`
 - The quality is specified as `cpm0.1` fitting the $\gamma$ above
 - Candidates are all the nodes (`all`)
 - Output file intended to be at `output.txt` (same format as input)
 - Attention is set to `0`

The following command can be run:

```bash
aocluster augment -g graph.txt -c clustering.txt -m cpm0.1 --candidates all -o output.txt -a 0
```

### Scenario 2: emulating AOC legacy

Given the following configuration:

 - IKC was run with $k = 10$
 - Existing graph located at `graph.txt`
 - IKC clustering (**converted** to the legacy-AOC expected cluster id node id format) located at `clustering.txt`
 - The quality is specified as `k10` (fitting the $k$, or just `mcd`)
 - Candidates are all the non-singleton nodes in clusters $\geq 10$
 - Use the `candidates.txt` file for the candidates
 - Use the `legacy` expansion strategy
 - Output file intended to be at `output.txt` (same format as input)
 - Attention is set to `0` (equivalent to default `-a 11`)

```bash
aocluster augment -g graph.txt -c clustering.txt -q k10 --candidates candidates.txt --legacy-cid-nid-order --strategy legacy -o output.txt -a 0
```

## Other Features

### External Documentation

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