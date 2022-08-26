AOCluster
===================

A rewrite of the Python version of assembling overlapping clusters. Currently, it is runnable but has discrepancies with the original Python version -- investigating.

## Getting Started

Install (TODO: add installation instructions) the `aocluster` binary first, then
run the following command for AOC_k with $k = 10$:

```bash
aocluster augment -g edgelistgraph.tsv -c existing_clustering.txt --mode k --min-k 10 -o output.txt
```

## Other Commands

### Packing graphs

```txt
aocluster-pack 
Pack an existing (large) graph into an internal binary format for speed

USAGE:
    aocluster pack --graph <GRAPH> --output <OUTPUT>

OPTIONS:
    -g, --graph <GRAPH>      Path to the edgelist graph
    -h, --help               Print help information
    -o, --output <OUTPUT>    Output path for the preprocessed graph, recommended suffix is `.bincode.lz4`
```

## Building

Install the [Rust toolchain](https://www.rust-lang.org/tools/install) and then compile the binary:

```bash
cargo build --release
```