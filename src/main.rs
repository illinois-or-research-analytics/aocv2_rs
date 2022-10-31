mod aoc;
mod base;
mod dump;
mod generators;
mod graph_gen;
mod io;
mod misc;
mod quality;
mod utils;
use std::io::{BufRead, BufWriter};
use std::time::Instant;
use std::{io::BufReader, path::PathBuf};

use aoc::AocConfig;
pub use base::*;
use clap::{ArgEnum, Parser, Subcommand};
use inc_stats::Percentiles;
use io::CandidateSpecifier;
use itertools::Itertools;
use tracing::{info, warn};

use crate::aoc::augment_clusters_from_cli_config;
use crate::dump::dump_graph_to_json;
use crate::misc::NodeList;
use crate::quality::ClusterInformation;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ArgEnum, Debug)]
pub enum AocMode {
    M,
    K,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(subcommand)]
    cmd: SubCommand,
    #[clap(short = 't', long)]
    threads: Option<usize>,
}

#[derive(Subcommand, Debug)]
enum SubCommand {
    /// Augment an existing disjoint cluster
    Augment {
        /// Path to the edgelist graph
        #[clap(short, long)]
        graph: PathBuf,
        /// Path to the clustering file
        #[clap(short, long)]
        clustering: PathBuf,
        #[clap(long)]
        node_first_clustering: bool,
        #[clap(short, long, parse(try_from_str = aoc::parse_aoc_config))]
        mode: AocConfig,
        #[clap(long, parse(try_from_str = io::parse_specifier))]
        candidates: Option<CandidateSpecifier>,
        #[clap(short, long)]
        output: PathBuf,
    },

    /// Pack an existing (large) graph into an internal binary format for speed
    Pack {
        /// Path to the edgelist graph
        #[clap(short, long)]
        graph: PathBuf,
        /// Output path for the preprocessed graph, recommended suffix is `.bincode.lz4`
        #[clap(short, long)]
        output: PathBuf,
    },

    /// Calculate statistics for a given clustering/cluster
    Stats {
        /// Path to the edgelist graph
        #[clap(short, long)]
        graph: PathBuf,
        /// Path to the clusters/cluster file
        #[clap(short, long)]
        clusters: PathBuf,
        #[clap(long)]
        node_first_clustering: bool,
        /// Quality metric to calculate
        #[clap(short, long, parse(try_from_str = aoc::parse_aoc_config))]
        quality: Option<AocConfig>,
        /// If the given cluster file is only a newline separated node-list denoting one cluster
        #[clap(short, long)]
        single: bool,
        /// Output path for the statistics
        #[clap(short, long)]
        output: PathBuf,
    },

    /// Dump basic information about the graph
    Dump {
        /// Path to the graph
        #[clap(short, long)]
        graph: PathBuf,
        /// Output path for the dumped graph in json format
        #[clap(short, long)]
        output: PathBuf,
    },

    /// Filter a clustering
    Filter {
        /// Path to the edgelist graph
        #[clap(short, long)]
        graph: PathBuf,
        /// Path to the clusters/cluster file
        #[clap(short, long)]
        clusters: PathBuf,
        #[clap(long)]
        node_first_clustering: bool,
        /// Keep tree-like clusters
        #[clap(long)]
        keep_tree: bool,
        #[clap(long)]
        size_lower_bound: Option<usize>,
        #[clap(long)]
        percentile_lower_bound: Option<f64>,
        /// Output path
        #[clap(short, long)]
        output: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    // for benchmarking
    let now = Instant::now();
    let starting = now;
    // initialize logging
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let mut num_cpu = num_cpus::get_physical().min(32); // TODO: specify number of cores
    if let Some(specified_cores) = args.threads {
        if specified_cores > num_cpus::get_physical() {
            warn!("Specified more cores than available, using all available cores");
        } else {
            num_cpu = specified_cores;
        }
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpu)
        .build_global()?;
    match args.cmd {
        SubCommand::Augment {
            graph,
            clustering,
            node_first_clustering,
            mode,
            candidates,
            output,
        } => {
            let config = mode;
            info!("Augmenting clustering with config: {:?}", config);
            let graph = Graph::parse_from_file(&graph)?;
            info!(
                n = graph.n(),
                m = graph.m(),
                "Graph loaded in {:?}",
                now.elapsed()
            );
            let now = Instant::now();
            let mut clustering =
                Clustering::parse_from_file(&graph, &clustering, node_first_clustering)?;
            info!("Clustering loaded in {:?}", now.elapsed());
            info!(
                "Clustering contains {} clusters with {} entries",
                clustering.clusters.len(),
                clustering
                    .clusters
                    .values()
                    .map(|c| c.core_nodes.len())
                    .sum::<usize>()
            );
            let now = Instant::now();
            let candidates = candidates.unwrap_or(CandidateSpecifier::NonSingleton(2));
            info!("Candidates specified as: {:?}", candidates);
            let mut candidates = match candidates {
                CandidateSpecifier::NonSingleton(lb) => clustering
                    .clusters
                    .values()
                    .into_iter()
                    .filter(|c| c.size() >= lb)
                    .flat_map(|cluster| cluster.core_nodes.iter())
                    .copied()
                    .collect_vec(),
                CandidateSpecifier::File(p) => BufReader::new(std::fs::File::open(p)?)
                    .lines()
                    .map(|l| graph.retrieve(l.unwrap().trim()).unwrap())
                    .collect_vec(),
            };
            info!(
                "Candidates ({} of them) loaded in {:?}",
                candidates.len(),
                now.elapsed()
            );
            let now = Instant::now();
            augment_clusters_from_cli_config(&graph, &mut clustering, &mut candidates, &config);
            info!("Clusters augmented in {:?}", now.elapsed());
            clustering.write_file(&graph, &output, node_first_clustering)?;
        }
        SubCommand::Pack { graph, mut output } => {
            if !output.ends_with(".bincode.lz4") {
                output.set_extension("bincode.lz4");
                warn!(
                    "Output file does not end with .bincode.lz4, changing it to {:?}",
                    output
                );
            }
            let graph = Graph::parse_edgelist(&graph)?;
            utils::write_compressed_bincode(output, &graph)?;
        }
        SubCommand::Stats {
            graph,
            clusters,
            node_first_clustering,
            quality,
            single,
            output,
        } => {
            let quality = quality.unwrap_or(AocConfig::Mcd());
            let graph = Graph::parse_from_file(&graph)?;
            info!(
                n = graph.n(),
                m = graph.m(),
                "Graph loaded in {:?}",
                now.elapsed()
            );
            let entries = if single {
                let subset = NodeList::from_raw_file(&graph, &clusters)?.into_owned_subset();
                let ci = ClusterInformation::from_single_cluster(&graph, &subset, &quality);
                vec![ci]
            } else {
                let clustering =
                    Clustering::parse_from_file(&graph, &clusters, node_first_clustering)?;
                ClusterInformation::vec_from_clustering(&graph, &clustering, &quality)
            };
            let buf_writer = BufWriter::new(std::fs::File::create(output)?);
            let mut wtr = csv::Writer::from_writer(buf_writer);
            for entry in entries {
                wtr.serialize(entry)?;
            }
            wtr.flush()?;
        }
        SubCommand::Dump { graph, output } => {
            let graph = Graph::parse_from_file(&graph)?;
            dump_graph_to_json(&graph, &output)?;
        }
        SubCommand::Filter {
            graph,
            clusters,
            node_first_clustering,
            keep_tree,
            size_lower_bound,
            output,
            percentile_lower_bound,
        } => {
            let graph = Graph::parse_from_file(&graph)?;
            let mut clustering =
                Clustering::parse_from_file(&graph, &clusters, node_first_clustering)?;
            info!(
                "Clustering contains {} clusters with {} entries",
                clustering.clusters.len(),
                clustering
                    .clusters
                    .values()
                    .map(|c| c.core_nodes.len())
                    .sum::<usize>()
            );
            let percentile_size_lower_bound = percentile_lower_bound.map(|lb| {
                let perc: Percentiles<f64> = clustering
                    .clusters
                    .values()
                    .map(|c| c.size() as f64)
                    .collect();
                perc.percentile(lb).unwrap().unwrap()
            });
            clustering.retain(|_cid, c| {
                let core = c.core();
                if !keep_tree && core.num_nodes() > graph.num_edges_inside(&core) {
                    return false;
                }
                if let Some(lb) = size_lower_bound {
                    if c.size() < lb {
                        return false;
                    }
                }
                if let Some(lb) = percentile_size_lower_bound {
                    if (c.size() as f64) <= lb {
                        return false;
                    }
                }
                true
            });
            info!(
                "Clustering after filtering contains {} clusters with {} entries",
                clustering.clusters.len(),
                clustering
                    .clusters
                    .values()
                    .map(|c| c.core_nodes.len())
                    .sum::<usize>()
            );
            clustering.write_file(&graph, output, node_first_clustering)?;
        }
    }
    info!("AOC finished in {:?}", starting.elapsed());
    Ok(())
}
