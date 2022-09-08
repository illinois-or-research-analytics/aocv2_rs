mod aoc;
mod base;
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
use io::CandidateSpecifier;
use itertools::Itertools;
use tracing::{info, warn};

use crate::aoc::augment_clusters_from_cli_config;
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
        /// If the given cluster file is only a newline separated node-list denoting one cluster
        #[clap(short, long)]
        single: bool,
        /// Output path for the statistics
        #[clap(short, long)]
        output: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    let now = Instant::now();
    let starting = now;
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let num_cpu = num_cpus::get_physical().min(4);
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
            clustering.write_file(&graph, &output)?;
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
            single,
            output,
        } => {
            let graph = Graph::parse_from_file(&graph)?;
            info!(
                n = graph.n(),
                m = graph.m(),
                "Graph loaded in {:?}",
                now.elapsed()
            );
            let entries = if single {
                let subset = NodeList::from_raw_file(&graph, &clusters)?.into_owned_subset();
                let ci = ClusterInformation::from_single_cluster(&graph, &subset);
                vec![ci]
            } else {
                let clustering =
                    Clustering::parse_from_file(&graph, &clusters, node_first_clustering)?;
                ClusterInformation::vec_from_clustering(&graph, &clustering)
            };
            let buf_writer = BufWriter::new(std::fs::File::create(output)?);
            let mut wtr = csv::Writer::from_writer(buf_writer);
            for entry in entries {
                wtr.serialize(entry)?;
            }
            wtr.flush()?;
        }
    }
    info!("AOC finished in {:?}", starting.elapsed());
    Ok(())
}
