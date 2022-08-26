mod aoc;
mod base;
mod utils;
use std::path::PathBuf;
use std::time::Instant;

use aoc::{augment_clusters, AocConfig};
pub use base::*;
use clap::{ArgEnum, Parser, Subcommand};
use itertools::Itertools;
use tracing::{info, warn};

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
    Augment {
        /// Path to the edgelist graph
        #[clap(short, long)]
        graph: PathBuf,
        /// Path to the clustering file
        #[clap(short, long)]
        clustering: PathBuf,
        #[clap(short = 'k', long)]
        min_k: Option<usize>,
        #[clap(short, long, arg_enum)]
        mode: AocMode,
        #[clap(short, long)]
        output: PathBuf,
    },

    Pack {
        /// Path to the edgelist graph
        #[clap(short, long)]
        graph: PathBuf,
        #[clap(short, long)]
        output: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    let now = Instant::now();
    let starting = now;
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    match args.cmd {
        SubCommand::Augment {
            graph,
            clustering,
            min_k,
            mode,
            output,
        } => {
            let config = {
                match mode {
                    AocMode::M => AocConfig::AocM(),
                    AocMode::K => AocConfig::AocK(min_k.unwrap()),
                }
            };
            let graph = Graph::parse_from_file(&graph)?;
            info!("Graph loaded in {:?}", now.elapsed());
            let now = Instant::now();
            let mut clustering = Clustering::parse_from_file(&graph, &clustering)?;
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
            let mut candidates = clustering
                .clusters
                .values()
                .into_iter()
                .flat_map(|cluster| cluster.core_nodes.iter())
                .copied()
                .collect_vec();
            info!("Candidates built in {:?}", now.elapsed());
            let now = Instant::now();
            augment_clusters(&graph, &mut clustering, &mut candidates, config);
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
    }

    info!("AOC finished in {:?}", starting.elapsed());
    Ok(())
}
