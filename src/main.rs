mod aoc;
mod base;
mod utils;
use std::path::PathBuf;
use std::time::Instant;

use aoc::{augment_clusters, AocConfig};
pub use base::*;
use clap::{ArgEnum, Parser};
use itertools::Itertools;
use tracing::{debug, info};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ArgEnum, Debug)]
pub enum AocMode {
    M,
    K,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
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
}

fn args2aocconfig(args: &Args) -> AocConfig {
    match args.mode {
        AocMode::M => AocConfig::AocM(),
        AocMode::K => AocConfig::AocK(args.min_k.unwrap()),
    }
}

fn main() -> anyhow::Result<()> {
    let now = Instant::now();
    let starting = now.clone();
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    // debug!("Args: {:?}", args);
    let graph = Graph::parse_edgelist(&args.graph)?;
    info!("Graph loaded in {:?}", now.elapsed());
    let now = Instant::now();
    let mut clustering = Clustering::parse_from_file(&graph, &args.clustering)?;
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
    let config = args2aocconfig(&args);
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
    clustering.write_file(&graph, &args.output)?;
    info!("AOC finished in {:?}", starting.elapsed());
    Ok(())
}
