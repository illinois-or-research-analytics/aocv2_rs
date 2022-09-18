use std::{collections::BTreeMap, path::Path};

use serde::{Deserialize, Serialize};

use crate::{Graph, Node};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphDump {
    pub n: usize,
    pub m: usize,
    pub degrees: BTreeMap<String, usize>,
}

pub fn dump_graph(graph: &Graph<Node>) -> GraphDump {
    let mut degrees = BTreeMap::new();
    for node in graph.nodes.iter() {
        let degree = node.degree();
        let name = graph.name_set.rev(node.id).unwrap();
        degrees.insert(name.to_string(), degree);
    }
    GraphDump {
        n: graph.n(),
        m: graph.m(),
        degrees,
    }
}

pub fn dump_graph_to_json<P: AsRef<Path>>(graph: &Graph<Node>, p: P) -> anyhow::Result<()> {
    let dump = dump_graph(graph);
    let json = serde_json::to_string_pretty(&dump)?;
    std::fs::write(p, json)?;
    Ok(())
}
