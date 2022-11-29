use crate::DefaultGraph;

#[derive(Debug, Clone)]
pub struct CCLabels {
    pub labels: Vec<u32>,
    pub num_nodes: Vec<u32>,
}

pub fn cc_labeling(g: &DefaultGraph) -> CCLabels {
    let mut visited = vec![false; g.n()];
    let mut colors = vec![0; g.n()];
    let mut color = 0u32;
    let mut color_counts = vec![];
    for i in 0..g.n() {
        if !visited[i] {
            let r = color_from(g, &mut visited, &mut colors, i, color);
            color_counts.push(r);
            color += 1;
        }
    }
    CCLabels {
        labels: colors,
        num_nodes: color_counts,
    }
}

fn color_from(
    g: &DefaultGraph,
    visited: &mut [bool],
    colors: &mut [u32],
    node: usize,
    color: u32,
) -> u32 {
    let mut colored = 0u32;
    let mut stack = vec![node];
    while let Some(node) = stack.pop() {
        if visited[node] {
            continue;
        }
        visited[node] = true;
        colors[node] = color;
        colored += 1;
        for &edge in &g.nodes[node].edges {
            stack.push(edge);
        }
    }
    colored
}
