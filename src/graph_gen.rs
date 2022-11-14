pub mod concrete_examples {
    use crate::base::*;
    pub fn wikipedia_conductance_graph() -> DefaultGraph {
        // See https://en.wikipedia.org/wiki/Conductance_(graph)#/media/File:Graph_conductance.svg
        [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 3),
            (2, 3),
            (1, 4),
            (4, 6),
            (6, 5),
            (3, 5),
            (2, 5),
        ]
        .into_iter()
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::misc::{OnlineConductance, OwnedSubset};
    #[test]
    pub fn conductance_works() {
        let g = super::concrete_examples::wikipedia_conductance_graph();
        let s1 = [0].into_iter().collect::<OwnedSubset>();
        let s2 = [0, 1].into_iter().collect::<OwnedSubset>();
        let s3 = [0, 1, 2, 3, 4].into_iter().collect::<OwnedSubset>();
        assert_eq!(1.0, g.conductance_of(&s1));
        assert_eq!(2.0 / 3.0, g.conductance_of(&s2));
        assert_eq!(0.6, g.conductance_of(&s3));
    }

    #[test]
    pub fn online_conductance_works() {
        let g = super::concrete_examples::wikipedia_conductance_graph();
        let mut s1 = vec![0].into_iter().collect::<OwnedSubset>();
        let mut c = OnlineConductance::new(&g, &s1);
        assert_eq!(g.conductance_of(&s1), c.conductance());
        for &i in &[1usize, 2, 3, 4] {
            let new_c = c.add_node(&g, &g.nodes[i], &s1);
            s1.insert(i);
            let c2 = g.conductance_of(&s1);
            assert_eq!(c2, new_c, "i = {}", i);
        }
    }
}
