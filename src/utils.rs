pub fn calc_modularity(ls: usize, ds: usize, big_l: usize) -> f64 {
    let (ls, ds, big_l) = (ls as f64, ds as f64, big_l as f64);
    (ls / big_l) - (ds / (2.0 * big_l)).powi(2)
}
