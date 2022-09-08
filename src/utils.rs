use bimap::BiMap;
use lz4::EncoderBuilder;
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Serialize, Deserialize)]
pub struct NameSet {
    pub next_id: usize,
    pub bimap: BiMap<String, usize>,
}

impl NameSet {
    pub fn retrieve(&self, name: &str) -> Option<usize> {
        self.bimap.get_by_left(name).copied()
    }

    pub fn rev(&self, id: usize) -> Option<&str> {
        self.bimap.get_by_right(&id).map(|it| it.as_str())
    }
}

pub fn calc_modularity(ls: usize, ds: usize, big_l: usize) -> f64 {
    let (ls, ds, big_l) = (ls as f64, ds as f64, big_l as f64);
    (ls / big_l) - (ds / (2.0 * big_l)).powi(2)
}

pub fn calc_modularity_resolution(ls: usize, ds: usize, big_l: usize, resolution: f64) -> f64 {
    let (ls, ds, big_l) = (ls as f64, ds as f64, big_l as f64);
    (ls / big_l) - resolution * (ds / (2.0 * big_l)).powi(2)
}

pub fn write_compressed_bincode<S, P>(path: P, data: &S) -> anyhow::Result<()>
where
    S: Serialize,
    P: AsRef<std::path::Path>,
{
    let mut file = std::fs::File::create(path)?;
    let mut encoder = EncoderBuilder::new().level(4).build(&mut file)?;
    bincode::serialize_into(&mut encoder, data)?;
    let (_, res) = encoder.finish();
    res?;
    Ok(())
}

pub fn read_compressed_bincode<S, P>(path: P) -> anyhow::Result<S>
where
    S: for<'de> Deserialize<'de>,
    P: AsRef<std::path::Path>,
{
    let mut file = std::fs::File::open(path)?;
    let mut decoder = lz4::Decoder::new(&mut file)?;
    let data = bincode::deserialize_from(&mut decoder)?;
    Ok(data)
}
