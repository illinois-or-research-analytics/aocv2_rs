use nom::{
    branch::alt,
    bytes::complete::tag_no_case,
    character::complete::{anychar, char, one_of},
    combinator::recognize,
    multi::{many0, many1},
    sequence::{terminated, tuple},
    IResult, Parser,
};

#[derive(Clone, Debug)]
pub enum CandidateSpecifier {
    NonSingleton(usize),
    File(String),
}

pub fn token<'a>(i: &'a str) -> impl Fn(&'a str) -> IResult<&'a str, &'a str> {
    tag_no_case(i)
}

pub fn decimal(input: &str) -> IResult<&str, &str> {
    recognize(many1(terminated(one_of("0123456789"), many0(char('_')))))(input)
}

pub fn parse_specifier(s: &str) -> Result<CandidateSpecifier, String> {
    if s.is_empty() {
        return Err("Empty specifier".to_string());
    }
    let mut ps = alt((
        tuple((token("cluster_size"), token(">="), decimal))
            .map(|(_, _, n)| CandidateSpecifier::NonSingleton(n.parse().unwrap())),
        many1(anychar).map(|f| CandidateSpecifier::File(f.into_iter().collect())),
    ));
    let (rest, spec) = ps(s).map_err(|e| format!("Failed to parse candidate specifier: {}", e))?;
    if !rest.is_empty() {
        return Err(format!("Failed to parse candidate specifier: {}", rest));
    }
    Ok(spec)
}
