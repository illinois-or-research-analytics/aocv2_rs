use std::fmt::Display;

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
    Everything(),
    Degree(usize),
}

pub fn token<'a>(i: &'a str) -> impl Fn(&'a str) -> IResult<&'a str, &'a str> {
    tag_no_case(i)
}

pub fn decimal(input: &str) -> IResult<&str, &str> {
    recognize(many1(terminated(one_of("0123456789"), many0(char('_')))))(input)
}

pub fn parse_candidates_specifier(s: &str) -> Result<CandidateSpecifier, String> {
    if s.is_empty() {
        return Err("Empty specifier".to_string());
    }
    let mut ps = alt((
        tuple((token("cluster_size"), token(":"), decimal))
            .map(|(_, _, n)| CandidateSpecifier::NonSingleton(n.parse().unwrap())),
        tuple((token("degree"), token(":"), decimal))
            .map(|(_, _, n)| CandidateSpecifier::Degree(n.parse().unwrap())),
        token("all").map(|_| CandidateSpecifier::Everything()),
        many1(anychar).map(|f| CandidateSpecifier::File(f.into_iter().collect())),
    ));
    let (rest, spec) = ps(s).map_err(|e| format!("Failed to parse candidate specifier: {}", e))?;
    if !rest.is_empty() {
        return Err(format!("Failed to parse candidate specifier: {}", rest));
    }
    Ok(spec)
}

#[derive(Debug, PartialEq)]
pub enum FilesSpecifier {
    SingleFile(String),
    FileFamily(String, Vec<String>),
}

pub fn parse_files_specifier(s: &str) -> Result<FilesSpecifier, String> {
    match s.split_once(":") {
        Some((filename, indices)) => {
            if !filename.contains("{}") {
                return Err("File family specifier must contain a '{}'".to_string());
            }
            let indices = indices
                .split(",")
                .map(|i| {
                    i.parse()
                        .map_err(|e| format!("Failed to parse index: {}", e))
                })
                .collect::<Result<Vec<String>, String>>()?;
            Ok(FilesSpecifier::FileFamily(filename.to_string(), indices))
        }
        None => Ok(FilesSpecifier::SingleFile(s.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_parse_filename_specifier() {
        assert_eq!(
            parse_files_specifier("foo.txt"),
            Ok(FilesSpecifier::SingleFile("foo.txt".to_string()))
        );
        assert_eq!(
            parse_files_specifier("foo.txt:1,2,3"),
            Ok(FilesSpecifier::FileFamily(
                "foo.txt".to_string(),
                vec!["1".to_string(), "2".to_string(), "3".to_string()]
            ))
        );
    }
}
