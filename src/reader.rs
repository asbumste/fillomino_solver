use crate::board;

use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::digit1;
use nom::combinator::{eof, map_res};
use nom::multi::{many1, separated_list1};
use nom::sequence::terminated;
use std::fs;
use std::path::Path;

fn get_field(input: &str) -> nom::IResult<&str, u8> {
    alt((
        (map_res(digit1, |s: &str| s.parse::<u8>())),
        (map_res(tag(""), |_| "0".parse::<u8>())),
    ))(input)
}

#[test]
fn test_get_field() {
    assert_eq!(get_field("12\t34"), Ok(("\t34", 12)));
    assert_eq!(get_field("123"), Ok(("", 123)));
    assert_eq!(get_field(""), Ok(("", 0)));
}

fn get_line(input: &str) -> nom::IResult<&str, Vec<u8>> {
    terminated(separated_list1(tag("\t"), get_field), tag("\r\n"))(input)
}

#[test]
fn test_get_line() {
    assert_eq!(get_line("1\t2\t3\t4\r\n"), Ok(("", vec![1, 2, 3, 4])));
    assert_eq!(get_line("1\t2\t3\t\t4\r\n"), Ok(("", vec![1, 2, 3, 0, 4])));
    assert_eq!(get_line("\t\t\t\r\n"), Ok(("", vec![0, 0, 0, 0])));
    assert_eq!(
        get_line("\t1\t\t3\t\t4\t\r\n"),
        Ok(("", vec![0, 1, 0, 3, 0, 4, 0]))
    );
}

fn get_grid(input: &str) -> nom::IResult<&str, Vec<Vec<u8>>> {
    //TODO don't require EOL at EOF...
    terminated(many1(get_line), eof)(input)
}

#[test]
fn test_get_grid() {
    assert_eq!(
        get_grid("1\t\t2\t\r\n\t\t\t\r\n"),
        Ok(("", vec![vec![1, 0, 2, 0], vec![0, 0, 0, 0]]))
    );
}

pub fn read_board(contents: &str) -> board::Board {
    let (_, grid) = get_grid(contents).expect("Failed to read grid");
    // TODO handle non-square boards?
    let grid_size = grid.len();

    board::Board::new(grid_size, &grid)
}

#[test]
fn test_read_board() {
    let input = "1\t2\t\t\r\n\t\t\t\r\n2\t\t3\t\r\n4\t\t\t\r\n";
    let expected = board::Board::new(
        4,
        &vec![
            vec![1, 2, 0, 0],
            vec![0, 0, 0, 0],
            vec![2, 0, 3, 0],
            vec![4, 0, 0, 0],
        ],
    );

    assert_eq!(read_board(input), expected);
}

pub fn read_file(filename: &Path) -> board::Board {
    let contents = fs::read_to_string(filename).expect("Failed to read file");
    read_board(&contents)
}

#[test]
fn test_read_file() {
    let expected = board::Board::new(
        8,
        &vec![
            vec![1, 5, 0, 0, 1, 0, 1, 0],
            vec![0, 0, 0, 0, 4, 2, 0, 0],
            vec![0, 0, 1, 4, 4, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 1, 7, 0],
            vec![7, 1, 5, 0, 1, 8, 0, 0],
            vec![1, 0, 0, 1, 0, 1, 0, 0],
            vec![0, 0, 0, 0, 6, 0, 0, 1],
            vec![0, 7, 1, 0, 0, 1, 0, 0],
        ],
    );
    assert_eq!(read_file(Path::new("input/test")), expected);
}
