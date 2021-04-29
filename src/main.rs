mod board;
mod graph;
mod reader;

use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(parse(from_os_str))]
    filename: PathBuf,
}

fn main() {
    let opt = Opt::from_args();
    let board = reader::read_file(&opt.filename);
    println!("Read Board:\n{}", board);
    let results = board.solve();
    println!("Found {} results", results.len());
    for board in results {
        println!("Result\n{}", board);
    }
}
