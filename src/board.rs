use petgraph::algo::dijkstra;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Undirected;
use std::collections::hash_map::HashMap;
use std::collections::HashSet;
use std::fmt;

type NodeGraph = Graph<Node, usize, Undirected>;

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
struct Pos {
    x: usize,
    y: usize,
}

impl Pos {
    pub fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }
}

impl fmt::Display for Pos {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)?;
        Ok(())
    }
}

#[derive(Debug, PartialEq, Clone)]
struct Node {
    value: Option<u8>,
    coord: Vec<Pos>,
    size: u8,
}

fn abs_difference(x: usize, y: usize) -> usize {
    if x < y {
        y - x
    } else {
        x - y
    }
}

impl Node {
    pub fn new(value: Option<u8>, coord: Vec<Pos>, size: u8) -> Self {
        Node { value, coord, size }
    }
    pub fn adjacent(graph: &NodeGraph, node1: NodeIndex, node2: NodeIndex) -> bool {
        for pos1 in &graph[node1].coord {
            for pos2 in &graph[node2].coord {
                match (
                    abs_difference(pos1.x, pos2.x),
                    abs_difference(pos1.y, pos2.y),
                ) {
                    (0, _) => {
                        return true;
                    }
                    (_, 0) => {
                        return true;
                    }
                    (_, _) => (),
                }
            }
        }
        false
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}/{}: ", self.size, self.value.unwrap_or(0))?;
        for pos in &self.coord {
            write!(f, "{} ", pos)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Board {
    width: usize,
    height: usize,
    graph: NodeGraph,
}

impl Board {
    pub fn new(width: usize, height: usize, grid: &Vec<Vec<u8>>) -> Self {
        // Iterate over the grid and create a node for each point
        let mut graph: NodeGraph = Graph::new_undirected();
        let mut y: usize = 0;
        let mut last_row: Vec<NodeIndex> = Vec::new();
        for row in grid {
            let mut x: usize = 0;
            let mut this_row: Vec<NodeIndex> = Vec::new();
            let mut last_node: Option<NodeIndex> = None;

            for value in row {
                let node = match value {
                    0 => graph.add_node(Node::new(None, vec![Pos::new(x, y)], 1)),
                    n => graph.add_node(Node::new(Some(*n), vec![Pos::new(x, y)], 1)),
                };

                // Add neighbours to nodes that need them
                if graph[node].value.is_none() || graph[node].value.unwrap() != 1 {
                    if let Some(last_node) = last_node {
                        if graph[last_node].value.is_none() || graph[last_node].value.unwrap() != 1
                        {
                            graph.add_edge(node, last_node, 1);
                        }
                    }

                    if !last_row.is_empty()
                        && (graph[last_row[x]].value.is_none()
                            || graph[last_row[x]].value.unwrap() != 1)
                    {
                        graph.add_edge(node, last_row[x], 1);
                    }
                }
                last_node = Some(node);
                this_row.push(node);

                x += 1;
            }
            last_row = this_row;
            y += 1;
        }

        Self::merge_neighbours(&mut graph).expect("Failed to simplify");

        Self::draw_graph(&graph);
        if !Self::is_valid(&graph) {
            panic!("Input not valid");
        }
        Self {
            width,
            height,
            graph,
        }
    }

    pub fn solve(&self) -> Vec<Board> {
        Board::solve_graph(&mut self.graph.clone())
            .iter()
            .map(|graph| Self {
                width: self.width,
                height: self.height,
                graph: graph.clone(),
            })
            .collect()
    }

    fn solve_graph(graph: &mut NodeGraph) -> Vec<NodeGraph> {
        if Self::is_solved(graph) {
            vec![graph.clone()]
        } else if let Ok(results) = Self::pulse(graph) {
            results
        } else {
            Vec::new()
        }
    }

    fn merge(
        graph: &mut NodeGraph,
        mut index: NodeIndex,
        mut other_idx: NodeIndex,
    ) -> Result<NodeIndex, &'static str> {
        // graph.remove_node inis_valids last node index,
        // so keep lesser index to prevent it from being inis_validd
        if other_idx < index {
            let tmp = other_idx;
            other_idx = index;
            index = tmp;
        }

        if graph[index].value.is_none() {
            graph[index].value = graph[other_idx].value;
        }

        // Set the value, but don't merge nodes that aren't adjacent
        // TODO "pulse" each path between nodes
        if !Node::adjacent(graph, index, other_idx) {
            // Guarantees both nodes are set, by above block
            eprintln!(
                "Setting value, but not merging non-adjacent nodes {} {}",
                graph[index], graph[other_idx]
            );
            graph[other_idx].value = graph[index].value;
            return Ok(index);
        }

        graph[index].size += graph[other_idx].size;

        let other_coord = graph[other_idx].coord.clone();
        graph[index].coord.extend(other_coord);

        if graph[index].value.is_some() && graph[index].value.unwrap() < graph[index].size {
            /*eprintln!(
                "Merge would overflow node {} {}",
                graph[index], graph[other_idx]
            );*/
            return Err("Merge would overflow node");
        }

        // Add valid neighbours of other_idx to index
        let mut neighbours = graph.neighbors(other_idx).detach();
        while let Some(neighbour_idx) = neighbours.next_node(&graph) {
            if neighbour_idx != index {
                match (graph[index].value, graph[neighbour_idx].value) {
                    (Some(a), Some(b)) if a != b => continue,
                    _ => {
                        graph.update_edge(index, neighbour_idx, 1);
                    }
                }
            }
        }

        graph.remove_node(other_idx);

        // Merge new neighbours that share a value
        let mut updated = true;
        while updated {
            updated = false;
            let mut neighbours = graph.neighbors(index).detach();
            while let Some(neighbour_idx) = neighbours.next_node(&graph) {
                if graph[index].value.is_some() && graph[index].value == graph[neighbour_idx].value
                {
                    index = Self::merge(graph, index, neighbour_idx)?;
                    updated = true;
                    break;
                }
            }
        }

        Ok(index)
    }

    fn merge_neighbour(graph: &mut NodeGraph, node_idx: NodeIndex) -> Result<bool, &'static str> {
        let mut updated = true;
        let mut updated_once = false;
        while updated {
            updated = false;
            let mut neighbors = graph.neighbors(node_idx).detach();
            while let Some(n) = neighbors.next_node(graph) {
                match (graph[node_idx].value, graph[n].value) {
                    (Some(v), Some(u)) if v == u => {
                        Board::merge(graph, node_idx, n)?;
                        updated = true;
                        break;
                    }
                    (Some(_), Some(_)) => {
                        graph.remove_edge(graph.find_edge(node_idx, n).unwrap());
                        updated = true;
                        break;
                    }
                    _ => (),
                }
            }
            if updated {
                updated_once = true;
            }
        }

        Ok(updated_once)
    }

    fn merge_neighbours(graph: &mut NodeGraph) -> Result<(), &'static str> {
        // Merge adjacent nodes with the same value
        // Remove edges between nodes with differing values
        let mut updated = true; // Only update a single node at a time, removing nodes inis_valids indices
        while updated {
            updated = false;
            for node_idx in graph.node_indices() {
                updated = Self::merge_neighbour(graph, node_idx)?;
                if updated {
                    break;
                }
            }
        }
        Ok(())
    }

    fn pulse(graph: &mut NodeGraph) -> Result<Vec<NodeGraph>, &'static str> {
        let mut results = Vec::new();

        for node_idx in graph.node_indices() {
            if graph[node_idx].value.is_none() {
                for i in 1..10 {
                    let mut g = graph.clone();
                    g[node_idx].value = Some(i);
                    if Self::merge_neighbour(&mut g, node_idx).is_ok() && Self::is_valid(&g) {
                        eprintln!("Pulsed graph {} = {}", g[node_idx], i);
                        Self::draw_grid(&g);
                        results.extend(Board::solve_graph(&mut g));
                        if !results.is_empty() {
                            return Ok(results);
                        }
                    }
                }
                /* TODO only do reachable unstead of all values
                eprintln!("Reachable from: {}", graph[node_idx]);
                let reachable = Self::get_reachable(graph, node_idx);
                for r in reachable {
                    eprintln!("\t{}", graph[r]);
                    let mut g = graph.clone();
                    if Board::merge(&mut g, node_idx, r).is_ok() {
                        if Board::is_valid(&g) {                                                        ;
                            results.extend(Board::solve_graph(&mut g));
                            if !results.is_empty() {
                                return Ok(results);
                            }
                        }
                    }

                }
                */

                // If there are no valid values for this node,
                //continuing to search isn't going to help
                break;
            }
        }
        Ok(results)
    }

    fn get_node(graph: &NodeGraph, pos: &Pos) -> Option<NodeIndex> {
        for node_idx in graph.node_indices() {
            for coord in &graph[node_idx].coord {
                if pos == coord {
                    return Some(node_idx);
                }
            }
        }
        None
    }

    fn is_solved(graph: &NodeGraph) -> bool {
        if !Self::is_valid(graph) {
            false
        } else {
            for node_idx in graph.node_indices() {
                match (graph[node_idx].value, graph[node_idx].size) {
                    (None, _) => return false,
                    (Some(n), m) if n != m => return false,
                    (_, _) => (),
                }
            }
            true
        }
    }

    // Get all nodes that could possibly be merged with node_idx
    fn get_reachable(graph: &NodeGraph, node_idx: NodeIndex) -> HashSet<NodeIndex> {
        if graph[node_idx].value.is_none() {
            // Cost Some(n) -> None = 1000,
            // None -> None = 1
            // A valid path only has a single Some(n) -> None
            let reachable: HashMap<NodeIndex, usize> =
                dijkstra(graph, node_idx, None, |e| {
                    match (graph[e.source()].value, graph[e.target()].value) {
                        (Some(_), Some(_)) => 10000,
                        (Some(_), None) => 1000,
                        (None, Some(_)) => 1000,
                        (None, None) => 1,
                    }
                });
            reachable
                .iter()
                .filter(|(idx, cost)| {
                    node_idx != **idx
                        && graph[**idx].value.is_some()
                        && **cost + (graph[**idx].size as usize) + (graph[node_idx].size as usize)
                            <= (graph[**idx].value.unwrap() as usize) + 1000
                })
                .map(|(idx, _)| *idx)
                .collect()
        } else {
            let reachable: HashMap<NodeIndex, usize> = dijkstra(graph, node_idx, None, |e| {
                match (graph[e.source()].value, graph[e.target()].value) {
                    (Some(_), Some(_)) => 1, // Shouldn't happen, will fail in later check
                    (Some(_), None) => 1000,
                    (None, Some(_)) => 1000,
                    (None, None) => 1,
                }
            });

            reachable
                .iter()
                .filter(|(idx, cost)| {
                    node_idx != **idx
                        && ((graph[**idx].value.is_none()
                            && **cost
                                + (graph[**idx].size as usize)
                                + (graph[node_idx].size as usize)
                                <= (graph[node_idx].value.unwrap() as usize) + 1000)
                            || (graph[**idx].value == graph[node_idx].value
                                && **cost
                                    + (graph[**idx].size as usize)
                                    + (graph[node_idx].size as usize)
                                    <= (graph[node_idx].value.unwrap() as usize) + 2000))
                })
                .map(|(idx, _)| *idx)
                .collect()
        }
    }

    // Get all nodes that are adjacent to node_idx that aren't neighbours
    fn get_adjacent(graph: &NodeGraph, node_idx: NodeIndex) -> HashSet<NodeIndex> {
        let mut check_pos = HashSet::new();
        for pos in &graph[node_idx].coord {
            check_pos.insert(Pos::new(pos.x + 1, pos.y));
            check_pos.insert(Pos::new(pos.x, pos.y + 1));
            if pos.x > 0 {
                check_pos.insert(Pos::new(pos.x - 1, pos.y));
            }
            if pos.y > 0 {
                check_pos.insert(Pos::new(pos.x, pos.y - 1));
            }
        }

        check_pos
            .iter()
            .filter_map(|p| Self::get_node(graph, p))
            .filter(|idx| *idx != node_idx && graph.find_edge(node_idx, *idx).is_none())
            .collect()
    }

    fn is_valid(graph: &NodeGraph) -> bool {
        for node_idx in graph.node_indices() {
            if !Self::is_valid_node(graph, node_idx) {
                //eprintln!("{} not valid", graph[node_idx]);
                //Self::draw_neighbours(graph, node_idx);
                return false;
            }
        }
        true
    }

    // Return true iff node_idx doesn't break any rule
    fn is_valid_node(graph: &NodeGraph, node_idx: NodeIndex) -> bool {
        if graph[node_idx].value.is_none() {
            // Cost Some(n) -> None = 1000,
            // None -> None = 1
            // A valid path only has a single Some(n) -> None
            if Self::get_reachable(graph, node_idx).is_empty() {
                //eprintln!("Node not valid, no reachable");
                false
            } else {
                true
            }
        } else {
            let mut size = graph[node_idx].size;
            for idx in Self::get_reachable(graph, node_idx) {
                if graph[idx].value.is_some() && graph[idx].value != graph[node_idx].value {
                    //eprintln!("Node not valid, numbered neighbour: {} {}", graph[node_idx], graph[idx]);
                    return false;
                }
                size += graph[idx].size;
            }
            // TODO need to check that no two nodes with the same value are adjacent
            if size < graph[node_idx].value.unwrap() {
                //eprintln!("Node not valid, overflow: {}", graph[node_idx]);
                false
            } else {
                // check adjacent nodes that aren't neighbours for value == value
                for adj in Self::get_adjacent(graph, node_idx) {
                    if graph[adj].value.is_some() && graph[adj].value == graph[node_idx].value {
                        //eprintln!("Node not valid, adjacent same value: {} {}", graph[node_idx], graph[adj]);
                        return false;
                    }
                }
                true
            }
        }
    }

    fn draw_neighbours(graph: &NodeGraph, node_idx: NodeIndex) {
        for n in &mut graph.neighbors(node_idx) {
            let n = &graph[n];
            println!("\t-> {}", n);
        }
    }

    fn draw_graph(graph: &NodeGraph) {
        for idx in graph.node_indices() {
            println!("{}", graph[idx]);
            Self::draw_neighbours(graph, idx);
        }
    }

    fn draw_grid(graph: &NodeGraph) {
        for y in 0..20 {
            if Self::get_node(graph, &Pos::new(0, y)).is_none() {
                break;
            }
            for x in 0..20 {
                if let Some(node_idx) = Self::get_node(graph, &Pos::new(x, y)) {
                    print!("{} ", graph[node_idx].value.unwrap_or(0));
                }
            }
            println!();
        }
    }
}

impl fmt::Display for Board {
    // TODO add "walls" between positions that don't have edges
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for y in 0..self.height {
            for x in 0..self.width {
                write!(
                    f,
                    "{} ",
                    self.graph[Self::get_node(&self.graph, &Pos::new(x, y))
                        .expect(&format!("No node at index {}", Pos::new(x, y)))]
                    .value
                    .unwrap_or(0)
                )?;
            }
            writeln!(f,)?;
        }
        Ok(())
    }
}

#[test]
fn test_is_solved() {
    assert_eq!(
        Board::is_solved(&Board::new(2, 2, &vec![vec![0, 1], vec![2, 2,],]).graph),
        false
    );
    assert_eq!(
        Board::is_solved(&Board::new(2, 2, &vec![vec![3, 1], vec![3, 3,],]).graph),
        true
    );
}

#[test]
fn test_merge() {}

#[test]
fn test_reachable() {
    let mut graph: NodeGraph = Graph::new_undirected();
    let node_0_0 = graph.add_node(Node::new(None, vec![Pos::new(0, 0)], 1));
    let node_1_0 = graph.add_node(Node::new(None, vec![Pos::new(1, 0)], 1));
    let node_0_1 = graph.add_node(Node::new(Some(2), vec![Pos::new(0, 1)], 1));
    graph.add_edge(node_0_0, node_0_1, 1);
    graph.add_edge(node_0_0, node_1_0, 1);

    let reachable = Board::get_reachable(&graph, node_0_0);
    assert_eq!(reachable.len(), 1);
    assert_eq!(reachable[0], node_0_1);
}
