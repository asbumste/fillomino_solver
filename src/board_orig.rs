use petgraph::algo::dijkstra;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Undirected;
use std::collections::hash_map::HashMap;
use std::fmt;
use std::io::{self, Read};

#[derive(Debug, PartialEq)]
pub struct Board {
    size: usize, //TODO allow non square boards
    grid: Vec<Vec<u8>>,
}

#[derive(Debug, PartialEq, Clone)]
struct Node {
    value: Option<u8>,
    coord: Vec<(usize, usize)>,
    size: u8,
    valid: bool,
}

fn abs_difference(x: usize, y: usize) -> usize {
    if x < y {
        y - x
    } else {
        x - y
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}/{}: ", self.size, self.value.unwrap_or(0))?;
        for (y, x) in &self.coord {
            write!(f, "({},{}) ", y, x)?;
        }
        Ok(())
    }
}

type NodeGraph = Graph<Node, usize, Undirected>;

impl Node {
    pub fn new(value: Option<u8>, coord: Vec<(usize, usize)>, size: u8) -> Self {
        Node {
            value,
            coord,
            size,
            valid: true,
        }
    }
    fn is_complete(&self) -> bool {
        self.value.unwrap_or(0) == self.size
    }

    fn get_index(&self, graph: &NodeGraph) -> NodeIndex {
        graph
            .node_indices()
            .find(|idx| graph[*idx] == *self)
            .unwrap()
    }

    /// Note: calls remove_node so graph[len] will be invalidated and will be in other_idx
    pub fn merge(
        graph: &mut NodeGraph,
        index: NodeIndex,
        other_idx: NodeIndex,
    ) -> Result<(), &'static str> {
        if graph[index].value.is_some()
            && graph[other_idx].size + graph[index].size > graph[index].value.unwrap()
        {
            eprintln!("merge would overflow value: {} {}", graph[index], graph[other_idx]);
            Err("merge would overflow value")
        } else {
            if graph[index].value.is_none() {
                graph[index].value = graph[other_idx].value;
            }

            // Set the value, but don't merge nodes that aren't adjacent
            if !Node::adjacent(graph, index, other_idx) {
                // Guarantees both nodes are set, by above block
                graph[other_idx].value = graph[index].value;
                return Ok(());
            }

            let mut coord = graph[other_idx].coord.clone();
            graph[index].coord.append(&mut coord);
            graph[index].size += graph[other_idx].size;

            let value = graph[index].value;

            // Add edges from other_idx
            let mut neighbors = graph.neighbors(other_idx).detach();
            while let Some(neighbour_idx) = neighbors.next_node(&graph) {
                if neighbour_idx != index {
                    graph.update_edge(index, neighbour_idx, 1);
                }
            }

            // TODO for pulse, merge neighbours that are value
            let mut neighbors = graph.neighbors(index).detach();
            while let Some(neighbour_idx) = neighbors.next_node(&graph) {
                let (merge, remove_edge) = match (graph[neighbour_idx].value, graph[neighbour_idx].size) {
                    (_, _) if !graph[neighbour_idx].valid => (false, false),
                    (_, _) if graph[index].coord.contains(&graph[neighbour_idx].coord[0]) => (false, true),
                    (Some(v), n) if v == value.unwrap_or(0) && n + graph[index].size > v => {
                        eprintln!("Value neighrbours itself {} {}", graph[index], graph[neighbour_idx]);
                        return Err("Value neighbours itself");
                    }
                    (Some(v), n) if v == value.unwrap_or(v) && n + graph[index].size <= v => (true, false),
                    (None, n) if value.is_none() || n + graph[index].size <= value.unwrap() => {
                        (false, false)
                    }
                    (_, _) => (false, true),
                };
                if merge {
                    Self::merge(graph, index, neighbour_idx)?;
                } else if remove_edge && graph.find_edge(index, neighbour_idx).is_some() {
                    graph.remove_edge(graph.find_edge(index, neighbour_idx).unwrap());
                }
            }

            // Remove invalidates, so keep an empty node
            //graph.remove_node(other_idx);
            graph[other_idx].valid = false;
            graph[other_idx].size = 0;
            graph[other_idx].value = None;
            graph[other_idx].coord.clear();
            let mut neighbors = graph.neighbors(other_idx).detach();
            while let Some(neighbour_idx) = neighbors.next_node(&graph) {
                graph.remove_edge(graph.find_edge(other_idx, neighbour_idx).unwrap());
            }
            Ok(())
        }
    }

    pub fn adjacent(graph: &NodeGraph, node1: NodeIndex, node2: NodeIndex) -> bool {
        for (x1, y1) in &graph[node1].coord {
            for (x2, y2) in &graph[node2].coord {
                match (abs_difference(*x1, *x2), abs_difference(*y1, *y2)) {
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

#[test]
fn test_merge() {
    let mut graph: NodeGraph = Graph::new_undirected();
    let n1 = graph.add_node(Node::new(None, vec![(0, 2)], 1));
    let n2 = graph.add_node(Node::new(Some(4), vec![(1, 2)], 2));
    let n3 = graph.add_node(Node::new(Some(4), vec![(2, 3)], 1));
    let n4 = graph.add_node(Node::new(None, vec![(0, 0)], 1));

    assert_eq!(Node::merge(&mut graph, n3, n4), Ok(()));
    assert_eq!(graph[n3], Node::new(Some(4), vec![(2, 3), (0, 0)], 2),);
    assert_eq!(Node::merge(&mut graph, n2, n3), Ok(()));
    assert_eq!(
        graph[n2],
        Node::new(Some(4), vec![(1, 2), (2, 3), (0, 0)], 4),
    );
    assert_eq!(
        Node::merge(&mut graph, n2, n1),
        Err("merge would overflow value")
    )
}

impl Board {
    pub fn new(size: usize, grid: &Vec<Vec<u8>>) -> Self {
        Board {
            size,
            grid: grid.clone(),
        }
    }

    fn from_graph(size: usize, graph: &NodeGraph) -> Self {
        let mut grid: Vec<Vec<u8>> = Vec::new();
        for row in 0..size {
            let row: usize = row.into();
            grid.push(Vec::new());
            for col in 0..size {
                let col: usize = col.into();
                let node_idx = Board::get_node(graph, row, col);
                grid[row].push(graph[node_idx].value.unwrap_or(0));
            }
        }
        Self::new(size, &grid)
    }

    // Merge nodes with only one neighbour
    fn lonely_nodes(graph: &mut NodeGraph) -> Result<bool, &'static str> {
        let mut updated = true;
        let mut updated_once = false;
        while updated {
            updated = false;
            for node_idx in graph.node_indices() {
                if graph[node_idx].valid && graph.neighbors(node_idx).count() == 1 {
                    let neighbour_idx = graph
                        .neighbors(node_idx)
                        .detach()
                        .next_node(&graph)
                        .unwrap();

                    if let Err(e) = Node::merge(graph, node_idx, neighbour_idx) {
                        eprintln!("merge failed in lonely nodes {}", e);
                        return Err(e);
                    }
                    updated = true;
                    // Break after update to avoid invalidated indices
                    break;
                } else {
                }
            }
            if updated {
                updated_once = true
            }
        }
        Ok(updated_once)
    }

    // Merge nodes with only one "valued" node in range
    fn value_search(graph: &mut NodeGraph) -> Result<bool, &'static str> {
        let mut updated = true;
        let mut updated_once = false;
        while updated {
            updated = false;
            for node_idx in graph.node_indices() {
                if graph[node_idx].valid && graph[node_idx].value.is_none() {
                    // Cost Some(n) -> None = 50
                    // Cost Some(n)
                    // and 1 for None -> None,
                    // then subtract 1000 from cost calculation,
                    // So there will only ever be 1 edge with Some(N)
                    let reachable: HashMap<NodeIndex, usize> =
                        dijkstra(&*graph, node_idx, None, |e| {
                            match (graph[e.source()].value, graph[e.target()].value) {
                                (Some(n), Some(m)) if m == n => 0,
                                (Some(_), Some(_)) => {
                                    eprintln!(
                                        "Valued nodes with edge: {} & {}",
                                        graph[e.source()],
                                        graph[e.target()]
                                    );
                                    1000
                                }
                                (Some(_), None) => 50,
                                (None, Some(_)) => 50,
                                (None, None) => 1,
                            }
                        });
                    let reachable: HashMap<&NodeIndex, &usize> = reachable
                        .iter()
                        .filter(|(idx, cost)| {
                            node_idx != **idx
                                && graph[**idx].value.is_some()
                                && **cost
                                    + (graph[**idx].size as usize)
                                    + (graph[node_idx].size as usize)
                                    <= (graph[**idx].value.unwrap() as usize) + 50
                        })
                        .collect();
                    println!("{} -> {}", graph[node_idx], reachable.len());
                    for (idx, cost) in &reachable {
                        println!("\t{} -> {}", graph[**idx], *cost - 50);
                    }
                    match reachable.len() {
                        0 => return Err("Unreachable node"),
                        1 => {
                            let other_idx: Vec<&&NodeIndex> = reachable.keys().collect();
                            let other_idx = other_idx[0];

                            //check if nodes are neighbours and set value if not
                            //TODO could handle non-contiguous nodes later?
                            //if Node::adjacent(graph, node_idx, **other_idx) {
                            Node::merge(graph, node_idx, **other_idx)?;
                            /*} else {
                                graph[node_idx].value = graph[**other_idx].value;
                            }*/

                            updated = true;
                            break;
                        }
                        _ => {
                            // Check if all reachable are same # and within value
                            let mut size = graph[node_idx].size;
                            let mut value = None;
                            let mut valid = true;
                            for idx in reachable.keys() {
                                size += graph[**idx].size;
                                if value.is_none() {
                                    value = graph[**idx].value;
                                } else if graph[**idx].value.is_some()
                                    && graph[**idx].value != value
                                {
                                    valid = false;
                                    break;
                                }
                            }
                            if valid {
                                /* TODO need to calulate distance in size
                                if size < value.unwrap() {
                                    return Err("Unreachable node");
                                } else*/ if size <= value.unwrap() {
                                    // TODO merge invalidates indices, this might go wrong?
                                    for idx in reachable.keys() {
                                        Node::merge(graph, node_idx, **idx)?;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if updated {
                updated_once = true;
            }
        }
        Ok(updated_once)
    }

    // Search for nodes with only value nodes within range
    fn within_search(graph: &mut NodeGraph) -> Result<bool, &'static str> {
        let mut updated_once = false;
        let mut updated = true;
        while updated {
            updated = false;
            for node_idx in graph.node_indices() {
                if graph[node_idx].valid
                    && graph[node_idx].value.is_some()
                    && graph[node_idx].value.unwrap() != graph[node_idx].size
                {
                    let reachable: HashMap<NodeIndex, usize> =
                        dijkstra(&*graph, node_idx, None, |e| {
                            match (graph[e.source()].value, graph[e.target()].value) {
                                (Some(_), Some(_)) => {
                                    eprintln!(
                                        "Valued nodes with edge: {} {}",
                                        graph[e.source()],
                                        graph[e.target()]
                                    );
                                    1000
                                }
                                (Some(_), None) => 50,
                                (None, Some(_)) => 50,
                                (None, None) => 1,
                            }
                        });

                    let reachable: HashMap<&NodeIndex, &usize> = reachable
                        .iter()
                        .filter(|(idx, cost)| {
                            node_idx != **idx
                                && ((graph[**idx].value.is_none()
                                    && **cost
                                        + (graph[**idx].size as usize)
                                        + (graph[node_idx].size as usize)
                                        <= (graph[node_idx].value.unwrap() as usize) + 50)
                                    || (graph[**idx].value == graph[node_idx].value
                                        && **cost
                                            + (graph[**idx].size as usize)
                                            + (graph[node_idx].size as usize)
                                            <= (graph[node_idx].value.unwrap() as usize) + 100))
                        })
                        .collect();
                    if reachable.len() == 0 {
                        return Err("Unreachable node");
                    } else {
                        let mut size = graph[node_idx].size;
                        for idx in reachable.keys() {
                            size += graph[**idx].size;
                        }
                        if size > graph[node_idx].value.unwrap() {
                            continue;
                        } else if size == graph[node_idx].value.unwrap() {
                            // TODO merge invalidates indices, this might go wrong?
                            for idx in reachable.keys() {
                                Node::merge(graph, node_idx, **idx)?;
                            }
                            updated = true;
                            break;
                        } else {
                            return Err("Unreachable node");
                        }
                    }
                }
            }
            if updated {
                updated_once = true;
            }
        }
        Ok(updated_once)
    }

    /* Find chokepoints that all possible solutions go through
    fn chokepoint(graph: &mut NodeGraph) -> Result<bool, &'static str> {
        let mut updated_once = false;
        let mut updated = true;
        while updated {
            updated = false;
            for node_idx in graph.node_indices() {
                if graph[node_idx].valid
                    && graph[node_idx].value.is_some()
                    && graph[node_idx].value.unwrap() != graph[node_idx].size
                {

                }
            }
        }
    }
    */

    fn validate_node(graph: &NodeGraph, node_idx: NodeIndex) -> bool {
        if !graph[node_idx].valid {
            true
        } else if graph[node_idx].value.is_none() {
            // Cost Some(n) -> None = 50
            // Cost Some(n)
            // and 1 for None -> None,
            // then subtract 1000 from cost calculation,
            // So there will only ever be 1 edge with Some(N)
            let reachable: HashMap<NodeIndex, usize> = dijkstra(&*graph, node_idx, None, |e| {
                match (graph[e.source()].value, graph[e.target()].value) {
                    (Some(_), Some(_)) => {
                        eprintln!(
                            "Valued nodes with edge: {} {}",
                            graph[e.source()],
                            graph[e.target()]
                        );
                        1000
                    }
                    (Some(_), None) => 50,
                    (None, Some(_)) => 50,
                    (None, None) => 1,
                }
            });
            let reachable: HashMap<&NodeIndex, &usize> = reachable
                .iter()
                .filter(|(idx, cost)| {
                    node_idx != **idx
                        && graph[**idx].value.is_some()
                        && **cost
                            + (graph[**idx].size as usize)
                            + (graph[node_idx].size as usize)
                            <= (graph[**idx].value.unwrap() as usize) + 50
                })
                .collect();

            // TODO are there other validations to do for an unassigned node?
            if reachable.is_empty() {
                false
            } else {
                true
            }
        } else {
            let reachable: HashMap<NodeIndex, usize> =
                dijkstra(&*graph, node_idx, None, |e| {
                    match (graph[e.source()].value, graph[e.target()].value) {
                        (Some(_), Some(_)) => 1, // Shouldn't happen, will fail in later check
                        (Some(_), None) => 50,
                        (None, Some(_)) => 50,
                        (None, None) => 1,
                    }
                });

            let reachable: HashMap<&NodeIndex, &usize> = reachable
                .iter()
                .filter(|(idx, cost)| {
                    node_idx != **idx
                        && ((graph[**idx].value.is_none()
                            && **cost
                                + (graph[**idx].size as usize)
                                + (graph[node_idx].size as usize)
                                <= (graph[node_idx].value.unwrap() as usize) + 50)
                            || (graph[**idx].value == graph[node_idx].value
                                && **cost
                                    + (graph[**idx].size as usize)
                                    + (graph[node_idx].size as usize)
                                    <= (graph[node_idx].value.unwrap() as usize) + 100))
                })
                .collect();
                let mut size = graph[node_idx].size;
                for idx in reachable.keys() {
                    if graph[**idx].value.is_some() {
                        return false;
                    }
                    size += graph[**idx].size;
                }
                // TODO need to check that no two nodes with the same value are adjacent
                if size < graph[node_idx].value.unwrap() {
                    false
                } else {
                    true
                }
        }
    }

    fn validate(graph: &NodeGraph) -> bool {
        for node_idx in graph.node_indices() {
            if !Board::validate_node(graph, node_idx) {
                return false;
            }
        }
        true
    }

    // Guess a value and try to solve from there
    fn pulse(size: usize, graph: &NodeGraph) -> Result<Vec<Board>, &'static str> {
        let mut results: Vec<Board> = Vec::new();
        // Basically do value_search, but guess each reachable value and see if the graph is solvable
        for node_idx in graph.node_indices() {
            if graph[node_idx].valid && graph[node_idx].value.is_none() {
                // Cost Some(n) -> None = 50
                // Cost Some(n)
                // and 1 for None -> None,
                // then subtract 1000 from cost calculation,
                // So there will only ever be 1 edge with Some(N)
                let reachable: HashMap<NodeIndex, usize> = dijkstra(&*graph, node_idx, None, |e| {
                    match (graph[e.source()].value, graph[e.target()].value) {
                        (Some(_), Some(_)) => {
                            eprintln!(
                                "Valued nodes with edge: {} {}",
                                graph[e.source()],
                                graph[e.target()]
                            );
                            1
                        }
                        (Some(_), None) => 50,
                        (None, Some(_)) => 50,
                        (None, None) => 1,
                    }
                });
                let reachable: HashMap<&NodeIndex, &usize> = reachable
                    .iter()
                    .filter(|(idx, cost)| {
                        node_idx != **idx
                            && graph[**idx].value.is_some()
                            && **cost
                                + (graph[**idx].size as usize)
                                + (graph[node_idx].size as usize)
                                <= (graph[**idx].value.unwrap() as usize) + 50
                    })
                    .collect();

                for idx in reachable.keys() {
                    let mut g : NodeGraph = (*graph).clone();
                    Node::merge(&mut g, node_idx, **idx)?;
                    if !Board::validate(&g) {
                        continue;
                    }
                    println!("After pulse");
                    let mut buffer = String::new();
                    io::stdin().read_line(&mut buffer).expect("Failed to read stdin");

                    Self::draw_graph(size, &g);
                    if let Ok(mut result) = Board::solve_graph(size, &mut g) {
                        results.append(&mut result);
                    }
                }
            }
        }
        Ok(results)
    }

    fn solve_graph(size: usize, graph: &mut NodeGraph) -> Result<Vec<Board>, &'static str> {
        let mut results: Vec<Board> = Vec::new();
        let mut updated = true;
        while graph.edge_count() != 0 && updated {
            updated = Self::lonely_nodes(graph)?;
            //println!("After Lonely");
            //Self::draw_graph(size, &graph);
            //println!("Value Search");
            updated = updated || Self::value_search(graph)?;
            //println!("After Value");
            //Self::draw_graph(size, &graph);
            //println!("Within Search");
            updated = updated || Self::within_search(graph)?;
            //println!("After Within");
            //Self::draw_graph(size, &graph);
            if graph.edge_count() != 0 && !updated {
                println!("Need pulse");
                Self::draw_graph(size, &graph);
                break;
                //return Self::pulse(size, graph);
                //results.append(&mut Self::pulse(size, graph)?);
            }
        }
        if graph.edge_count() == 0 {
            //TODO need to validate results, edges == 0 does not mean value == size
            println!("Solved");
            Self::draw_graph(size, &graph);
            results.push(Self::from_graph(size, graph));
        }

        Ok(results)
    }

    pub fn solve(&self) -> Vec<Board> {
        let mut graph = Self::to_graph(&self.grid).expect("Failed to convert to graph");

        Board::solve_graph(self.size, &mut graph).expect("Failed to get any results")
    }

    fn to_graph(grid: &Vec<Vec<u8>>) -> Result<NodeGraph, &'static str> {
        let mut graph: NodeGraph = Graph::new_undirected();
        let mut y: usize = 0;
        let mut last_row: Vec<NodeIndex> = Vec::new();
        for row in grid {
            let mut x: usize = 0;
            let mut this_row: Vec<NodeIndex> = Vec::new();
            let mut last_node: Option<NodeIndex> = None;
            for value in row {
                let mut node = match value {
                    0 => graph.add_node(Node::new(None, vec![(y, x)], 1)),
                    n => graph.add_node(Node::new(Some(*n), vec![(y, x)], 1)),
                };
                if let Some(last_node) = last_node {
                    match (graph[last_node].value, graph[node].value) {
                        (Some(n), Some(m)) if n == m => {
                            Node::merge(&mut graph, last_node, node)?;
                            if x > 0 && this_row[x - 1] == node {
                                this_row[x - 1] = last_node;
                            }
                            node = last_node;
                        }
                        (Some(_), Some(_)) => (),
                        _ => {
                            if !graph[node].is_complete() && !graph[last_node].is_complete() {
                                graph.add_edge(last_node, node, 1);
                            }
                        }
                    }
                }
                if !last_row.is_empty() {
                    match (graph[node].value, graph[last_row[x]].value) {
                        (Some(n), Some(m)) if n == m => {
                            Node::merge(&mut graph, last_row[x], node)?;
                            node = last_row[x];
                            ()
                        }
                        (Some(_), Some(_)) => (),
                        _ => {
                            if !graph[node].is_complete() && !graph[last_row[x]].is_complete() {
                                graph.add_edge(last_row[x], node, 1);
                            }
                            ()
                        }
                    }
                }
                last_node = Some(node);
                this_row.push(node);
                x += 1;
            }
            last_row = this_row;
            y += 1;
        }
        Ok(graph)
    }

    fn draw_graph(size: usize, graph: &NodeGraph) {
        Self::draw_graph_as_grid(size, graph);
        let node_idxs = graph.node_indices();
        for idx in node_idxs {
            if graph[idx].valid == false || graph[idx].is_complete() {
                continue;
            }
            let node = &graph[idx];
            let neighbours = &mut graph.neighbors(idx);
            println!("{}", node);
            for n in neighbours {
                let n = &graph[n];
                println!("\t-> {}", n);
            }
        }
    }

    fn get_node(graph: &NodeGraph, row: usize, col: usize) -> NodeIndex {
        graph
            .node_indices()
            .find(|i| {
                graph[*i]
                    .coord
                    .iter()
                    .find(|(y, x)| *y == row && *x == col)
                    .is_some()
            })
            .expect(&format!("Failed to get ({},{})", row, col))
    }

    fn draw_graph_as_grid(size: usize, graph: &NodeGraph) {
        println!("{}", Self::from_graph(size, graph));
    }
}

#[test]
fn test_to_graph() {
    let mut expected: NodeGraph = Graph::new_undirected();
    let n1 = expected.add_node(Node::new(None, vec![(0, 0)], 1));
    let n2 = expected.add_node(Node::new(Some(4), vec![(0, 1), (1, 1), (1, 0)], 3));
    expected.add_edge(n1, n2, 1);
    /* Board::draw_graph(
        &Board::to_graph(&Board::new(2, &vec![vec![0, 4], vec![4, 4]]).grid).unwrap(),
    );

    //assert_eq!(Board::to_graph(&Board::new(2, &vec![vec![0, 4], vec![4, 4]]).grid).unwrap().into_nodes_edges(), expected.into_nodes_edges());
    */
}

//TODO use rstest
#[test]
fn test_to_validate() {
    assert_eq!(Board::validate(&Board::to_graph(&Board::new(2, &vec![vec![0, 4], vec![4, 4]]).grid).unwrap()), true);
    assert_eq!(Board::validate(&Board::to_graph(&Board::new(2, &vec![vec![1, 4], vec![4, 4]]).grid).unwrap()), false);

    let mut g: NodeGraph = Graph::new_undirected();
    let n1 = g.add_node(Node::new(Some(2), vec![(0, 0), (0, 1)], 2));
    let n2 = g.add_node(Node::new(Some(2), vec![(1, 0), (1, 1)], 2));
    g.add_edge(n1, n2, 1);

    assert_eq!(Board::validate(&g), false);
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for row in &self.grid {
            for i in row {
                write!(f, "{} ", i)?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}
