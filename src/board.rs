use petgraph::graph::{Graph, NodeIndex};
use petgraph::Undirected;
use std::fmt;

#[derive(Debug, PartialEq)]
pub struct Board {
    size: usize, //TODO allow now square boards
    grid: Vec<Vec<u8>>,
}

#[derive(Debug, PartialEq)]
struct Node {
    value: Option<u8>,
    coord: Vec<(usize, usize)>,
    size: u8,
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}/{}: ", self.size, self.value.unwrap_or(0))?;
        for (x, y) in &self.coord {
            write!(f, "({},{}) ", y, x)?;
        }
        Ok(())
    }
}

impl Node {
    fn is_complete(&self) -> bool {
        self.value.unwrap_or(0) == self.size
    }

    fn get_index(&self, graph: &Graph<Node, u8, Undirected>) -> NodeIndex {
        graph
            .node_indices()
            .find(|idx| graph[*idx] == *self)
            .unwrap()
    }

    /// Note: calls remove_node so graph[len] will be invalidated and will be in other_idx
    pub fn merge_node(
        graph: &mut Graph<Node, u8, Undirected>,
        index: NodeIndex,
        other_idx: NodeIndex,
    ) -> Result<(), &'static str> {
        if graph[index].value.is_none() {
            Err("merge_node called on node with no value")
        } else if graph[other_idx].size + graph[index].size > graph[index].value.unwrap() {
            Err("merge_node would overflow value")
        } else {
            println!("Merging {} and {}", graph[index], graph[other_idx]);
            let mut coord = graph[other_idx].coord.clone();
            graph[index].coord.append(&mut coord);
            graph[index].size += graph[other_idx].size;

            let mut neighbors = graph.neighbors(other_idx).detach();
            while let Some(neighbour_idx) = neighbors.next_node(&graph) {
                if neighbour_idx != index {
                    graph.update_edge(index, neighbour_idx, 1);
                }
            }
            println!("Merged {}", graph[index]);
            graph.remove_node(other_idx);
            Ok(())
        }
    }
}

#[test]
fn test_merge_node() {
    let mut graph : Graph<Node, u8, Undirected> = Graph::new_undirected();
    let n1 = graph.add_node(Node { value : None, coord: vec![(0, 2)], size: 1});
    let n2 = graph.add_node(Node { value : Some(4), coord: vec![(1, 2)], size: 2});
    let n3 = graph.add_node(Node { value: Some(4), coord: vec![(2,3)], size: 1});
    let n4 = graph.add_node(Node { value: None, coord: vec![(0, 0)], size: 1});

    assert_eq!(Node::merge_node(&mut graph, n3, n4), Ok(()));
    assert_eq!(graph[n3], Node{value: Some(4), coord: vec![(2,3), (0, 0)], size: 2});
    assert_eq!(Node::merge_node(&mut graph, n2, n3), Ok(()));
    assert_eq!(graph[n2], Node{value: Some(4), coord: vec![(1, 2), (2, 3), (0,0)], size: 4});
    assert_eq!(Node::merge_node(&mut graph, n2, n1), Err("merge_node would overflow value"))
}

impl Board {
    pub fn new(size: usize, grid: &Vec<Vec<u8>>) -> Self {
        Board {
            size,
            grid: grid.clone(),
        }
    }

    pub fn solve(&self) -> Vec<Board> {
        let graph = Self::to_graph(&self.grid).expect("Failed to convert to graph");
        Self::draw_graph(&graph);

        Vec::new()
    }

    fn to_graph(grid: &Vec<Vec<u8>>) -> Result<Graph<Node, u8, Undirected>, &'static str> {
        let mut graph: Graph<Node, u8, Undirected> = Graph::new_undirected();
        let mut y: usize = 0;
        let mut last_row: Vec<NodeIndex> = Vec::new();
        for row in grid {
            let mut x: usize = 0;
            let mut this_row: Vec<NodeIndex> = Vec::new();
            let mut last_node: Option<NodeIndex> = None;
            for value in row {
                let mut node = match value {
                    0 => graph.add_node(Node {
                        value: None,
                        coord: vec![(x, y)],
                        size: 1,
                    }),
                    n => graph.add_node(Node {
                        value: Some(*n),
                        coord: vec![(x, y)],
                        size: 1,
                    }),
                };
                if let Some(last_node) = last_node {
                    match (graph[last_node].value, graph[node].value) {
                        (Some(n), Some(m)) if n == m => {
                            Node::merge_node(&mut graph, last_node, node)?;
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
                            Node::merge_node(&mut graph, last_row[x], node)?;
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

    fn draw_graph(graph: &Graph<Node, u8, Undirected>) {
        let node_idxs = graph.node_indices();
        for idx in node_idxs {
            let node = &graph[idx];
            println!("{}", node);

            let neighbours = &mut graph.neighbors(idx);
            for n in neighbours {
                let n = &graph[n];
                println!("\t-> {}", n);
            }
        }
    }
}

#[test]
fn test_to_graph() {
    let mut expected : Graph<Node, u8, Undirected> = Graph::new_undirected();
    let n1 = expected.add_node(Node { value: None, coord: vec![(0,0)], size: 1});
    let n2 = expected.add_node(Node { value: Some(4), coord: vec![(0, 1), (1,1), (1,0)], size: 3});
    expected.add_edge(n1, n2, 1);
    Board::draw_graph(&Board::to_graph(&Board::new(2, &vec![vec![0, 4], vec![4, 4]]).grid).unwrap());

    //assert_eq!(Board::to_graph(&Board::new(2, &vec![vec![0, 4], vec![4, 4]]).grid).unwrap().into_nodes_edges(), expected.into_nodes_edges());
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
