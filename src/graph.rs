//TODO use depth_first_search maybe? use petgraph::visit::{depth_first_search, DfsEvent};
use petgraph::visit::{IntoEdges, IntoNeighbors, IntoNodeIdentifiers, VisitMap, Visitable, NodeIndexable, EdgeRef};
use std::collections::HashMap;


fn find_bridges_dfs<G>(
    graph: G,
    node: G::NodeId,
    parent: Option<G::NodeId>,
    timer: &mut usize,
    visited: &mut VisitMap<G::NodeId>,
    low: &mut HashMap<usize, usize>,
    tin: &mut HashMap<usize, usize>,
    bridges: &mut Vec<G::EdgeRef>,
) where
    G: IntoEdges + IntoNeighbors + NodeIndexable
{
    visited.visit(node);

    let node_index = graph.to_index(node);
    tin.insert(node_index, *timer);
    low.insert(node_index, *timer);
    *timer += 1;

    for neighbour in graph.neighbors(node) {
        let neighbour_index = graph.to_index(neighbour);
        if Some(neighbour) == parent {
            continue;
        } else if visited.is_visited(&neighbour) {
            low.insert(node_index, std::cmp::min(low[&node_index], tin[&neighbour_index]));
        } else {
            find_bridges_dfs(graph, neighbour, Some(node), timer, visited, low, tin, bridges);
            low.insert(node_index, std::cmp::min(low[&node_index], low[&neighbour_index]));
            if low[&neighbour_index] > tin[&node_index] {
                for edge in graph.edges(node) {
                    // can't get graph.find_edge() because it isn't a trait, make it?
                    match (edge.source(), edge.target()) {
                        (i, j) if i == node && j == neighbour => {
                            bridges.push(edge);
                            break;
                        }
                        (j, i) if i == node && j == neighbour => {
                            bridges.push(edge);
                            break;
                        }
                        (_, _) => (),
                    }
                }
            }
        }
    }
}

/// O(V+E) implementation to find bridges in graph
pub fn find_bridges<G>(graph: G) -> Vec<G::EdgeRef>
where
    G: IntoEdges + IntoNeighbors + Visitable + IntoNodeIdentifiers + NodeIndexable
{
    let mut visited = graph.visit_map();
    let mut low = HashMap::new();
    let mut tin = HashMap::new();
    let mut bridges = Vec::new();

    let mut timer = 0;

    for node in graph.node_identifiers() {
        if !visited.is_visited(&node) {
            find_bridges_dfs(
                &graph,
                node,
                None,
                &mut timer,
                &mut visited,
                &mut low,
                &mut tin,
                &mut bridges,
            );
        }
    }

    bridges
}

//TODO make a function that gets the components that would be greated if bridge was cut

#[test]
fn test_find_bridges() {
    use petgraph::graph::Graph;
    use petgraph::Undirected;
    let mut graph : Graph<(), (), Undirected> = Graph::new_undirected();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let d = graph.add_node(());
    let e = graph.add_node(());
    let f = graph.add_node(());
    let g = graph.add_node(());
    let h = graph.add_node(());
    let i = graph.add_node(());

    graph.extend_with_edges(&[
        (a, b),
        (b, c),
        (c, a),
        (c, d), // bridge
        (d, e),
        (e, f),
        (f, g),
        (g, d),
        (b, h),
        (c, h),
        (h, i), // bridge
    ]);

    let bridges = find_bridges(&graph);
    assert_eq!(bridges.len(), 2);
}
