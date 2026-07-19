"""
circuits.py

Trotterized quantum-circuit construction for the TFIM: edge coloring of the
interaction graph (so ZZ rotations on non-overlapping edges can run in
parallel/same layer) and the single Trotter layer (Rx/2 - Rzz - Rx/2).
"""
import networkx as nx
from qiskit import QuantumCircuit


def edge_coloring(graph: nx.Graph):
    """Color edges so that no two adjacent edges share the same color.

    Returns a list of edge lists, one per color class.
    """
    line_graph = nx.line_graph(graph)
    edge_colors = nx.coloring.greedy_color(line_graph)
    color_groups = {}
    for edge, color in edge_colors.items():
        color_groups.setdefault(color, []).append(edge)
    return list(color_groups.values())


def build_chain_color_edges(N: int):
    """Build the periodic 1D chain graph for N sites and its edge coloring."""
    graph = nx.path_graph(N)
    graph.add_edge(N - 1, 0)
    return edge_coloring(graph)


def build_single_layer_circuit(N, color_edges, theta_x, theta_zz, mirror=True):
    """One (symmetrized) Trotter layer: Rx(theta_x/2) - Rzz(theta_zz) - Rx(theta_x/2).

    If mirror=False, applies a single Rx(theta_x) instead of splitting it
    (asymmetric/first-order Trotter step).
    """
    layer = QuantumCircuit(N)
    for i in range(N):
        layer.rx(theta_x / 2 if mirror else theta_x, i)
    for edge_list in color_edges:
        for edge in edge_list:
            layer.rzz(theta_zz, edge[0], edge[1])
    for i in range(N):
        layer.rx(theta_x / 2 if mirror else theta_x, i)
    return layer
