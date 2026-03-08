"""Tests for PAC verification system."""
import pytest
import networkx as nx

from verification.pac_verify import (
    generate_test_graphs,
    verify_wcc,
    edges_to_csr,
)


@pytest.mark.slow
def test_generate_test_graphs_returns_five():
    graphs = generate_test_graphs(num_vertices=100)
    assert len(graphs) == 5


@pytest.mark.slow
def test_generate_test_graphs_valid_edges():
    graphs = generate_test_graphs(num_vertices=100)
    for name, edges, n in graphs:
        for u, v in edges:
            assert 0 <= u < n, f"{name}: vertex {u} out of range"
            assert 0 <= v < n, f"{name}: vertex {v} out of range"


def test_verify_correct_labels():
    """Correct labels (from NetworkX) should pass all 3 invariants."""
    edges = [(0, 1), (1, 2), (3, 4)]
    n = 5
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)

    # Build reference labels
    labels = {}
    for i, comp in enumerate(nx.connected_components(G)):
        for v in comp:
            labels[v] = min(comp)

    passed, msg = verify_wcc(labels, edges, n)
    assert passed, f"Correct labels should pass: {msg}"


def test_verify_wrong_count():
    """Wrong component count should fail Invariant 1."""
    edges = [(0, 1), (2, 3)]
    n = 4
    # Merge all into one component (wrong: should be 2)
    labels = {0: 0, 1: 0, 2: 0, 3: 0}
    passed, msg = verify_wcc(labels, edges, n)
    assert not passed


def test_verify_cross_component_edge():
    """Edge crossing component boundary should fail Invariant 2."""
    edges = [(0, 1)]
    n = 2
    # Put connected vertices in different components
    labels = {0: 0, 1: 1}
    passed, msg = verify_wcc(labels, edges, n)
    assert not passed


def test_edges_to_csr_roundtrip():
    """CSR should faithfully represent the adjacency."""
    edges = [(0, 1), (1, 2), (0, 2)]
    n = 3
    row_ptr, col_idx = edges_to_csr(edges, n)

    # Reconstruct adjacency and verify
    assert len(row_ptr) == n + 1
    for v in range(n):
        neighbors = set(col_idx[row_ptr[v]:row_ptr[v + 1]])
        expected = set()
        for u, w in edges:
            if u == v:
                expected.add(w)
            if w == v:
                expected.add(u)
        assert neighbors == expected, f"Vertex {v}: got {neighbors}, expected {expected}"
