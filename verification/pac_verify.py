"""
PAC (Probably Approximately Correct) Reasoning Verification System.

Implements DoubleAI's verification methodology:
- Input Generator: Creates adversarial test graphs
- Algorithmic Verifier: Mathematically pure reference implementation
- Three invariants: component count, edge consistency, cross-component distinctness
"""
import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Any


def generate_test_graphs(num_vertices: int = 10000) -> List[Tuple[str, List[Tuple[int, int]], int]]:
    """
    Input Generator: 5 adversarial graphs designed to break parallelization strategies.
    
    Returns:
        List of (graph_type, edge_list, num_vertices) tuples
    """
    graphs = []

    # 1-2: RMAT power-law (skewed degrees -> race conditions at hubs)
    for seed in [42, 137]:
        edges = generate_rmat(num_vertices, num_vertices * 10, seed)
        graphs.append(("rmat", edges, num_vertices))

    # 3-4: SBM planted communities (cross-partition merging)
    for n_comm in [5, 50]:
        sizes = [num_vertices // n_comm] * n_comm
        p_matrix = [[0.1 if i == j else 0.001 for j in range(n_comm)]
                     for i in range(n_comm)]
        G = nx.stochastic_block_model(sizes, p_matrix, seed=n_comm)
        graphs.append(("sbm", list(G.edges()), sum(sizes)))

    # 5: Sparse Erdos-Renyi (many isolates, tiny components)
    G = nx.erdos_renyi_graph(num_vertices, 0.0005, seed=99)
    graphs.append(("er_sparse", list(G.edges()), num_vertices))

    return graphs


def verify_wcc(kernel_labels: Dict[int, int], edges: List[Tuple[int, int]], num_vertices: int) -> Tuple[bool, str]:
    """
    Algorithmic Verifier: 3 mathematical invariants for WCC correctness.
    
    Args:
        kernel_labels: Dictionary mapping vertex_id -> component_label from kernel
        edges: List of (u, v) edges
        num_vertices: Total number of vertices
    
    Returns:
        (passed, message) tuple
    """
    # Build reference graph and compute ground truth
    G = nx.Graph()
    G.add_nodes_from(range(num_vertices))
    G.add_edges_from(edges)
    ref_components = list(nx.connected_components(G))

    # Invariant 1: Component count matches reference exactly
    kernel_component_count = len(set(kernel_labels.get(v, v) for v in range(num_vertices)))
    if kernel_component_count != len(ref_components):
        return False, f"Count mismatch: kernel={kernel_component_count} vs ref={len(ref_components)}"

    # Invariant 2: Every edge connects vertices with the same label
    for u, v in edges:
        label_u = kernel_labels.get(u, u)
        label_v = kernel_labels.get(v, v)
        if label_u != label_v:
            return False, f"Edge ({u},{v}) crosses components: {label_u} != {label_v}"

    # Invariant 3: Vertices in different reference components have different labels
    label_to_components = {}
    for comp_id, component in enumerate(ref_components):
        for v in component:
            label = kernel_labels.get(v, v)
            if label in label_to_components:
                if label_to_components[label] != comp_id:
                    return False, f"Label {label} spans multiple components ({label_to_components[label]}, {comp_id})"
            else:
                label_to_components[label] = comp_id

    return True, f"Verified: {len(ref_components)} components, all invariants hold"


def generate_rmat(n: int, m: int, seed: int) -> List[Tuple[int, int]]:
    """
    RMAT Kronecker graph generator (Graph500 standard parameters).
    
    Generates power-law graphs that expose race conditions at hub nodes.
    
    Args:
        n: Number of vertices
        m: Number of edges
        seed: Random seed
    
    Returns:
        List of (u, v) edges
    """
    rng = np.random.default_rng(seed)
    a, b, c = 0.57, 0.19, 0.19  # Graph500 standard parameters
    edges = set()
    
    for _ in range(m):
        u, v = 0, 0
        step = n // 2
        while step >= 1:
            r = rng.random()
            if r < a:
                pass  # Stay in same quadrant
            elif r < a + b:
                v += step  # Move to right quadrant
            elif r < a + b + c:
                u += step  # Move to bottom quadrant
            else:
                u += step
                v += step  # Move to diagonal quadrant
            step //= 2
        
        # Avoid self-loops and ensure valid vertex range
        if u != v and 0 <= u < n and 0 <= v < n:
            edges.add((min(u, v), max(u, v)))
    
    return list(edges)


def edges_to_csr(edges: List[Tuple[int, int]], num_vertices: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert edge list to CSR format for CUDA kernels.
    
    Args:
        edges: List of (u, v) edges
        num_vertices: Number of vertices
    
    Returns:
        (row_ptr, col_idx) CSR arrays
    """
    # Build adjacency list
    adj = [[] for _ in range(num_vertices)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)  # Undirected graph
    
    # Convert to CSR
    row_ptr = np.zeros(num_vertices + 1, dtype=np.int32)
    col_idx = []
    
    for i, neighbors in enumerate(adj):
        row_ptr[i + 1] = row_ptr[i] + len(neighbors)
        col_idx.extend(sorted(neighbors))  # Sort for consistency
    
    return row_ptr, np.array(col_idx, dtype=np.int32)


def run_kernel_verification(kernel_lib_path: str, edges: List[Tuple[int, int]], num_vertices: int) -> Dict[int, int]:
    """
    Run compiled CUDA kernel and return component labels.

    Expects a C-exported symbol with the signature:
    `wcc_kernel(const int* row_ptr, const int* col_idx, int num_vertices, int* labels)`.
    
    Args:
        kernel_lib_path: Path to compiled .so file
        edges: Edge list
        num_vertices: Number of vertices
    
    Returns:
        Dictionary mapping vertex_id -> component_label
    """
    import ctypes
    import numpy as np
    
    # Convert to CSR
    row_ptr, col_idx = edges_to_csr(edges, num_vertices)
    
    # Load library and call kernel
    lib = ctypes.CDLL(kernel_lib_path)
    
    # Assuming kernel exports: wcc_kernel(row_ptr, col_idx, num_vertices, labels)
    # This needs to match the actual kernel's C interface
    try:
        # Prepare arrays
        row_ptr_c = row_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        col_idx_c = col_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        
        # Output array
        labels = np.arange(num_vertices, dtype=np.int32)
        labels_c = labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        
        # Call kernel function (adjust name as needed)
        lib.wcc_kernel(row_ptr_c, col_idx_c, ctypes.c_int32(num_vertices), labels_c)
        
        return {i: int(labels[i]) for i in range(num_vertices)}
    
    except Exception as e:
        raise RuntimeError(
            f"Kernel FFI verification failed for '{kernel_lib_path}': {e}"
        ) from e


if __name__ == "__main__":
    # Test PAC verification system
    print("Testing PAC verification system...")
    
    # Generate test graphs
    graphs = generate_test_graphs(1000)
    print(f"Generated {len(graphs)} test graphs")
    
    for graph_type, edges, n_verts in graphs:
        print(f"\nTesting {graph_type} graph: {n_verts} vertices, {len(edges)} edges")
        
        # Use networkx as reference
        G = nx.Graph()
        G.add_nodes_from(range(n_verts))
        G.add_edges_from(edges)
        ref_labels = {}
        for comp_id, component in enumerate(nx.connected_components(G)):
            for v in component:
                ref_labels[v] = comp_id
        
        # Verify reference implementation
        passed, msg = verify_wcc(ref_labels, edges, n_verts)
        print(f"  Reference verification: {'PASS' if passed else 'FAIL'} - {msg}")
