"""
H100 Profiling and Benchmarking Utilities for KernelForge.

Integrates NVIDIA Nsight Compute (NCU) for hardware-level telemetry
and provides standardized benchmarking with cudaEvent timing.
"""
import argparse
import subprocess
import tempfile
import os
import json
import time
import numpy as np
import ctypes
from typing import Dict, List, Any, Optional
import cupy as cp
import networkx as nx

from verification.pac_verify import generate_test_graphs, edges_to_csr, run_kernel_verification


class H100Profiler:
    """Advanced profiler for H100 CUDA kernels with NCU integration."""
    
    def __init__(self, kernel_path: str):
        self.kernel_path = kernel_path
        self.ncu_available = self._check_ncu()
        self.baseline_results = {}
        
    def _check_ncu(self) -> bool:
        """Check if Nsight Compute (NCU) is available."""
        try:
            result = subprocess.run(["ncu", "--version"], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            print("Warning: Nsight Compute (NCU) not found. Hardware profiling disabled.")
            return False
    
    def profile_baseline(self, graph_sizes: List[int] = None) -> Dict[str, Any]:
        """
        Profile baseline implementations (cuGraph and reference).
        
        Args:
            graph_sizes: List of vertex counts to test
            
        Returns:
            Baseline performance metrics
        """
        if graph_sizes is None:
            graph_sizes = [1000, 5000, 10000, 50000, 100000]
        
        print("Profiling baseline implementations...")
        
        results = {
            "cuGraph": {},
            "reference": {},
            "graph_sizes": graph_sizes,
        }
        
        for n_vertices in graph_sizes:
            print(f"  Testing {n_vertices} vertices...")
            
            # Generate test graph
            graphs = generate_test_graphs(n_vertices)
            graph_type, edges, _ = graphs[0]  # Use RMAT graph
            
            # Profile cuGraph directly, with measured NetworkX fallback if unavailable.
            cuGraph_time = self._profile_cugraph(edges, n_vertices)
            results["cuGraph"][n_vertices] = cuGraph_time
            
            # Profile reference implementation
            ref_time = self._profile_reference_implementation(edges, n_vertices)
            results["reference"][n_vertices] = ref_time
        
        self.baseline_results = results
        return results
    
    def _profile_cugraph(self, edges: List[tuple], n_vertices: int) -> Dict[str, float]:
        """Profile cuGraph WCC implementation with measured fallback."""
        try:
            # Try to use actual cuGraph if available
            import cugraph
            import cudf
            
            # Create cuGraph
            gdf = cudf.DataFrame()
            gdf['src'] = [u for u, v in edges]
            gdf['dst'] = [v for u, v in edges]
            G = cugraph.Graph()
            G.from_cudf_edgelist(gdf, source='src', destination='dst')
            
            # Warmup
            for _ in range(10):
                cugraph.weakly_connected_components(G)
            
            # Benchmark
            times = []
            for _ in range(50):
                start = time.time()
                result = cugraph.weakly_connected_components(G)
                cp.cuda.Device(0).synchronize()
                end = time.time()
                times.append((end - start) * 1000)  # Convert to ms
            
            times = np.array(times)
            return {
                "mean_ms": float(np.mean(times)),
                "median_ms": float(np.median(times)),
                "std_ms": float(np.std(times)),
                "min_ms": float(np.min(times)),
                "max_ms": float(np.max(times)),
            }
            
        except ImportError:
            # Fallback: measured CPU baseline via NetworkX.
            return self._profile_networkx_baseline(edges, n_vertices, "cuGraph")
    
    def _profile_reference_implementation(self, edges: List[tuple], n_vertices: int) -> Dict[str, float]:
        """Profile reference WCC implementation."""
        return self._profile_networkx_baseline(edges, n_vertices, "reference")
    
    def _profile_networkx_baseline(self, edges: List[tuple], n_vertices: int, name: str) -> Dict[str, float]:
        """Profile NetworkX baseline implementation."""
        G = nx.Graph()
        G.add_nodes_from(range(n_vertices))
        G.add_edges_from(edges)
        
        # Warmup
        for _ in range(10):
            list(nx.connected_components(G))
        
        # Benchmark
        times = []
        for _ in range(50):
            start = time.time()
            components = list(nx.connected_components(G))
            end = time.time()
            times.append((end - start) * 1000)
        
        times = np.array(times)
        return {
            "mean_ms": float(np.mean(times)),
            "median_ms": float(np.median(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
        }
    
    def profile_kernel(self, 
                       warmup_iters: int = 50,
                       benchmark_runs: int = 30,
                       ncu_profile: bool = False,
                       graph_size: int = 10000) -> Dict[str, Any]:
        """
        Profile compiled CUDA kernel with comprehensive metrics.
        
        Args:
            warmup_iters: Number of warmup iterations
            benchmark_runs: Number of timed benchmark runs
            ncu_profile: Whether to run NCU hardware profiling
            graph_size: Size of test graph
            
        Returns:
            Comprehensive profiling results
        """
        print(f"Profiling kernel: {self.kernel_path}")
        
        # Generate test graph
        graphs = generate_test_graphs(graph_size)
        graph_type, edges, n_vertices = graphs[0]
        
        results = {
            "kernel_path": self.kernel_path,
            "graph_info": {
                "type": graph_type,
                "vertices": n_vertices,
                "edges": len(edges),
            },
            "compilation": self._check_compilation(),
            "correctness": self._verify_correctness(edges, n_vertices),
            "performance": self._benchmark_performance(edges, n_vertices, warmup_iters, benchmark_runs),
        }
        
        if ncu_profile and self.ncu_available:
            results["hardware_metrics"] = self._run_ncu_profiling(edges, n_vertices)
        
        # Calculate speedups
        if self.baseline_results:
            results["speedups"] = self._calculate_speedups(results["performance"], graph_size)
        
        return results
    
    def _check_compilation(self) -> Dict[str, Any]:
        """Check if kernel compiles successfully."""
        try:
            # Try to load the shared library
            lib = ctypes.CDLL(self.kernel_path)
            return {
                "compiles": True,
                "error": None,
            }
        except Exception as e:
            return {
                "compiles": False,
                "error": str(e),
            }
    
    def _verify_correctness(self, edges: List[tuple], n_vertices: int) -> Dict[str, Any]:
        """Verify kernel correctness using PAC verification."""
        try:
            from pac_verify import verify_wcc, run_kernel_verification
            
            kernel_labels = run_kernel_verification(self.kernel_path, edges, n_vertices)
            passed, message = verify_wcc(kernel_labels, edges, n_vertices)
            
            return {
                "passed": passed,
                "message": message,
            }
        except Exception as e:
            return {
                "passed": False,
                "message": f"Verification failed: {str(e)}",
            }
    
    def _benchmark_performance(self, 
                              edges: List[tuple], 
                              n_vertices: int,
                              warmup_iters: int,
                              benchmark_runs: int) -> Dict[str, Any]:
        """Benchmark kernel performance with cudaEvent timing."""
        try:
            # Convert to CSR format
            row_ptr, col_idx = edges_to_csr(edges, n_vertices)
            
            # Load kernel
            lib = ctypes.CDLL(self.kernel_path)
            
            # Setup CUDA events for precise timing
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            
            # Warmup iterations (critical for stable measurements)
            print(f"  Running {warmup_iters} warmup iterations...")
            for _ in range(warmup_iters):
                run_kernel_verification(self.kernel_path, edges, n_vertices)
            cp.cuda.Device(0).synchronize()
            
            # Timed benchmark runs
            print(f"  Running {benchmark_runs} timed iterations...")
            times = []
            for i in range(benchmark_runs):
                start.record()
                run_kernel_verification(self.kernel_path, edges, n_vertices)
                end.record()
                end.synchronize()
                
                elapsed = cp.cuda.get_elapsed_time(start, end)
                times.append(elapsed)
                
                if (i + 1) % 10 == 0:
                    print(f"    Completed {i + 1}/{benchmark_runs} runs")
            
            times = np.array(times)
            
            return {
                "mean_ms": float(np.mean(times)),
                "median_ms": float(np.median(times)),
                "std_ms": float(np.std(times)),
                "min_ms": float(np.min(times)),
                "max_ms": float(np.max(times)),
                "warmup_iters": warmup_iters,
                "benchmark_runs": benchmark_runs,
                "times_ms": times.tolist(),
            }
            
        except Exception as e:
            return {
                "error": f"Benchmarking failed: {str(e)}",
                "mean_ms": 0.0,
                "median_ms": 0.0,
            }
    
    def _run_ncu_profiling(self, edges: List[tuple], n_vertices: int) -> Dict[str, Any]:
        """Run Nsight Compute profiling for hardware metrics."""
        if not self.ncu_available:
            return {"error": "NCU not available"}
        
        print("  Running NCU hardware profiling...")
        
        with tempfile.NamedTemporaryFile(suffix='.nsight-profiler', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        try:
            # Create a simple test program that calls the kernel
            test_program = self._generate_ncu_test_program(edges, n_vertices)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as prog_file:
                prog_file.write(test_program)
                prog_path = prog_file.name
            
            # Compile test program
            compile_cmd = [
                "nvcc", f"-arch={os.getenv('KERNELFORGE_TARGET_ARCH', 'sm_80')}", "-O3",
                prog_path, self.kernel_path,
                "-o", prog_path.replace('.cpp', ''),
                "-I/usr/local/cuda/include"
            ]
            
            subprocess.run(compile_cmd, check=True, capture_output=True)
            
            # Run NCU profiling
            ncu_cmd = [
                "ncu",
                "--target-processes", "all",
                "--kernel-name-base", "function",
                "--section", "ComputeWorkloadAnalysis",
                "--section", "MemoryWorkloadAnalysis",
                "--section", "LaunchStatistics",
                "--section", "MemoryTopology",
                "--section", "SMActivity",
                "--export", "json",
                "--output", output_path,
                prog_path.replace('.cpp', '')
            ]
            
            result = subprocess.run(ncu_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Parse NCU JSON output
                return self._parse_ncu_output(output_path + ".json")
            else:
                return {
                    "error": f"NCU profiling failed: {result.stderr}",
                    "stdout": result.stdout,
                }
                
        except Exception as e:
            return {"error": f"NCU profiling exception: {str(e)}"}
        finally:
            # Cleanup temporary files
            for path in [output_path, output_path + ".json", prog_path, prog_path.replace('.cpp', '')]:
                if os.path.exists(path):
                    os.unlink(path)
    
    def _generate_ncu_test_program(self, edges: List[tuple], n_vertices: int) -> str:
        """Generate C++ test program for NCU profiling."""
        row_ptr, col_idx = edges_to_csr(edges, n_vertices)
        
        return f'''
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

extern "C" {{
    void wcc_kernel(const int* row_ptr, const int* col_idx, int num_vertices, int* labels);
}}

int main() {{
    // Test graph data
    int n_vertices = {n_vertices};
    std::vector<int> row_ptr = {{{', '.join(map(str, row_ptr))}}};
    std::vector<int> col_idx = {{{', '.join(map(str, col_idx))}}};
    std::vector<int> labels(n_vertices);
    
    // Run kernel multiple times for profiling
    for (int i = 0; i < 10; i++) {{
        wcc_kernel(row_ptr.data(), col_idx.data(), n_vertices, labels.data());
        cudaDeviceSynchronize();
    }}
    
    return 0;
}}
'''
    
    def _parse_ncu_output(self, json_path: str) -> Dict[str, Any]:
        """Parse NCU JSON output for key metrics."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Extract key metrics (simplified parsing)
            metrics = {}
            
            for report in data.get('Reports', []):
                for kernel in report.get('Kernels', []):
                    kernel_name = kernel.get('Name', 'unknown')
                    metrics[kernel_name] = {}
                    
                    for metric in kernel.get('Metrics', []):
                        name = metric.get('Name', '')
                        value = metric.get('Value', 0)
                        
                        # Extract key metrics
                        if any(key in name.lower() for key in [
                            'sm__warps_active.avg.pct_of_peak_sustained',
                            'sm__inst_executed.avg.pct_of_peak_sustained',
                            'dram__throughput.avg.pct_of_peak_sustained',
                            'l2__throughput.avg.pct_of_peak_sustained',
                            'memory__throughput.avg.pct_of_peak_sustained',
                            'launch__blocks_per_sm.avg',
                            'launch__threads_per_block.avg',
                        ]):
                            metrics[kernel_name][name] = value
            
            return {
                "ncu_metrics": metrics,
                "summary": "NCU profiling completed successfully",
            }
            
        except Exception as e:
            return {"error": f"Failed to parse NCU output: {str(e)}"}
    
    def _calculate_speedups(self, performance: Dict[str, Any], graph_size: int) -> Dict[str, float]:
        """Calculate speedups relative to baselines."""
        speedups = {}
        
        if not self.baseline_results:
            return speedups
        
        kernel_time = performance.get("median_ms", 0)
        
        # Speedup vs cuGraph
        if graph_size in self.baseline_results.get("cuGraph", {}):
            cugraph_time = self.baseline_results["cuGraph"][graph_size].get("median_ms", 0)
            if cugraph_time > 0:
                speedups["vs_cuGraph"] = cugraph_time / kernel_time
        
        # Speedup vs reference
        if graph_size in self.baseline_results.get("reference", {}):
            ref_time = self.baseline_results["reference"][graph_size].get("median_ms", 0)
            if ref_time > 0:
                speedups["vs_reference"] = ref_time / kernel_time
        
        return speedups
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive profiling report."""
        report = []
        report.append("=" * 60)
        report.append("KERNELFORGE H100 PROFILING REPORT")
        report.append("=" * 60)
        
        # Basic info
        report.append(f"Kernel: {results['kernel_path']}")
        graph_info = results['graph_info']
        report.append(f"Graph: {graph_info['type']} with {graph_info['vertices']} vertices, {graph_info['edges']} edges")
        
        # Compilation status
        compilation = results['compilation']
        report.append(f"\nCompilation: {'✓ PASS' if compilation['compiles'] else '✗ FAIL'}")
        if not compilation['compiles']:
            report.append(f"Error: {compilation['error']}")
            return "\n".join(report)
        
        # Correctness
        correctness = results['correctness']
        report.append(f"Correctness: {'✓ PASS' if correctness['passed'] else '✗ FAIL'}")
        report.append(f"Message: {correctness['message']}")
        
        if not correctness['passed']:
            return "\n".join(report)
        
        # Performance
        perf = results['performance']
        if 'error' in perf:
            report.append(f"\nPerformance: ERROR - {perf['error']}")
            return "\n".join(report)
        
        report.append(f"\nPerformance Metrics:")
        report.append(f"  Mean time:   {perf['mean_ms']:.3f} ms")
        report.append(f"  Median time: {perf['median_ms']:.3f} ms")
        report.append(f"  Std dev:     {perf['std_ms']:.3f} ms")
        report.append(f"  Min time:    {perf['min_ms']:.3f} ms")
        report.append(f"  Max time:    {perf['max_ms']:.3f} ms")
        
        # Speedups
        if 'speedups' in results and results['speedups']:
            report.append(f"\nSpeedups:")
            for baseline, speedup in results['speedups'].items():
                report.append(f"  {baseline}: {speedup:.2f}x")
        
        # Hardware metrics
        if 'hardware_metrics' in results:
            hw = results['hardware_metrics']
            if 'error' in hw:
                report.append(f"\nHardware Profiling: ERROR - {hw['error']}")
            else:
                report.append(f"\nHardware Metrics:")
                ncu_metrics = hw.get('ncu_metrics', {})
                for kernel, metrics in ncu_metrics.items():
                    report.append(f"  {kernel}:")
                    for name, value in list(metrics.items())[:5]:  # Show first 5 metrics
                        report.append(f"    {name}: {value}")
        
        return "\n".join(report)


def main():
    """Main profiling interface."""
    parser = argparse.ArgumentParser(description="KernelForge H100 Profiler")
    parser.add_argument("--kernel", required=True, help="Path to compiled kernel (.so file)")
    parser.add_argument("--baseline", action="store_true", help="Profile baseline implementations")
    parser.add_argument("--ncu", action="store_true", help="Run NCU hardware profiling")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=50, help="Benchmark runs")
    parser.add_argument("--graph-size", type=int, default=10000, help="Test graph size")
    
    args = parser.parse_args()
    
    profiler = H100Profiler(args.kernel)
    
    # Profile baselines if requested
    if args.baseline:
        profiler.profile_baseline()
    
    # Profile kernel
    results = profiler.profile_kernel(
        warmup_iters=args.warmup,
        benchmark_runs=args.runs,
        ncu_profile=args.ncu,
        graph_size=args.graph_size
    )
    
    # Generate and print report
    report = profiler.generate_report(results)
    print(report)
    
    # Save results
    output_file = f"profile_results_{os.path.basename(args.kernel)}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
