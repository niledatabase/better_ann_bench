#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import importlib.util

from benchmark_config import BenchmarkConfig, AlgorithmConfig, DatasetGenerationConfig, DatasetMode
from concurrent_benchmark import ConcurrentBenchmark, VectorIndex, WorkloadSpec
from metrics import BenchmarkResults
from dataset_generator import DatasetGenerator, DatasetConfig
import numpy as np


def load_algorithm(algorithm_config: AlgorithmConfig) -> VectorIndex:
    """Dynamically load algorithm implementation"""
    module_name = f"algorithms.{algorithm_config.implementation}"
    
    try:
        spec = importlib.util.spec_from_file_location(
            module_name, 
            f"algorithms/{algorithm_config.implementation}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Handle special cases for class names
        if algorithm_config.implementation == 'pgvector':
            class_name = 'PgVector'
        else:
            class_name = algorithm_config.implementation.capitalize()
        algorithm_class = getattr(module, class_name)
        
        return algorithm_class(
            build_params=algorithm_config.build_params,
            search_params=algorithm_config.search_params
        )
    except Exception as e:
        print(f"Failed to load algorithm {algorithm_config.implementation}: {e}")
        sys.exit(1)


def generate_synthetic_dataset(config: BenchmarkConfig) -> WorkloadSpec:
    """Generate synthetic dataset based on configuration"""
    gen_config = config.dataset_generation
    
    # Convert dict to DatasetGenerationConfig if needed
    if isinstance(gen_config, dict):
        from benchmark_config import DatasetGenerationConfig
        gen_config = DatasetGenerationConfig(**gen_config)
    
    # Skip ground truth for very large datasets to avoid memory issues
    skip_ground_truth = config.dataset_size > 10000000  # 10M vectors threshold
    
    dataset_config = DatasetConfig(
        name="synthetic_clustered",
        size=config.dataset_size,
        dimension=gen_config.dimension,
        query_size=config.query_count,
        seed=gen_config.seed,
        clusters=gen_config.clusters,
        cluster_std=gen_config.cluster_std,
        k_neighbors=gen_config.k_neighbors,
        skip_ground_truth=skip_ground_truth,
        chunk_size=gen_config.chunk_size,
        query_chunk_size=gen_config.query_chunk_size,
        gt_train_block_size=gen_config.gt_train_block_size,
        gt_query_block_size=gen_config.gt_query_block_size
    )
    
    generator = DatasetGenerator(dataset_config)
    
    # Save dataset if path is specified
    save_path = config.save_dataset_path if hasattr(config, 'save_dataset_path') else None
    train_vectors, query_vectors, ground_truth = generator.generate_dataset(save_path)
    
    # Handle file paths for large datasets
    if isinstance(train_vectors, str):
        # Large dataset - use memory-mapped files
        print(f"Using memory-mapped dataset from {train_vectors}")
        return WorkloadSpec(
            insert_vectors_path=train_vectors,
            search_queries_path=query_vectors,
            ground_truth=ground_truth,
            k=10
        )
    else:
        # Small dataset - use numpy arrays
        return WorkloadSpec(
            insert_vectors=train_vectors,
            search_queries=query_vectors,
            ground_truth=ground_truth,
            k=10
        )


def load_existing_dataset(dataset_path: str) -> WorkloadSpec:
    """Load existing dataset from file"""
    print(f"Loading dataset from {dataset_path}")
    
    try:
        import h5py
        with h5py.File(dataset_path, 'r') as f:
            train_data = f['train'][:]
            test_data = f['test'][:]
            # Try to load neighbors, but don't fail if they don't exist
            try:
                neighbors = f['neighbors'][:]
                print("Loaded ground truth from dataset")
            except KeyError:
                print("Warning: No ground truth (neighbors) found in dataset - recall calculation will be skipped")
                neighbors = None
    except ImportError:
        print("h5py not available, trying numpy format")
        data = np.load(dataset_path, allow_pickle=True)
        train_data = data['train']
        test_data = data['test'] 
        # Try to load neighbors, but don't fail if they don't exist
        try:
            neighbors = data['neighbors']
            print("Loaded ground truth from dataset")
        except KeyError:
            print("Warning: No ground truth (neighbors) found in dataset - recall calculation will be skipped")
            neighbors = None
    
    print(f"Loaded dataset: {len(train_data)} training vectors, {len(test_data)} test queries")
    
    return WorkloadSpec(
        insert_vectors=train_data,
        search_queries=test_data,
        ground_truth=neighbors,
        k=10
    )


def save_results(results: BenchmarkResults, output_path: str, algorithm_name: str):
    """Save benchmark results to JSON"""
    output_data = {
        'algorithm': algorithm_name,
        'dataset_size': results.dataset_size,
        'recall': results.recall,
        'search_latencies': {
            'p50': results.search_latencies.p50,
            'p95': results.search_latencies.p95,
            'p99': results.search_latencies.p99,
            'mean': results.search_latencies.mean,
            'min': results.search_latencies.min,
            'max': results.search_latencies.max
        },
        'search_qps': results.search_qps,
        'insert_qps': results.insert_qps,
        'total_queries': results.total_queries,
        'total_inserts': results.total_inserts,
        'runtime_seconds': results.runtime_seconds,
        'search_params': results.search_params,
        # Concurrency information
        'concurrency': {
            'concurrent_searchers': results.concurrent_searchers,
            'concurrent_inserters': results.concurrent_inserters,
            'benchmark_mode': results.benchmark_mode
        }
    }
    
    if results.insert_latencies:
        output_data['insert_latencies'] = {
            'p50': results.insert_latencies.p50,
            'p95': results.insert_latencies.p95,
            'p99': results.insert_latencies.p99,
            'mean': results.insert_latencies.mean,
            'min': results.insert_latencies.min,
            'max': results.insert_latencies.max
        }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {output_path}")


def print_results(results: BenchmarkResults, algorithm_name: str):
    """Print results to console"""
    print(f"\n=== Benchmark Results for {algorithm_name} ===")
    print(f"Dataset Size: {results.dataset_size:,} vectors")
    print(f"Runtime: {results.runtime_seconds:.2f} seconds")
    if results.recall is not None:
        print(f"Recall: {results.recall:.4f}")
    else:
        print(f"Recall: Not calculated (no ground truth available)")
    print(f"Search Parameters: {results.search_params}")
    print(f"Benchmark Mode: {results.benchmark_mode}")
    print(f"Concurrency: {results.concurrent_searchers} searchers, {results.concurrent_inserters} inserters")
    print(f"\nSearch Performance:")
    print(f"  QPS: {results.search_qps:.2f}")
    print(f"  Total Queries: {results.total_queries}")
    print(f"  Latency (ms) - P50: {results.search_latencies.p50:.2f}, P95: {results.search_latencies.p95:.2f}, P99: {results.search_latencies.p99:.2f}")
    
    if results.insert_latencies:
        print(f"\nInsert Performance:")
        print(f"  QPS: {results.insert_qps:.2f}")
        print(f"  Total Inserts: {results.total_inserts}")
        print(f"  Latency (ms) - P50: {results.insert_latencies.p50:.2f}, P95: {results.insert_latencies.p95:.2f}, P99: {results.insert_latencies.p99:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Scalable Vector Benchmark')
    parser.add_argument('--config', required=True, help='Path to benchmark config YAML')
    parser.add_argument('--algorithm', required=True, help='Path to algorithm config YAML')
    
    args = parser.parse_args()
    
    # Load configurations
    benchmark_config = BenchmarkConfig.from_yaml(args.config)
    with open(args.algorithm, 'r') as f:
        import yaml
        algorithm_data = yaml.safe_load(f)
        algorithm_config = AlgorithmConfig(**algorithm_data)
        
    #print("=== Benchmark Configuration ===")
    #print(json.dumps(benchmark_config.__dict__, indent=2))
    #print("\n=== Algorithm Configuration ===")
    #print(json.dumps(algorithm_config.__dict__, indent=2))

    # light validations of config consistency:
    if algorithm_config.build_params.get('reuse_table', False) and benchmark_config.concurrent_inserters > 0:
        raise ValueError("Reuse table mode is not supported with concurrent insert workers. Please set concurrent_inserters to 0.")
    if algorithm_config.build_params.get('reuse_table', False) and benchmark_config.dataset_mode == DatasetMode.GENERATE_SYNTHETIC:
        raise ValueError("Reuse table mode is not supported with synthetic data generation. Please make sure dataset_path is set to a file.")

    # Load dataset
    if benchmark_config.dataset_mode == DatasetMode.GENERATE_SYNTHETIC:
        workload = generate_synthetic_dataset(benchmark_config)
    elif benchmark_config.dataset_mode == DatasetMode.LOAD_FROM_FILE:
        workload = load_existing_dataset(benchmark_config.dataset_path)
    else:
        raise ValueError(f"Invalid dataset mode: {benchmark_config.dataset_mode}")

    # Load algorithm
    algorithm = load_algorithm(algorithm_config)

    # Run benchmark
    benchmark = ConcurrentBenchmark(benchmark_config, algorithm_config)
    results = benchmark.run_benchmark(algorithm, workload)

    # Output results
    print_results(results, algorithm_config.name)

    # Save results with timestamp
    import os
    from datetime import datetime
    
    # Always save results even if no output path configured
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    algorithm_name = algorithm_config.name.lower().replace('-', '_')
    output_file = os.path.join("results", f"{algorithm_name}_{timestamp}.json")
    save_results(results, output_file, algorithm_config.name)


if __name__ == '__main__':
    main()