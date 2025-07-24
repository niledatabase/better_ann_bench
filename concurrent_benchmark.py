import time
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Tuple, Protocol, Any
from abc import abstractmethod
import random
from dataclasses import dataclass

from benchmark_config import BenchmarkConfig, AlgorithmConfig
from metrics import MetricsCollector, BenchmarkResults


class VectorIndex(Protocol):
    @abstractmethod
    def build(self, vectors: np.ndarray) -> None:
        pass
    
    @abstractmethod
    def insert(self, vectors: np.ndarray) -> None:
        pass
    
    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> np.ndarray:
        pass
    
    @abstractmethod
    def supports_concurrent_operations(self) -> bool:
        pass


@dataclass
class WorkloadSpec:
    insert_vectors: np.ndarray
    search_queries: np.ndarray
    ground_truth: np.ndarray
    k: int = 10


class ConcurrentBenchmark:
    def __init__(self, config: BenchmarkConfig, algorithm_config: AlgorithmConfig):
        self.config = config
        self.algorithm_config = algorithm_config
        self.metrics = MetricsCollector()
        self._stop_event = threading.Event()
        
    def _search_worker(self, index: VectorIndex, queries: np.ndarray, k: int, worker_id: int):
        local_query_count = 0
        query_idx = 0
        
        while not self._stop_event.is_set():
            # Loop through queries repeatedly until timeout
            if query_idx >= len(queries):
                query_idx = 0  # Reset to start of queries
            
            batch_end = min(query_idx + self.config.search_batch_size, len(queries))
            batch_queries = queries[query_idx:batch_end]
            
            for query in batch_queries:
                if self._stop_event.is_set():
                    break
                    
                start_time = time.time()
                try:
                    results = index.search(query, k)
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000
                    self.metrics.record_search_latency(latency_ms)
                    local_query_count += 1
                except Exception as e:
                    print(f"Search worker {worker_id} error: {e}")
                    
            query_idx = batch_end
            
        print(f"Search worker {worker_id} completed {local_query_count} queries")
    
    def _insert_worker(self, index: VectorIndex, vectors: np.ndarray, worker_id: int):
        local_insert_count = 0
        vector_idx = 0
        
        while not self._stop_event.is_set() and vector_idx < len(vectors):
            batch_end = min(vector_idx + self.config.insert_batch_size, len(vectors))
            batch_vectors = vectors[vector_idx:batch_end]
            
            start_time = time.time()
            try:
                index.insert(batch_vectors)
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                self.metrics.record_insert_latency(latency_ms)
                local_insert_count += len(batch_vectors)
            except Exception as e:
                print(f"Insert worker {worker_id} error: {e}")
                
            vector_idx = batch_end
            
        print(f"Insert worker {worker_id} completed {local_insert_count} inserts")
    
    def run_benchmark(self, index: VectorIndex, workload: WorkloadSpec) -> BenchmarkResults:
        print(f"Starting benchmark")
        print(f"Dataset size: {len(workload.insert_vectors)}")
        print(f"Query count: {len(workload.search_queries)}")
        print(f"Concurrent inserters: {self.config.concurrent_inserters}")
        print(f"Concurrent searchers: {self.config.concurrent_searchers}")
        
        # Set concurrency information in metrics collector
        self.metrics.set_concurrency_info(
            concurrent_searchers=self.config.concurrent_searchers,
            concurrent_inserters=self.config.concurrent_inserters,
            benchmark_mode=self.config.benchmark_mode
        )
        
        # Check if we should reuse existing table
        reuse_table = getattr(index, 'reuse_table', False)
        if reuse_table:
            print(f"Reuse table mode enabled - skipping data loading and insert operations")
        
        if self.config.benchmark_mode == "search_only":
            print(f"Running in search-only mode")
            if hasattr(index, 'build') and len(workload.insert_vectors) > 0 and not reuse_table:
                print(f"Inserting and indexing {len(workload.insert_vectors)} vectors")
                index.build(workload.insert_vectors)
            elif reuse_table:
                print(f"Building index for existing table (no data insertion)")
                index.build(workload.insert_vectors)  # This will just verify table exists
            
            # No remaining vectors since we built everything or reusing table
            remaining_vectors = []
        else:  # hybrid mode
            print(f"Running in hybrid mode (concurrent inserts + searches)")
            if hasattr(index, 'build') and len(workload.insert_vectors) > 0 and not reuse_table:
                # Insert initial vectors with subset of vectors
                percentage_size = len(workload.insert_vectors) // (100 // self.config.initial_build_percentage)
                initial_build_size = min(self.config.initial_build_size_cap, percentage_size)
                print(f"Inserting and indexing initial {initial_build_size} vectors ({self.config.initial_build_percentage}% of {len(workload.insert_vectors)}, capped at {self.config.initial_build_size_cap})...")
                index.build(workload.insert_vectors[:initial_build_size])
                remaining_vectors = workload.insert_vectors[initial_build_size:]
            elif reuse_table:
                print(f"Building index for existing table (no data insertion)")
                index.build(workload.insert_vectors)  # This will just verify table exists
                remaining_vectors = []  # No vectors to insert in reuse mode
            else:
                remaining_vectors = workload.insert_vectors
        
        # Split vectors and queries among workers
        vectors_per_inserter = len(remaining_vectors) // max(1, self.config.concurrent_inserters)
        queries_per_searcher = len(workload.search_queries) // max(1, self.config.concurrent_searchers)
        
        self.metrics.start_timing()
        
        max_workers = self.config.concurrent_searchers
        if self.config.benchmark_mode == "hybrid":
            max_workers += self.config.concurrent_inserters
            
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: List[Future] = []
            
            # Start insert workers (if in hybrid mode and there are vectors to insert)
            if self.config.benchmark_mode == "hybrid" and len(remaining_vectors) > 0 and not reuse_table:
                for i in range(self.config.concurrent_inserters):
                    start_idx = i * vectors_per_inserter
                    end_idx = start_idx + vectors_per_inserter if i < self.config.concurrent_inserters - 1 else len(remaining_vectors)
                    worker_vectors = remaining_vectors[start_idx:end_idx]
                    
                    if len(worker_vectors) > 0:
                        future = executor.submit(self._insert_worker, index, worker_vectors, i)
                        futures.append(future)
            
            # Start search workers
            for i in range(self.config.concurrent_searchers):
                start_idx = i * queries_per_searcher
                end_idx = start_idx + queries_per_searcher if i < self.config.concurrent_searchers - 1 else len(workload.search_queries)
                worker_queries = workload.search_queries[start_idx:end_idx]
                
                if len(worker_queries) > 0:
                    future = executor.submit(self._search_worker, index, worker_queries, workload.k, i)
                    futures.append(future)
            
            # Run for specified duration
            time.sleep(self.config.max_runtime_seconds)
            self._stop_event.set()
            
            # Wait for all workers to complete
            for future in futures:
                try:
                    future.result(timeout=30)
                except Exception as e:
                    print(f"Worker error: {e}")
        
        self.metrics.end_timing()
        
        # Calculate recall by running all queries once:
        
        recall_sum = 0.0
        print(f"Calculating recall on {len(workload.search_queries)} queries with k={workload.k}")
        
        # Calculate recall for all queries
        for idx in range(len(workload.search_queries)):
            query = workload.search_queries[idx]
            results = index.search(query, workload.k)
            ground_truth = workload.ground_truth[idx]
            recall = self.metrics.calculate_recall(ground_truth, results, workload.k, workload.insert_vectors)
            recall_sum += recall
            
        average_recall = recall_sum / len(workload.search_queries)
        
        return self.metrics.get_results(
            average_recall, 
            len(workload.insert_vectors),
            self.algorithm_config.search_params
        )