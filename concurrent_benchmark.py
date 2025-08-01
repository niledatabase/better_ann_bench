import time
import threading
import numpy as np
import h5py
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Tuple, Protocol, Any, Optional
from abc import abstractmethod
import random
from dataclasses import dataclass

from benchmark_config import BenchmarkConfig, AlgorithmConfig, DatasetMode
from metrics import MetricsCollector, BenchmarkResults


class VectorIndex(Protocol):
    @abstractmethod
    def build(self, vectors: np.ndarray) -> None:
        """Build index with all vectors loaded in memory"""
        pass
    
    @abstractmethod
    def build_streaming(self, vectors_path: str, chunk_size: int = 10000) -> None:
        """Build index by streaming vectors from file in chunks"""
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
    insert_vectors: Optional[np.ndarray] = None
    search_queries: Optional[np.ndarray] = None
    insert_vectors_path: Optional[str] = None
    search_queries_path: Optional[str] = None
    ground_truth: Optional[np.ndarray] = None
    k: int = 10
    
    def __post_init__(self):
        # Ensure we have either arrays or paths, not both
        if self.insert_vectors is not None and self.insert_vectors_path is not None:
            raise ValueError("Cannot specify both insert_vectors and insert_vectors_path")
        if self.search_queries is not None and self.search_queries_path is not None:
            raise ValueError("Cannot specify both search_queries and search_queries_path")
        
        # Ensure we have at least one of each
        if self.insert_vectors is None and self.insert_vectors_path is None:
            raise ValueError("Must specify either insert_vectors or insert_vectors_path")
        if self.search_queries is None and self.search_queries_path is None:
            raise ValueError("Must specify either search_queries or search_queries_path")
    
    @property
    def dataset_size(self) -> int:
        """Get dataset size from arrays or files"""
        if self.insert_vectors is not None:
            return len(self.insert_vectors)
        else:
            # Load size from file
            import h5py
            with h5py.File(self.insert_vectors_path, 'r') as f:
                return f['train'].shape[0]
    
    @property
    def query_count(self) -> int:
        """Get query count from arrays or files"""
        if self.search_queries is not None:
            return len(self.search_queries)
        else:
            # Load count from file
            import h5py
            with h5py.File(self.search_queries_path, 'r') as f:
                return f['test'].shape[0]


class MemoryMappedVectors:
    """Memory-mapped interface for loading vectors from HDF5 files in chunks"""
    
    def __init__(self, file_path: str, dataset_name: str):
        self.file_path = file_path
        self.dataset_name = dataset_name
        self._file = None
        self._dataset = None
        self._shape = None

    
    def __enter__(self):
        self._file = h5py.File(self.file_path, 'r')
        self._dataset = self._file[self.dataset_name]
        self._shape = self._dataset.shape
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
            self._file = None
            self._dataset = None
    
    def __len__(self):
        if self._shape is None:
            with h5py.File(self.file_path, 'r') as f:
                return f[self.dataset_name].shape[0]
        return self._shape[0]
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._dataset[idx]
        else:
            return self._dataset[idx:idx+1]
    
    def get_batch(self, start: int, end: int) -> np.ndarray:
        """Get a batch of vectors from the file"""
        return self._dataset[start:end]
    
    def get_all(self) -> np.ndarray:
        """Load all vectors into memory (use with caution for large datasets)"""
        return self._dataset[:]


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
    
    def _search_worker_memory_mapped(self, index: VectorIndex, queries_path: str, start_idx: int, end_idx: int, k: int, worker_id: int):
        """Search worker that uses memory-mapped query files"""
        local_query_count = 0
        query_idx = 0
        
        with MemoryMappedVectors(queries_path, 'test') as queries:
            worker_queries = queries.get_batch(start_idx, end_idx)
            
            while not self._stop_event.is_set():
                # Loop through queries repeatedly until timeout
                if query_idx >= len(worker_queries):
                    query_idx = 0  # Reset to start of queries
                
                batch_end = min(query_idx + self.config.search_batch_size, len(worker_queries))
                batch_queries = worker_queries[query_idx:batch_end]
                
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
    
    def run_benchmark(self, index: VectorIndex, workload: WorkloadSpec) -> BenchmarkResults:
        print(f"Starting benchmark")
        
        # Handle memory-mapped files vs numpy arrays
        if workload.insert_vectors_path is not None:
            print(f"Using memory-mapped training vectors from {workload.insert_vectors_path}")
            print(f"Dataset size: {workload.dataset_size}")
        else:
            print(f"Dataset size: {len(workload.insert_vectors)}")
        
        if workload.search_queries_path is not None:
            print(f"Using memory-mapped query vectors from {workload.search_queries_path}")
            print(f"Query count: {workload.query_count}")
        else:
            print(f"Query count: {len(workload.search_queries)}")
        
        print(f"Concurrent inserters: {self.config.concurrent_inserters}")
        print(f"Concurrent searchers: {self.config.concurrent_searchers}")
        
        # Set concurrency information in metrics collector
        self.metrics.set_concurrency_info(
            concurrent_searchers=self.config.concurrent_searchers,
            concurrent_inserters=self.config.concurrent_inserters,
            benchmark_mode=self.config.benchmark_mode
        )
        
        if self.config.benchmark_mode == "search_only":
            print(f"Running in search-only mode")
            ## Reuse logic belongs to the algorithm, so its contained in the build method
            if workload.insert_vectors_path is not None:
                # Use streaming build for large datasets that don't fit in memory
                print(f"Using streaming build for large dataset...")
                index.build_streaming(workload.insert_vectors_path, chunk_size=50000)
                remaining_vectors = None  # No remaining vectors in search-only mode
            else:
                index.build(workload.insert_vectors)
                remaining_vectors = None  # No remaining vectors in search-only mode
        else:  # hybrid mode
            print(f"Running in hybrid mode (concurrent inserts + searches)")
            # Insert initial vectors with subset of vectors
            if workload.insert_vectors_path is not None:
                # Load vectors from memory-mapped file
                with MemoryMappedVectors(workload.insert_vectors_path, 'train') as vectors:
                    percentage_size = len(vectors) // (100 // self.config.initial_build_percentage)
                    initial_build_size = min(self.config.initial_build_size_cap, percentage_size)
                    print(f"Inserting and indexing initial {initial_build_size} vectors ({self.config.initial_build_percentage}% of {len(vectors)}, capped at {self.config.initial_build_size_cap})...")
                    index.build(vectors.get_batch(0, initial_build_size))
                    remaining_vectors = vectors.get_batch(initial_build_size, len(vectors))
            else:
                percentage_size = len(workload.insert_vectors) // (100 // self.config.initial_build_percentage)
                initial_build_size = min(self.config.initial_build_size_cap, percentage_size)
                print(f"Inserting and indexing initial {initial_build_size} vectors ({self.config.initial_build_percentage}% of {len(workload.insert_vectors)}, capped at {self.config.initial_build_size_cap})...")
                index.build(workload.insert_vectors[:initial_build_size])
                remaining_vectors = workload.insert_vectors[initial_build_size:]
        
        # Split vectors and queries among workers
        if remaining_vectors is not None:
            vectors_per_inserter = len(remaining_vectors) // max(1, self.config.concurrent_inserters)
        else:
            vectors_per_inserter = 0
        
        if workload.search_queries_path is not None:
            # Load queries from memory-mapped file
            with MemoryMappedVectors(workload.search_queries_path, 'test') as queries:
                queries_per_searcher = len(queries) // max(1, self.config.concurrent_searchers)
        else:
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
            if workload.search_queries_path is not None:
                # Use memory-mapped queries
                for i in range(self.config.concurrent_searchers):
                    start_idx = i * queries_per_searcher
                    end_idx = start_idx + queries_per_searcher if i < self.config.concurrent_searchers - 1 else workload.query_count
                    
                    if end_idx > start_idx:
                        future = executor.submit(self._search_worker_memory_mapped, index, workload.search_queries_path, start_idx, end_idx, workload.k, i)
                        futures.append(future)
            else:
                # Use numpy array queries
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
        

        # Calculate recall if ground truth is available
        if workload.ground_truth is not None:
            recall_sum = 0.0
            query_count = workload.query_count if workload.search_queries_path is not None else len(workload.search_queries)
            print(f"Calculating recall on {query_count} queries with k={workload.k}")
            
            # Load queries for recall calculation
            if workload.search_queries_path is not None:
                with MemoryMappedVectors(workload.search_queries_path, 'test') as queries:
                    # Calculate recall for all queries
                    for idx in range(query_count):
                        query = queries[idx]
                        results = index.search(query, workload.k)
                        ground_truth = workload.ground_truth[idx]
                        recall = self.metrics.calculate_recall(ground_truth, results, workload.k, None)  # No insert vectors needed for recall
                        recall_sum += recall
            else:
                # Calculate recall for all queries
                for idx in range(len(workload.search_queries)):
                    query = workload.search_queries[idx]
                    results = index.search(query, workload.k)
                    ground_truth = workload.ground_truth[idx]
                    recall = self.metrics.calculate_recall(ground_truth, results, workload.k, workload.insert_vectors)
                    recall_sum += recall
                    
            average_recall = recall_sum / query_count
        else:
            print("No ground truth available - skipping recall calculation")
            average_recall = None
        
        dataset_size = workload.dataset_size if workload.insert_vectors_path is not None else len(workload.insert_vectors)
        return self.metrics.get_results(
            average_recall, 
            dataset_size,
            self.algorithm_config.search_params
        )