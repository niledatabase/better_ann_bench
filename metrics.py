import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import statistics


@dataclass
class LatencyMetrics:
    p50: float
    p95: float
    p99: float
    mean: float
    min: float
    max: float


@dataclass
class BenchmarkResults:
    recall: Optional[float]
    search_latencies: LatencyMetrics
    insert_latencies: Optional[LatencyMetrics]
    search_qps: float
    insert_qps: float
    total_queries: int
    total_inserts: int
    runtime_seconds: float
    dataset_size: int
    search_params: Dict[str, Any]
    # Concurrency information
    concurrent_searchers: int
    concurrent_inserters: int
    benchmark_mode: str


class MetricsCollector:
    def __init__(self):
        self._search_latencies: List[float] = []
        self._insert_latencies: List[float] = []
        self._search_times: List[float] = []
        self._insert_times: List[float] = []
        self._lock = threading.Lock()
        self._start_time = None
        self._end_time = None
        # Concurrency information
        self._concurrent_searchers: int = 0
        self._concurrent_inserters: int = 0
        self._benchmark_mode: str = "search_only"
        # Error tracking
        self._search_errors: int = 0
        self._insert_errors: int = 0
        self._search_retries: int = 0
        self._insert_retries: int = 0
        
    def set_concurrency_info(self, concurrent_searchers: int, concurrent_inserters: int, benchmark_mode: str):
        """Set concurrency information for the benchmark"""
        self._concurrent_searchers = concurrent_searchers
        self._concurrent_inserters = concurrent_inserters
        self._benchmark_mode = benchmark_mode
        
    def start_timing(self):
        self._start_time = time.time()
        
    def end_timing(self):
        self._end_time = time.time()
        
    def record_search_latency(self, latency_ms: float):
        with self._lock:
            self._search_latencies.append(latency_ms)
            self._search_times.append(time.time())
            
    def record_insert_latency(self, latency_ms: float):
        with self._lock:
            self._insert_latencies.append(latency_ms)
            self._insert_times.append(time.time())
    
    def record_search_error(self):
        with self._lock:
            self._search_errors += 1
    
    def record_insert_error(self):
        with self._lock:
            self._insert_errors += 1
    
    def record_search_retry(self):
        with self._lock:
            self._search_retries += 1
    
    def record_insert_retry(self):
        with self._lock:
            self._insert_retries += 1
    
    def _calculate_latency_metrics(self, latencies: List[float]) -> LatencyMetrics:
        if not latencies:
            return LatencyMetrics(0, 0, 0, 0, 0, 0)
            
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        return LatencyMetrics(
            p50=np.percentile(sorted_latencies, 50),
            p95=np.percentile(sorted_latencies, 95),
            p99=np.percentile(sorted_latencies, 99),
            mean=statistics.mean(sorted_latencies),
            min=min(sorted_latencies),
            max=max(sorted_latencies)
        )
    
    def _calculate_qps(self, operation_times: List[float]) -> float:
        if not operation_times or not self._start_time or not self._end_time:
            return 0.0
        
        runtime = self._end_time - self._start_time
        return len(operation_times) / runtime if runtime > 0 else 0.0
    
    def get_progress_info(self, since_time: float) -> Dict[str, Any]:
        """Get progress information since a specific time"""
        with self._lock:
            current_time = time.time()
            elapsed = current_time - self._start_time if self._start_time else 0
            
            # Calculate QPS since the specified time
            recent_search_times = [t for t in self._search_times if t >= since_time]
            recent_insert_times = [t for t in self._insert_times if t >= since_time]
            
            time_window = current_time - since_time if since_time > 0 else elapsed
            search_qps = len(recent_search_times) / time_window if time_window > 0 else 0
            insert_qps = len(recent_insert_times) / time_window if time_window > 0 else 0
            
            # Calculate latency metrics for recent operations
            recent_search_latencies = [lat for i, lat in enumerate(self._search_latencies) 
                                     if self._search_times[i] >= since_time]
            recent_insert_latencies = [lat for i, lat in enumerate(self._insert_latencies) 
                                     if self._insert_times[i] >= since_time]
            
            search_latency_p50 = np.percentile(recent_search_latencies, 50) if recent_search_latencies else 0
            search_latency_p99 = np.percentile(recent_search_latencies, 99) if recent_search_latencies else 0
            insert_latency_p50 = np.percentile(recent_insert_latencies, 50) if recent_insert_latencies else 0
            insert_latency_p99 = np.percentile(recent_insert_latencies, 99) if recent_insert_latencies else 0
            
            return {
                'elapsed_seconds': elapsed,
                'search_qps': search_qps,
                'insert_qps': insert_qps,
                'search_latency_p50': search_latency_p50,
                'search_latency_p99': search_latency_p99,
                'insert_latency_p50': insert_latency_p50,
                'insert_latency_p99': insert_latency_p99,
                'search_errors': self._search_errors,
                'insert_errors': self._insert_errors,
                'search_retries': self._search_retries,
                'insert_retries': self._insert_retries,
                'total_searches': len(self._search_latencies),
                'total_inserts': len(self._insert_latencies)
            }
    
    def calculate_recall(self, ground_truth: np.ndarray, results: np.ndarray, k: int, training_vectors: np.ndarray = None) -> float:
        if len(results.shape) == 1:
            results = results.reshape(1, -1)
        if len(ground_truth.shape) == 1:
            ground_truth = ground_truth.reshape(1, -1)
            
        total_recall = 0.0
        for i in range(len(results)):
            if training_vectors is not None:
                # Compare actual vectors instead of indices
                gt_vectors = training_vectors[ground_truth[i][:k]]
                result_vectors = training_vectors[results[i][:k]]
                
                # Convert to tuples for set operations
                gt_set = set(tuple(vec) for vec in gt_vectors)
                result_set = set(tuple(vec) for vec in result_vectors)
                recall = len(gt_set.intersection(result_set)) / len(gt_set)
            else:
                # Fallback to index comparison
                gt_set = set(ground_truth[i][:k])
                result_set = set(results[i][:k])
                recall = len(gt_set.intersection(result_set)) / len(gt_set)
            
            total_recall += recall
            
        return total_recall / len(results)
    
    def get_results(self, recall: Optional[float], dataset_size: int, search_params: Dict[str, Any]) -> BenchmarkResults:
        search_latencies = self._calculate_latency_metrics(self._search_latencies)
        insert_latencies = self._calculate_latency_metrics(self._insert_latencies) if self._insert_latencies else None
        
        search_qps = self._calculate_qps(self._search_times)
        insert_qps = self._calculate_qps(self._insert_times)
        
        runtime = (self._end_time - self._start_time) if self._start_time and self._end_time else 0
        
        return BenchmarkResults(
            recall=recall,
            search_latencies=search_latencies,
            insert_latencies=insert_latencies,
            search_qps=search_qps,
            insert_qps=insert_qps,
            total_queries=len(self._search_latencies),
            total_inserts=len(self._insert_latencies),
            runtime_seconds=runtime,
            dataset_size=dataset_size,
            search_params=search_params,
            concurrent_searchers=self._concurrent_searchers,
            concurrent_inserters=self._concurrent_inserters,
            benchmark_mode=self._benchmark_mode
        )