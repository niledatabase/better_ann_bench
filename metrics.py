import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional
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
    recall: float
    search_latencies: LatencyMetrics
    insert_latencies: Optional[LatencyMetrics]
    search_qps: float
    insert_qps: float
    total_queries: int
    total_inserts: int
    runtime_seconds: float
    dataset_size: int


class MetricsCollector:
    def __init__(self):
        self._search_latencies: List[float] = []
        self._insert_latencies: List[float] = []
        self._search_times: List[float] = []
        self._insert_times: List[float] = []
        self._lock = threading.Lock()
        self._start_time = None
        self._end_time = None
        
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
    
    def get_results(self, recall: float, dataset_size: int) -> BenchmarkResults:
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
            dataset_size=dataset_size
        )