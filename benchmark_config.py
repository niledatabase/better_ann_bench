from dataclasses import dataclass
from typing import Dict, Any, Optional
import yaml


@dataclass
class DatasetGenerationConfig:
    enabled: bool = False
    dimension: int = 128
    clusters: int = 100
    cluster_std: float = 0.1
    correlation: float = 0.5
    noise_level: float = 0.1
    k_neighbors: int = 100
    seed: int = 42


@dataclass
class BenchmarkConfig:
    # Dataset configuration
    dataset_size: int  # Absolute size, not scaled
    query_count: int   # Absolute count, not scaled
    dataset_path: Optional[str] = None  # If None, generate synthetic data
    dataset_generation: Optional[DatasetGenerationConfig] = None
    save_dataset_path: Optional[str] = None  # Path to save generated dataset
    
    # Concurrency settings
    concurrent_inserters: int = 4
    concurrent_searchers: int = 8
    search_batch_size: int = 10
    insert_batch_size: int = 100
    
    # Runtime settings
    max_runtime_seconds: int = 300
    
    # Benchmark mode
    benchmark_mode: str = "search_only"  # "search_only" or "hybrid"
    
    # Hybrid mode settings
    initial_build_percentage: int = 10  # Percentage of dataset to build initially
    initial_build_size_cap: int = 1000  # Maximum initial build size
    
    # Output settings
    output_path: str = "results/"
    recall_targets: list[float] = None
    
    def __post_init__(self):
        if self.recall_targets is None:
            self.recall_targets = [0.9, 0.95, 0.99]
            
        if self.dataset_path is None and self.dataset_generation is None:
            # Default to synthetic data generation
            self.dataset_generation = DatasetGenerationConfig(enabled=True)
    
    @property
    def use_synthetic_data(self) -> bool:
        # If dataset_path is provided, use existing dataset
        if self.dataset_path is not None:
            return False
        
        # If dataset_generation is enabled, use synthetic data
        if self.dataset_generation is not None:
            if isinstance(self.dataset_generation, dict):
                return self.dataset_generation.get('enabled', False)
            return self.dataset_generation.enabled
        
        # Default to synthetic data if no dataset_path provided
        return True
    
    @classmethod
    def from_yaml(cls, path: str) -> 'BenchmarkConfig':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)


@dataclass
class AlgorithmConfig:
    name: str
    implementation: str
    build_params: Dict[str, Any]
    search_params: Dict[str, Any]
    supports_concurrent_insert: bool = True
    supports_incremental_build: bool = True