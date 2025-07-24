from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum
import yaml


class DatasetMode(Enum):
    """Enum representing different dataset handling modes"""
    LOAD_FROM_FILE = "load_from_file"  # Use existing dataset file
    GENERATE_SYNTHETIC = "generate_synthetic"  # Generate synthetic data

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
    def dataset_mode(self) -> DatasetMode:
        """Determine the dataset mode based on configuration"""
        #print(f"dataset_path: {self.dataset_path}")
        if self.dataset_path is not None:
            return DatasetMode.LOAD_FROM_FILE
        else:
            return DatasetMode.GENERATE_SYNTHETIC
    
    @property
    def use_synthetic_data(self) -> bool:
        # If dataset_path is provided, use existing dataset
        if self.dataset_path is not None:
            return False
        
        # If dataset_generation is enabled, use synthetic data
        if self.dataset_generation is not None:
            # Convert dict to DatasetGenerationConfig if needed
            if isinstance(self.dataset_generation, dict):
                gen_config = DatasetGenerationConfig(**self.dataset_generation)
            else:
                gen_config = self.dataset_generation
            return gen_config.enabled
        
        # Default to synthetic data if no dataset_path provided
        return True
    
    def validate_configuration(self) -> None:
        """Validate configuration for potential issues"""
        warnings = []
        
        # Check for redundant dataset_generation when dataset_path is specified
        if self.dataset_path is not None and self.dataset_generation is not None:
            warnings.append("dataset_generation is specified but will be ignored since dataset_path is provided")
        
        # Check for missing dataset_generation when no dataset_path is provided
        if self.dataset_path is None and self.dataset_generation is None:
            warnings.append("No dataset_path provided and no dataset_generation config - will use default synthetic data generation")
        
        # Check for potentially problematic synthetic data parameters
        if self.dataset_generation is not None:
            # Convert dict to DatasetGenerationConfig if needed
            if isinstance(self.dataset_generation, dict):
                gen_config = DatasetGenerationConfig(**self.dataset_generation)
            else:
                gen_config = self.dataset_generation
                
            if gen_config.dimension <= 0:
                warnings.append("dataset_generation.dimension should be positive")
            if gen_config.clusters <= 0:
                warnings.append("dataset_generation.clusters should be positive")
            if gen_config.cluster_std <= 0:
                warnings.append("dataset_generation.cluster_std should be positive")
            if gen_config.k_neighbors <= 0:
                warnings.append("dataset_generation.k_neighbors should be positive")
        
        # Print warnings if any
        if warnings:
            print("Configuration warnings:")
            for warning in warnings:
                print(f"  ⚠️  {warning}")
            print()
    
    @classmethod
    def from_yaml(cls, path: str) -> 'BenchmarkConfig':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        config = cls(**data)
        config.validate_configuration()
        return config
    
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