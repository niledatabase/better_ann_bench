import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import h5py
from pathlib import Path
import time


@dataclass
class DatasetConfig:
    name: str
    size: int
    dimension: int
    query_size: int
    seed: int = 42
    
    # Distribution-specific parameters
    clusters: int = 10  # For clustered data
    cluster_std: float = 0.1  # Standard deviation within clusters
    
    # Ground truth parameters
    k_neighbors: int = 100  # Number of true neighbors to compute
    
    # Memory management
    chunk_size: int = 10000  # Generate in chunks to manage memory
    
    def __post_init__(self):
        if self.query_size > self.size:
            self.query_size = min(self.query_size, self.size // 10)


class DatasetGenerator:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        
    def _generate_clustered(self, size: int) -> np.ndarray:
        """Generate clustered data with specified number of clusters"""
        # Generate cluster centers
        centers = self.rng.randn(self.config.clusters, self.config.dimension) * 2
        
        # Assign points to clusters
        cluster_assignments = self.rng.randint(0, self.config.clusters, size)
        
        # Generate points around cluster centers
        vectors = np.zeros((size, self.config.dimension), dtype=np.float32)
        for i in range(size):
            cluster_id = cluster_assignments[i]
            center = centers[cluster_id]
            noise = self.rng.randn(self.config.dimension) * self.config.cluster_std
            vectors[i] = center + noise
            
        return vectors
    
    # ------------------------------------------------------------------
    # Extensible wrapper, in case we want to add more distributions in the future
    # ------------------------------------------------------------------
    def _generate_vectors(self, size: int) -> np.ndarray:
        """Generate vectors (clustered only)."""
        return self._generate_clustered(size)


    def _compute_ground_truth(self, train_vectors: np.ndarray, query_vectors: np.ndarray) -> np.ndarray:
        """Compute ground truth nearest neighbors using brute force"""
        print(f"Computing ground truth for {len(query_vectors)} queries...")
        
        from sklearn.neighbors import NearestNeighbors
        
        # Use brute force for exact neighbors
        nn = NearestNeighbors(
            n_neighbors=self.config.k_neighbors,
            algorithm='brute',
            metric='euclidean',
            n_jobs=-1
        )
        nn.fit(train_vectors)
        
        # Compute in chunks to manage memory
        chunk_size = min(1000, len(query_vectors))
        all_neighbors = []
        
        for i in range(0, len(query_vectors), chunk_size):
            end_idx = min(i + chunk_size, len(query_vectors))
            chunk_queries = query_vectors[i:end_idx]
            
            _, neighbors = nn.kneighbors(chunk_queries)
            all_neighbors.append(neighbors)
            
            if (i // chunk_size) % 10 == 0:
                print(f"  Processed {end_idx}/{len(query_vectors)} queries")
        
        return np.vstack(all_neighbors)
    
    def generate_dataset(self, save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate complete dataset with train vectors, queries, and ground truth"""
        print(f"Generating {self.config.name} dataset:")
        print(f"  Size: {self.config.size:,} vectors")
        print(f"  Dimension: {self.config.dimension}")
        print(f"  Distribution: clustered")
        print(f"  Query size: {self.config.query_size:,}")
        
        start_time = time.time()
        
        # Generate training vectors
        print("Generating training vectors...")
        train_vectors = self._generate_clustered(self.config.size)

        # Generate query vectors
        print("Generating query vectors...")
        query_vectors = self._generate_clustered(self.config.query_size)

        # Compute ground truth using brute force search
        print("Computing ground truth using brute force search...")
        ground_truth = self._compute_ground_truth(train_vectors, query_vectors)
        
        generation_time = time.time() - start_time
        print(f"Dataset generation completed in {generation_time:.2f} seconds")
        
        # Save to file if path provided
        if save_path:
            self._save_dataset(save_path, train_vectors, query_vectors, ground_truth)
        
        return train_vectors, query_vectors, ground_truth
    
    def _save_dataset(self, save_path: str, train_vectors: np.ndarray, 
                     query_vectors: np.ndarray, ground_truth: np.ndarray):
        """Save dataset to HDF5 file"""
        print(f"Saving dataset to {save_path}")
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('train', data=train_vectors, compression='gzip')
            f.create_dataset('test', data=query_vectors, compression='gzip')
            f.create_dataset('neighbors', data=ground_truth, compression='gzip')
            
            # Store metadata
            f.attrs['size'] = self.config.size
            f.attrs['dimension'] = self.config.dimension
            f.attrs['query_size'] = self.config.query_size
            f.attrs['distribution'] = 'clustered'
            f.attrs['k_neighbors'] = self.config.k_neighbors
            f.attrs['seed'] = self.config.seed
    
    def generate_streaming(self, chunk_callback):
        """Generate dataset in chunks for memory efficiency"""
        print(f"Generating dataset in chunks of {self.config.chunk_size}")
        
        num_chunks = (self.config.size + self.config.chunk_size - 1) // self.config.chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.config.chunk_size
            end_idx = min(start_idx + self.config.chunk_size, self.config.size)
            chunk_size = end_idx - start_idx
            
            chunk_vectors = self._generate_vectors(chunk_size)
            chunk_callback(chunk_vectors, chunk_idx, num_chunks)
            
            print(f"Generated chunk {chunk_idx + 1}/{num_chunks}")


