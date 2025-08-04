import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union
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
    # Ground truth computation block sizes (for memory management) - optional with defaults
    gt_train_block_size: int = 50000  # Training vectors per block for ground truth computation
    gt_query_block_size: int = 1000  # Query vectors per block for ground truth computation
    # Memory management
    chunk_size: int = 10000  # Generate training vectors in chunks to manage memory
    query_chunk_size: int = 1000  # Generate query vectors in chunks to manage memory
    # Dataset generation parameters
    seed: int = 42
    clusters: int = 10  # For clustered data
    cluster_std: float = 0.1  # Standard deviation within clusters
    k_neighbors: int = 100  # Number of true neighbors to compute
    skip_ground_truth: bool = False  # Skip ground truth computation for large datasets
    
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
    
    def _generate_clustered_chunk(self, size: int, cluster_centers: np.ndarray) -> np.ndarray:
        """Generate a chunk of clustered vectors using pre-computed cluster centers"""
        # Assign vectors to clusters
        cluster_assignments = self.rng.randint(0, self.config.clusters, size)
        
        # Generate vectors around cluster centers
        vectors = np.zeros((size, self.config.dimension), dtype=np.float32)
        
        for i in range(size):
            cluster_idx = cluster_assignments[i]
            center = cluster_centers[cluster_idx]
            # Add Gaussian noise around the cluster center
            noise = self.rng.randn(self.config.dimension) * self.config.cluster_std
            vectors[i] = center + noise
        
        return vectors


    def _compute_ground_truth(self, train_vectors: np.ndarray, query_vectors: np.ndarray) -> np.ndarray:
        """Compute ground truth nearest neighbors using brute force for small datasets"""
        print(f"Computing ground truth for {len(query_vectors)} queries and {len(train_vectors)} training vectors using brute force.")
        
        from sklearn.neighbors import NearestNeighbors
        
        # Use brute force for exact neighbors
        nn = NearestNeighbors(
            n_neighbors=self.config.k_neighbors,
            algorithm='brute',
            metric='euclidean',
            n_jobs=-1
        )
        nn.fit(train_vectors)
        
        # For small datasets, process all queries at once
        _, neighbors = nn.kneighbors(query_vectors)
        
        return neighbors
    

    
    def _compute_ground_truth_blockwise_hdf5(self, hdf5_path: str) -> np.ndarray:
        """Memory-efficient ground truth computation using HDF5 files"""
        
        with h5py.File(hdf5_path, 'r') as f:
            train_dataset = f['train']
            query_dataset = f['test']
            
            n = train_dataset.shape[0]
            nq = query_dataset.shape[0]
            dim = train_dataset.shape[1]
            k = self.config.k_neighbors
        
        print(f"Computing ground truth for {nq:,} queries using blockwise approach...")
        
        gt_idx = np.empty((nq, k), dtype=np.int32)
            
        # Precompute training vector norms
        print("Precomputing training vector norms...")
        train_norm2 = np.empty(n, dtype=np.float64)
        train_block_rows = self.config.gt_train_block_size
            
        for t0 in range(0, n, train_block_rows):
            t1 = min(t0 + train_block_rows, n)
            block = train_dataset[t0:t1].astype(np.float64, copy=False)
            train_norm2[t0:t1] = np.einsum('ij,ij->i', block, block)
            print(f"  Processed training block {t0//train_block_rows + 1}/{(n + train_block_rows - 1)//train_block_rows}")
        
        # Process all queries against training blocks
        print(f"Computing ground truth for {nq} queries...")
        q_block_rows = self.config.gt_query_block_size
            
        for q0 in range(0, nq, q_block_rows):
            q1 = min(q0 + q_block_rows, nq)
            q = query_dataset[q0:q1].astype(np.float64, copy=False)
            q_norm2 = np.einsum('ij,ij->i', q, q)[:, None]
            
            bestD = np.full((q1 - q0, k), np.inf, dtype=np.float64)
            bestI = np.full((q1 - q0, k), -1, dtype=np.int32)
            
            for t0 in range(0, n, train_block_rows):
                t1 = min(t0 + train_block_rows, n)
                block = train_dataset[t0:t1].astype(np.float64, copy=False)
                dot = q @ block.T
                D = q_norm2 + train_norm2[t0:t1][None, :] - 2.0 * dot
                
                # Efficient top-K merging using argpartition
                idx_local = np.argpartition(D, kth=k-1, axis=1)[:, :k]
                vals_local = np.take_along_axis(D, idx_local, axis=1)
                idx_local += t0
                
                D_merge = np.concatenate([bestD, vals_local], axis=1)
                I_merge = np.concatenate([bestI, idx_local], axis=1)
                sel = np.argpartition(D_merge, kth=k-1, axis=1)[:, :k]
                bestD = np.take_along_axis(D_merge, sel, axis=1)
                bestI = np.take_along_axis(I_merge, sel, axis=1)
                
            gt_idx[q0:q1] = bestI
            print(f"  Completed query block {q0//q_block_rows + 1}/{(nq + q_block_rows - 1)//q_block_rows}")
            
        return gt_idx
    
    def generate_dataset(self, save_path: Optional[str] = None) -> Tuple[Union[np.ndarray, str], Union[np.ndarray, str], Optional[np.ndarray]]:
        """Generate complete dataset with train vectors, queries, and ground truth"""
        print(f"Generating {self.config.name} dataset:")
        print(f"  Size: {self.config.size:,} vectors")
        print(f"  Dimension: {self.config.dimension}")
        print(f"  Distribution: clustered")
        print(f"  Query size: {self.config.query_size:,}")
        
        start_time = time.time()
        
        # If save path is provided, generate and save in chunks to avoid memory issues
        if save_path:
            print("Using memory-efficient chunked generation...")
            return self._generate_dataset_chunked(save_path)
        
        # For small datasets, use the original approach
        if self.config.size <= 1000000:  # 1M vectors threshold
            print("Using standard generation for small dataset...")
            return self._generate_dataset_standard()
        else:
            print("Large dataset detected, using memory-efficient generation...")
            # Create a temporary file for large datasets
            temp_path = f"temp_dataset_{int(time.time())}.h5"
            try:
                result = self._generate_dataset_chunked(temp_path)
                # Don't clean up temp file when returning file paths
                # The file will be used by the benchmark
                return result
            except Exception as e:
                # Clean up temp file on error
                import os
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise e
    
    def _generate_dataset_standard(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Original generation method for small datasets"""
        start_time = time.time()
        
        # Generate training vectors
        print("Generating training vectors...")
        train_vectors = self._generate_clustered(self.config.size)

        # Generate query vectors
        print("Generating query vectors...")
        query_vectors = self._generate_clustered(self.config.query_size)

        # Compute ground truth using brute force search
        if self.config.skip_ground_truth:
            print("Skipping ground truth computation as requested...")
            ground_truth = None
        else:
            print("Computing ground truth using brute force search...")
            ground_truth = self._compute_ground_truth(train_vectors, query_vectors)
        
        generation_time = time.time() - start_time
        print(f"Dataset generation completed in {generation_time:.2f} seconds")
        
        return train_vectors, query_vectors, ground_truth
    
    def _generate_dataset_chunked(self, save_path: str) -> Tuple[Union[np.ndarray, str], Union[np.ndarray, str], Optional[np.ndarray]]:
        """Memory-efficient generation using chunks"""
        start_time = time.time()
        
        print("Generating dataset in chunks to manage memory...")
        
        # Create HDF5 file with chunked datasets
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(save_path, 'w') as f:
            # Create datasets with chunking for memory efficiency
            train_dataset = f.create_dataset(
                'train', 
                shape=(self.config.size, self.config.dimension), 
                dtype=np.float32,
                chunks=(self.config.chunk_size, self.config.dimension),
                compression='gzip'
            )
            
            test_dataset = f.create_dataset(
                'test', 
                shape=(self.config.query_size, self.config.dimension), 
                dtype=np.float32,
                chunks=(self.config.query_chunk_size, self.config.dimension),
                compression='gzip'
            )
            
            # Generate training vectors in chunks with proper clustering
            print("Generating training vectors in chunks...")
            
            # Pre-generate cluster centers to ensure consistent distribution across chunks
            np.random.seed(self.config.seed)
            cluster_centers = np.random.randn(self.config.clusters, self.config.dimension).astype(np.float32)
            
            for i in range(0, self.config.size, self.config.chunk_size):
                end_idx = min(i + self.config.chunk_size, self.config.size)
                chunk_vectors = self._generate_clustered_chunk(end_idx - i, cluster_centers)
                train_dataset[i:end_idx] = chunk_vectors
                print(f"  Generated training chunk {i//self.config.chunk_size + 1}/{(self.config.size + self.config.chunk_size - 1)//self.config.chunk_size}")
            
            # Generate query vectors in chunks with same clustering
            print("Generating query vectors in chunks...")
            for i in range(0, self.config.query_size, self.config.query_chunk_size):
                end_idx = min(i + self.config.query_chunk_size, self.config.query_size)
                chunk_vectors = self._generate_clustered_chunk(end_idx - i, cluster_centers)
                test_dataset[i:end_idx] = chunk_vectors
                print(f"  Generated query chunk {i//self.config.query_chunk_size + 1}/{(self.config.query_size + self.config.query_chunk_size - 1)//self.config.query_chunk_size}")
            
            # Store metadata
            f.attrs['size'] = self.config.size
            f.attrs['dimension'] = self.config.dimension
            f.attrs['query_size'] = self.config.query_size
            f.attrs['distribution'] = 'clustered'
            f.attrs['k_neighbors'] = self.config.k_neighbors
            f.attrs['seed'] = self.config.seed
            f.attrs['has_ground_truth'] = False  # Will be updated after ground truth computation
        
        # Compute ground truth using memory-efficient method
        if self.config.skip_ground_truth:
            print("Skipping ground truth computation as requested...")
            ground_truth = None
        else:
            print("Computing ground truth using blockwise approach...")
            # Use memory-efficient blockwise computation with HDF5 files
            ground_truth = self._compute_ground_truth_blockwise_hdf5(save_path)
            
            # Save ground truth to the file
            with h5py.File(save_path, 'a') as f:
                f.create_dataset('neighbors', data=ground_truth, compression='gzip')
                f.attrs['has_ground_truth'] = True
        
        generation_time = time.time() - start_time
        print(f"Dataset generation completed in {generation_time:.2f} seconds")
        
        # Return file paths instead of loading into memory
        return save_path, save_path, ground_truth
    
    def _save_dataset(self, save_path: str, train_vectors: np.ndarray, 
                     query_vectors: np.ndarray, ground_truth: Optional[np.ndarray]):
        """Save dataset to HDF5 file"""
        print(f"Saving dataset to {save_path}")
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('train', data=train_vectors, compression='gzip')
            f.create_dataset('test', data=query_vectors, compression='gzip')
            
            # Only save ground truth if it exists
            if ground_truth is not None:
                f.create_dataset('neighbors', data=ground_truth, compression='gzip')
            
            # Store metadata
            f.attrs['size'] = self.config.size
            f.attrs['dimension'] = self.config.dimension
            f.attrs['query_size'] = self.config.query_size
            f.attrs['distribution'] = 'clustered'
            f.attrs['k_neighbors'] = self.config.k_neighbors
            f.attrs['seed'] = self.config.seed
            f.attrs['has_ground_truth'] = ground_truth is not None
    
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


