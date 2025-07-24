# Vector Search Benchmark

A comprehensive benchmark for vector search algorithms, currently supporting PostgreSQL with pgvector via Nile Cloud.

## Features

- **Multi-tenant support**: Tenant-aware tables for realistic workloads
- **Multiple algorithms**: Currently only supports Nile-pgvector, but it is designed to be extensible to other vector search algorithms.
- **Synthetic datasets**: Generate clustered vector data for testing
- **Two benchmark modes**: Search-only and hybrid (concurrent inserts + searches)
- **Comprehensive metrics**: Recall, QPS, latency percentiles
- **Configurable initial batch size**: For hybrid mode optimization
- **Automatic results saving**: Timestamped JSON files

## Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension (tested with Nile and pgvector 0.8.0)
  
### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run a small benchmark
python main.py --config configs/benchmark_10k.yaml --algorithm configs/algorithm_pgvector.yaml

# Run a larger benchmark
python main.py --config configs/benchmark_1m.yaml --algorithm configs/algorithm_pgvector.yaml
```

## Configuration System

The benchmark uses a **YAML-based configuration system** with three main components:

### 1. Benchmark Configs (`configs/benchmark_*.yaml`)

Define dataset size, concurrency, runtime, and dataset generation parameters:

```yaml
# Dataset settings
dataset_size: 10000
query_count: 1000
dataset_path: null  # Use synthetic data

# Dataset generation (when dataset_path is null)
dataset_generation:
  enabled: true
  dimension: 128
  clusters: 5
  cluster_std: 0.2
  k_neighbors: 100
  seed: 42

# Concurrency settings
concurrent_inserters: 2
concurrent_searchers: 4
search_batch_size: 10
insert_batch_size: 100

# Runtime settings
max_runtime_seconds: 60
warmup_seconds: 5
benchmark_mode: "search_only"  # "search_only" or "hybrid"

# Hybrid mode settings (used when benchmark_mode is "hybrid")
initial_build_percentage: 10  # Percentage of dataset to build initially
initial_build_size_cap: 1000  # Maximum initial build size

# Output settings
output_path: "results/"
recall_targets: [0.9, 0.95, 0.99]
```

### 2. Algorithm Configs (`configs/algorithm_*.yaml`)

Define algorithm-specific parameters:

```yaml
name: "Nile-pgvector"
type: "pgvector"
connection_string: "postgres://..."
table_name: "vectors"
tenant_id: "0197f6a3-3337-775d-83cd-f5da818c10b8"
```

### 3. Available Configurations

| Config | Dataset Size | Use Case |
|--------|-------------|----------|
| `benchmark_10k.yaml` | 10K vectors | Quick testing |
| `benchmark_1m.yaml` | 1M vectors | Medium-scale testing |
| `benchmark_10m.yaml` | 10M vectors | Large-scale testing |
| `benchmark_100m.yaml` | 100M vectors | Very large-scale testing |
| `benchmark_10k_hybrid.yaml` | 10K vectors | Hybrid mode testing |

## Benchmark Modes

### Search-Only Mode

The entire dataset is indexed upfront, then only search operations are performed. This is useful for:

- Testing pure search performance
- Comparing different index configurations
- Measuring recall accuracy

```yaml
benchmark_mode: "search_only"
```

### Hybrid Mode

An initial subset of vectors is indexed, then inserts and searches run concurrently. This is useful for:

- Testing real-world scenarios with ongoing inserts
- Measuring performance under load
- Testing index maintenance overhead

```yaml
benchmark_mode: "hybrid"
initial_build_percentage: 10  # 10% of dataset built initially
initial_build_size_cap: 1000  # Maximum initial build size
```

## Available Algorithms

Currently only supports Nile-pgvector, but it is designed to be extensible to other vector search algorithms.

### Nile-pgvector

- **Type**: PostgreSQL with pgvector extension
- **Algorithm**: HNSW (Hierarchical Navigable Small Worlds)
- **Features**: Multi-tenant support, tenant-aware tables
- **Configuration**: See `configs/algorithm_pgvector.yaml`

## Dataset Generation and Management

The benchmark can generate synthetic datasets inline or load existing datasets from files.

### Generating and Saving Datasets

To generate a dataset and save it for reuse:

```yaml
# Generate synthetic data
dataset_path: null  # Use synthetic data

# Save the generated dataset
save_dataset_path: "datasets/synthetic_10k.h5"

dataset_generation:
  enabled: true
  dimension: 128
  clusters: 100
  cluster_std: 0.15
  k_neighbors: 100
  seed: 42
```

### Loading Existing Datasets

To use a previously saved dataset:

```yaml
# Load existing dataset
dataset_path: "datasets/synthetic_10k.h5"

# Dataset generation is ignored when dataset_path is provided
dataset_generation:
  enabled: false
```

### Dataset Formats

The benchmark supports two dataset formats:

1. **HDF5 format** (recommended): `.h5` files with train/test/neighbors datasets
2. **NumPy format**: `.npz` files with train/test/neighbors arrays

### Available Sample Configs

| Config | Purpose |
|--------|---------|
| `benchmark_10k_save.yaml` | Generate and save 10K dataset |
| `benchmark_10k_load.yaml` | Load saved 10K dataset |

## Metrics

The benchmark measures:

- **Recall**: Percentage of ground truth nearest neighbors found
- **QPS**: Queries per second for search operations
- **Latency**: P50, P95, P99, mean, min, max latencies
- **Insert Performance**: QPS and latency for insert operations (hybrid mode)

## Results

Results are automatically saved to JSON files with timestamps:

```text
results/
├── nile_pgvector_20250723_155003.json
├── nile_pgvector_20250723_155311.json
└── ...
```

## Examples

### Small Dataset Benchmark

```bash
python main.py --config configs/benchmark_10k.yaml --algorithm configs/algorithm_pgvector.yaml
```

### Large Dataset Benchmark

```bash
python main.py --config configs/benchmark_1m.yaml --algorithm configs/algorithm_pgvector.yaml
```

### Generate and Save Dataset

```bash
# Generate dataset and save it for reuse
python main.py --config configs/benchmark_10k_save.yaml --algorithm configs/algorithm_pgvector.yaml
```

### Load Saved Dataset

```bash
# Use previously saved dataset (faster startup)
python main.py --config configs/benchmark_10k_load.yaml --algorithm configs/algorithm_pgvector.yaml
```

### Hybrid Benchmark with Custom Initial Batch

```bash
# 25% initial build, higher cap
python main.py --config configs/benchmark_10k_hybrid_large.yaml --algorithm configs/algorithm_pgvector.yaml
```

## Adding New Algorithms

To add a new vector search algorithm:

1. Create a new class implementing the `VectorIndex` protocol
2. Add algorithm configuration in `configs/algorithm_*.yaml`
3. Update `main.py` to load the new algorithm

## Troubleshooting

### Common Issues

1. **Database connection errors**: Check your connection string in `algorithm_pgvector.yaml`
2. **Duplicate table errors**: Drop existing tables before running: `DROP TABLE IF EXISTS vectors;`
3. **Low recall**: Ensure ground truth is computed correctly using brute force search
4. **Memory issues**: Reduce dataset size or use chunked processing

### Debugging

- Check logs for detailed error messages
- Verify database connectivity
- Ensure pgvector extension is installed
- Check tenant ID configuration for Nile Cloud

## Database Cleanup

After running benchmarks, you may want to clean up the database tables and indexes. A cleanup script is provided for this purpose:

### Using the Cleanup Script

```bash
# Basic cleanup (uses default config)
python cleanup_pgvector.py

# With custom config file
python cleanup_pgvector.py --config path/to/your/config.yaml

# With custom table/index names
python cleanup_pgvector.py --table my_vectors --index my_index
```

### What the Cleanup Script Does

The script will:
1. Connect to the database using connection parameters from the config
2. Drop the HNSW index (`vectors_hnsw_idx`)
3. Drop the tenant index (`vectors_tenant_idx`)
4. Drop the vectors table (`vectors`)

### Manual Cleanup

If you prefer to clean up manually via SQL:

```sql
-- Drop indexes first
DROP INDEX IF EXISTS vectors_hnsw_idx;
DROP INDEX IF EXISTS vectors_tenant_idx;

-- Drop the table
DROP TABLE IF EXISTS vectors;
``` 