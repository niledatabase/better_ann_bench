# PostgreSQL + pgvector Benchmark

This guide shows how to run the vector benchmark using PostgreSQL with the pgvector extension.

## Prerequisites

1. **Docker and Docker Compose**: Required to run PostgreSQL with pgvector
2. **Python 3.8+**: Required to run the benchmark
3. **Python dependencies**: Install with `pip install -r requirements.txt`

## Quick Start

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start PostgreSQL with pgvector**:
   ```bash
   ./setup_pgvector.sh
   ```

3. **Run a benchmark**:
   ```bash
   python main.py --config configs/benchmark_10k.yaml --algorithm configs/algorithm_pgvector.yaml
   ```

## Configuration

### Database Connection

The PostgreSQL connection parameters are configured in `configs/algorithm_pgvector.yaml`:

```yaml
build_params:
  connection_params:
    host: "localhost"
    port: 5432
    database: "vector_benchmark"
    user: "postgres"
    password: "postgres"
```

### pgvector Parameters

The pgvector implementation supports the following parameters:

**Build Parameters**:
- `batch_size`: Number of vectors to insert in each batch (default: 1000)
- `m`: HNSW index parameter for number of connections per layer (default: 16)
- `ef_construction`: HNSW index parameter for search depth during construction (default: 64)

**Search Parameters**:
- `ef`: HNSW search parameter for search depth (default: 100)

## Available Benchmarks

The following benchmark configurations are available:

- `configs/benchmark_10k.yaml`: 10,000 vectors
- `configs/benchmark_1m.yaml`: 1,000,000 vectors  
- `configs/benchmark_10m.yaml`: 10,000,000 vectors
- `configs/benchmark_100m.yaml`: 100,000,000 vectors

## Performance Tuning

### Database Tuning

For better performance, consider adjusting PostgreSQL parameters:

```sql
-- Increase shared buffers
ALTER SYSTEM SET shared_buffers = '256MB';

-- Increase work memory for complex queries
ALTER SYSTEM SET work_mem = '64MB';

-- Increase maintenance work memory
ALTER SYSTEM SET maintenance_work_mem = '256MB';

-- Reload configuration
SELECT pg_reload_conf();
```

### pgvector Index Tuning

The HNSW index parameters can be tuned for your specific use case:

- **Higher recall**: Increase `ef` and `ef_construction`
- **Faster search**: Decrease `ef` and `m`
- **Faster build**: Decrease `ef_construction` and `m`

## Troubleshooting

### Connection Issues

If you can't connect to PostgreSQL:

1. Check if the container is running:
   ```bash
   docker compose ps
   ```

2. Check the logs:
   ```bash
   docker compose logs postgres
   ```

3. Restart the container:
   ```bash
   docker compose restart
   ```

### Performance Issues

1. **Slow inserts**: Increase `batch_size` in the algorithm config
2. **Low recall**: Increase `ef` parameter in search_params
3. **Slow search**: Decrease `ef` parameter or increase `m` parameter

### Memory Issues

If you encounter memory issues with large datasets:

1. Increase Docker memory limits
2. Use smaller batch sizes
3. Consider using a smaller dataset for testing

## Cleanup

To stop PostgreSQL and clean up:

```bash
docker compose down -v
```

This will remove the PostgreSQL container and all data. 