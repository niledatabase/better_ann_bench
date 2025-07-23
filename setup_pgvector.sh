#!/bin/bash

echo "Setting up PostgreSQL with pgvector for vector benchmark..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker compose is available
if ! docker compose version &> /dev/null; then
    echo "Error: docker compose is not available. Please install Docker Desktop or docker-compose."
    exit 1
fi

# Start PostgreSQL with pgvector
echo "Starting PostgreSQL with pgvector..."
docker compose up -d

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
sleep 10

# Check if PostgreSQL is running
if docker compose ps | grep -q "Up"; then
    echo "PostgreSQL with pgvector is running!"
    echo "Connection details:"
    echo "  Host: localhost"
    echo "  Port: 5432"
    echo "  Database: vector_benchmark"
    echo "  Username: postgres"
    echo "  Password: postgres"
    echo ""
    echo "You can now run the benchmark with:"
    echo "python main.py --config configs/benchmark_10k.yaml --algorithm configs/algorithm_pgvector.yaml"
else
    echo "Error: Failed to start PostgreSQL. Check the logs with: docker-compose logs"
    exit 1
fi 