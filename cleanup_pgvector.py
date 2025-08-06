#!/usr/bin/env python3
"""
Cleanup script for pgvector database.
Drops the vectors table and all associated indexes.
"""

import yaml
import psycopg2
import argparse
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def cleanup_pgvector(config_path: str, table_name: str = None, index_name: str = None):
    """Drop the pgvector table and indexes"""
    
    # Load configuration
    config = load_config(config_path)
    build_params = config['build_params']
    connection_params = build_params['connection_params']
    
    # Use provided names or defaults from config
    table_name = table_name or build_params.get('table_name', 'vectors')
    index_name = index_name or build_params.get('index_name', 'vectors_hnsw_idx')
    tenant_index_name = f"{table_name}_tenant_idx"
    
    print(f"Connecting to database: {connection_params['database']} on {connection_params['host']}")
    print(f"Table to drop: {table_name}")
    print(f"Indexes to drop: {index_name}, {tenant_index_name}")
    
    try:
        # Connect to database with timeout
        connection_params['connect_timeout'] = 5  # 5 second connection timeout
        conn = psycopg2.connect(**connection_params)
        conn.autocommit = True
        
        with conn.cursor() as cursor:
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (table_name,))
            
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                print(f"Table '{table_name}' does not exist. Nothing to clean up.")
                return
            
            # Drop indexes first (if they exist)
            print(f"Dropping index: {index_name}")
            cursor.execute(f"DROP INDEX IF EXISTS {index_name}")
            
            print(f"Dropping index: {tenant_index_name}")
            cursor.execute(f"DROP INDEX IF EXISTS {tenant_index_name}")
            
            # Drop the table
            print(f"Dropping table: {table_name}")
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            print("Cleanup completed successfully!")
            
    except Exception as e:
        print(f"Error during cleanup: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()


def main():
    parser = argparse.ArgumentParser(description='Cleanup pgvector table and indexes')
    parser.add_argument('--config', 
                       default='configs/algorithm_pgvector.yaml',
                       help='Path to algorithm config file (default: configs/algorithm_pgvector.yaml)')
    parser.add_argument('--table', 
                       help='Table name to drop (default: from config)')
    parser.add_argument('--index', 
                       help='Index name to drop (default: from config)')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"Error: Config file '{args.config}' not found.")
        return 1
    
    try:
        cleanup_pgvector(args.config, args.table, args.index)
        return 0
    except Exception as e:
        print(f"Cleanup failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 