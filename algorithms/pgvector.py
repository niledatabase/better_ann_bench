import numpy as np
import sys
import threading
import psycopg2
import psycopg2.extras
from typing import Dict, Any, Optional
from concurrent_benchmark import VectorIndex


class PgVector(VectorIndex):
    """
    PostgreSQL pgvector implementation with support for reusing existing tables.
    
    This implementation supports two modes:
    1. Normal mode (reuse_table=False): Insert vectors and create HNSW index
    2. Reuse mode (reuse_table=True): Use existing table data, skip insertion
    
    The reuse mode is useful for benchmarking search performance without the overhead
    of data loading and index creation.
    """
    def __init__(self, build_params: Dict[str, Any], search_params: Dict[str, Any]):
        self.build_params = build_params
        self.search_params = search_params
        self.connection_params = build_params.get('connection_params', {})
        self.table_name = build_params.get('table_name', 'vectors')
        self.index_name = build_params.get('index_name', 'vectors_hnsw_idx')
        self.tenant_id = build_params.get('tenant_id', '550e8400-e29b-41d4-a716-446655440000')
        self.reuse_table = build_params.get('reuse_table', False)
        self.dimension = None
        self.lock = threading.RLock()
        self._connection = None
        self._id_counter = 0
        # Thread-local storage for worker connections
        self._thread_local = threading.local()
        
    def _get_connection(self):
        """Get the shared database connection for build operations"""
        if self._connection is None or self._connection.closed:
            try:
                self._connection = psycopg2.connect(**self.connection_params)
                self._connection.autocommit = True
            except Exception as e:
                raise RuntimeError(f"Failed to connect to PostgreSQL: {e}")
        return self._connection
    
    def _create_table(self, dimension: int, cursor):
        """Create the vectors table  - it should fail if the table already exists"""
        print(f"Creating table '{self.table_name}' with dimension {dimension}")
        try:    
            cursor.execute(f"""
                CREATE TABLE {self.table_name} (
                    id INTEGER,
                    tenant_id UUID NOT NULL,
                    vector vector({dimension}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (tenant_id, id)
                )
            """)
        except psycopg2.errors.DuplicateTable:
            print(f"Table '{self.table_name}' already exists. You either forgot to use the cleanup script or failed to set the reuse_table flag to false")
            sys.exit(1)
        except Exception as e:
            print(f"Failed to create table '{self.table_name}': {e}")
            raise e
    
    def _create_hnsw_index(self, cursor):
        """Create the HNSW index after table is populated"""
        print(f"Creating HNSW index '{self.index_name}' (m={self.build_params.get('m', 16)}, ef_construction={self.build_params.get('ef_construction', 64)})")
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS {self.index_name} 
            ON {self.table_name} 
            USING hnsw (vector vector_l2_ops)
            WITH (m = {self.build_params.get('m', 16)}, 
                ef_construction = {self.build_params.get('ef_construction', 64)})
        """)
        print(f"HNSW index '{self.index_name}' created successfully")
    
    def _verify_existing_table(self, file_dimension: int) -> None:
        """Verify that the existing table exists and has the correct structure"""
        conn = self._get_connection()
        with conn.cursor() as cursor:
            try:
                # Get a vector to check its dimensions
                cursor.execute(f"""
                    SELECT vector 
                    FROM {self.table_name} 
                    WHERE tenant_id = %s 
                    LIMIT 1
                """, (self.tenant_id,))
                
                result = cursor.fetchone()
                if result is None:
                    raise RuntimeError(f"No vectors found in table '{self.table_name}' for tenant '{self.tenant_id}'")
                
                # Get the vector and check its dimensions

                from ast import literal_eval
                vector_data = literal_eval(result[0])
                print(f"Vector data: {vector_data}")
                print(f"Vector data size: {len(vector_data)}")
                if not hasattr(vector_data, '__len__'):
                    raise RuntimeError(f"Could not determine vector dimension from data in table '{self.table_name}'")
                if len(vector_data) != file_dimension:
                    raise RuntimeError(f"Vector dimension in table '{self.table_name}' ({len(vector_data)}) does not match the file dimension {file_dimension}")
                                
                self.dimension = file_dimension
                print(f"Verified table '{self.table_name}' exists and contains valid vectors with dimension {self.dimension}")
            except psycopg2.errors.UndefinedTable:
                raise RuntimeError(f"Table '{self.table_name}' does not exist")
    
    def _build_with_existing_table(self, vectors: np.ndarray) -> None:
        """Build index using existing table data"""
        print(f"Reusing existing table '{self.table_name}' - skipping data loading and index creation")
        file_dimension = vectors.shape[1]
        self._verify_existing_table(file_dimension)
    
    ### This is used for the initial data load and build. It assumes the table and index don't exist.
    ### There is a cleanup script that can be used to remove the table and index if needed.
    def _build_with_new_data(self, vectors: np.ndarray) -> None:
        """Build index by inserting new data and creating index"""
        self.dimension = vectors.shape[1]
        
        conn = self._get_connection()
        with conn.cursor() as cursor:
            # Create the table
            self._create_table(self.dimension, cursor)
            
            # Insert vectors using multi-row inserts (100 rows at a time)
            multi_row_size = 100
            
            for chunk_start in range(0, len(vectors), multi_row_size):
                chunk = vectors[chunk_start:chunk_start + multi_row_size]
                
                # Prepare multi-row insert data
                values_list = []
                for j, vector in enumerate(chunk):
                    vector_list = vector.tolist()
                    vector_index = chunk_start + j  # This ensures the ID is the exact index in the input array
                    values_list.append((vector_index, self.tenant_id, vector_list))
                
                # Execute multi-row insert
                args_str = ','.join(cursor.mogrify(
                    f"(%s, %s, %s::vector)", 
                    (id_val, tenant_val, vector_val)
                ).decode('utf-8') for id_val, tenant_val, vector_val in values_list)
                
                cursor.execute(f"INSERT INTO {self.table_name} (id, tenant_id, vector) VALUES {args_str}")
                
                print(f"Inserted {(chunk_start + len(chunk))} vectors")
            
            # Create HNSW index after all vectors are inserted
            self._create_hnsw_index(cursor)
    
    def build(self, vectors: np.ndarray) -> None:
        with self.lock:
            if self.reuse_table:
                self._build_with_existing_table()
            else:
                self._build_with_new_data(vectors)
    
    ### This is used by concurrent insert workers
    ### TODO: figure out what to do in case of table re-use. For now we prevent this combo in the config
    def insert(self, vectors: np.ndarray) -> None:        
        # For non-reuse mode, insert vectors
        if self.dimension is None:
            self.dimension = vectors.shape[1]
        
        conn = self._get_connection()
        with conn.cursor() as cursor:
            # Insert vectors using multi-row inserts (100 rows at a time)
            multi_row_size = 100
            
            for i in range(0, len(vectors), multi_row_size):
                chunk = vectors[i:i + multi_row_size]
                
                # Prepare multi-row insert data
                values_list = []
                for j, vector in enumerate(chunk):
                    vector_list = vector.tolist()
                    vector_index = i + j  # This ensures the ID is the exact index in the input array
                    values_list.append((vector_index, self.tenant_id, vector_list))
                
                # Execute multi-row insert
                args_str = ','.join(cursor.mogrify(
                    f"(%s, %s, %s::vector)", 
                    (id_val, tenant_val, vector_val)
                ).decode('utf-8') for id_val, tenant_val, vector_val in values_list)
                
                cursor.execute(f"INSERT INTO {self.table_name} (id, tenant_id, vector) VALUES {args_str}")
    
    def search(self, query: np.ndarray, k: int) -> np.ndarray:
        # Initialize thread-local connection and cursor for this worker (only once per thread)
        if not hasattr(self._thread_local, 'connection'):
            try:
                self._thread_local.connection = psycopg2.connect(**self.connection_params)
                self._thread_local.connection.autocommit = True
                self._thread_local.cursor = self._thread_local.connection.cursor()
                self._thread_local.cursor.execute(f"SET nile.tenant_id = '{self.tenant_id}';")
                ef_search = self.search_params.get('ef', 100)
                self._thread_local.cursor.execute(f"SET hnsw.ef_search = {ef_search};")
            except Exception as e:
                raise RuntimeError(f"Failed to connect to PostgreSQL: {e}")
        
        if self.dimension is None:
            raise RuntimeError("Index not built yet")
        
        cursor = self._thread_local.cursor

        query_vector = query.tolist()
        
        cursor.execute(f"""
            SELECT id
            FROM {self.table_name} 
            ORDER BY vector <-> %s::vector 
            LIMIT %s
        """, (query_vector, k))
        
        results = cursor.fetchall()
        return np.array([row[0] for row in results])
    
    def supports_concurrent_operations(self) -> bool:
        return True
    
    def __del__(self):
        """Clean up database connections"""
        # Clean up shared connection
        if self._connection and not self._connection.closed:
            self._connection.close()
        
        # Clean up thread-local connections and cursors
        if hasattr(self._thread_local, 'connection') and self._thread_local.connection:
            if hasattr(self._thread_local, 'cursor') and self._thread_local.cursor:
                self._thread_local.cursor.close()
            if not self._thread_local.connection.closed:
                self._thread_local.connection.close() 