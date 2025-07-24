import numpy as np
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
        self._inserted_vectors = []
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
    
    def _ensure_table_exists(self, dimension: int):
        """Create the vectors table if it doesn't exist"""
        conn = self._get_connection()
        with conn.cursor() as cursor:
            # Create table if not exists
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER,
                    tenant_id UUID NOT NULL,
                    vector vector({dimension}),
                    original_index INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (tenant_id, id)
                )
            """)
    
    def _create_hnsw_index(self):
        """Create the HNSW index after table is populated"""
        conn = self._get_connection()
        with conn.cursor() as cursor:
            print(f"Creating HNSW index '{self.index_name}' (m={self.build_params.get('m', 16)}, ef_construction={self.build_params.get('ef_construction', 64)})")
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.index_name} 
                ON {self.table_name} 
                USING hnsw (vector vector_l2_ops)
                WITH (m = {self.build_params.get('m', 16)}, 
                     ef_construction = {self.build_params.get('ef_construction', 64)})
            """)
            print(f"HNSW index '{self.index_name}' created successfully")
    
    def _verify_existing_table(self) -> None:
        """Verify that the existing table exists and has the correct structure"""
        conn = self._get_connection()
        with conn.cursor() as cursor:
            # Check if table exists and get its dimension
            cursor.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = %s AND column_name = 'vector'
            """, (self.table_name,))
            
            result = cursor.fetchone()
            if not result:
                raise RuntimeError(f"Table '{self.table_name}' does not exist or has no vector column")
            
            # Get the dimension from the vector column type
            cursor.execute(f"""
                SELECT vector[1:1] IS NOT NULL as has_vectors
                FROM {self.table_name} 
                WHERE tenant_id = %s 
                LIMIT 1
            """, (self.tenant_id,))
            
            result = cursor.fetchone()
            if not result:
                raise RuntimeError(f"No vectors found in table '{self.table_name}' for tenant '{self.tenant_id}'")
            
            # Get the actual dimension by examining a vector
            cursor.execute(f"""
                SELECT array_length(vector, 1) as dimension
                FROM {self.table_name} 
                WHERE tenant_id = %s 
                LIMIT 1
            """, (self.tenant_id,))
            
            result = cursor.fetchone()
            if result:
                self.dimension = result[0]
                print(f"Using existing table with dimension {self.dimension}")
            else:
                raise RuntimeError(f"Could not determine vector dimension from table '{self.table_name}'")
    
    def _initialize_existing_data_state(self) -> None:
        """Initialize internal state based on existing data in the table"""
        conn = self._get_connection()
        with conn.cursor() as cursor:
            # Get count of existing vectors for this tenant
            cursor.execute(f"""
                SELECT COUNT(*) as vector_count
                FROM {self.table_name} 
                WHERE tenant_id = %s
            """, (self.tenant_id,))
            
            result = cursor.fetchone()
            if result:
                existing_count = result[0]
                print(f"Found {existing_count} existing vectors in table '{self.table_name}'")
                # Initialize _inserted_vectors to reflect existing data
                self._inserted_vectors = list(range(existing_count))
                self._id_counter = existing_count
            else:
                print("Warning: Could not determine existing vector count")
    
    def _build_with_existing_table(self) -> None:
        """Build index using existing table data"""
        print(f"Reusing existing table '{self.table_name}' - skipping data loading and index creation")
        self._verify_existing_table()
        self._initialize_existing_data_state()
    
    def _build_with_new_data(self, vectors: np.ndarray) -> None:
        """Build index by inserting new data and creating index"""
        self.dimension = vectors.shape[1]
        self._ensure_table_exists(self.dimension)
        
        conn = self._get_connection()
        with conn.cursor() as cursor:
            # Insert vectors in batches
            batch_size = self.build_params.get('batch_size', 1000)
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                
                for i, vector in enumerate(batch):
                    vector_list = vector.tolist()
                    with self.lock:
                        # original_index should be the position in the training set
                        original_index = len(self._inserted_vectors)
                        db_id = self._id_counter + 1
                        self._id_counter += 1
                    
                    cursor.execute(
                        f"INSERT INTO {self.table_name} (id, tenant_id, vector, original_index) VALUES (%s, %s, %s::vector, %s)",
                        (db_id, self.tenant_id, vector_list, original_index)
                    )
                    self._inserted_vectors.append(original_index)
                
                if i % (batch_size * 10) == 0:
                    print(f"Inserted {i + len(batch)} vectors")
            
            # Create HNSW index after all vectors are inserted
            self._create_hnsw_index()
    
    def build(self, vectors: np.ndarray) -> None:
        with self.lock:
            if self.reuse_table:
                self._build_with_existing_table()
            else:
                self._build_with_new_data(vectors)
    
    def _insert_with_reuse_table(self) -> None:
        """Handle insert operation when reusing existing table"""
        print(f"Skipping insert operation - reusing existing table '{self.table_name}'")
    
    def _insert_with_new_data(self, vectors: np.ndarray) -> None:
        """Handle insert operation with new data"""
        if self.dimension is None:
            self.dimension = vectors.shape[1]
            self._ensure_table_exists(self.dimension)
        
        conn = self._get_connection()
        with conn.cursor() as cursor:
            for i, vector in enumerate(vectors):
                vector_list = vector.tolist()
                with self.lock:
                    original_index = len(self._inserted_vectors)
                    db_id = self._id_counter + 1
                    self._id_counter += 1
                
                cursor.execute(
                    f"INSERT INTO {self.table_name} (id, tenant_id, vector, original_index) VALUES (%s, %s, %s::vector, %s)",
                    (db_id, self.tenant_id, vector_list, original_index)
                )
                self._inserted_vectors.append(original_index)
    
    def insert(self, vectors: np.ndarray) -> None:
        with self.lock:
            if self.reuse_table:
                self._insert_with_reuse_table()
            else:
                self._insert_with_new_data(vectors)
    
    def search(self, query: np.ndarray, k: int) -> np.ndarray:
        # Initialize thread-local connection and cursor for this worker (only once per thread)
        if not hasattr(self._thread_local, 'connection'):
            try:
                self._thread_local.connection = psycopg2.connect(**self.connection_params)
                self._thread_local.connection.autocommit = True
                self._thread_local.cursor = self._thread_local.connection.cursor()
                self._thread_local.cursor.execute(f"SET nile.tenant_id = '{self.tenant_id}';")
            except Exception as e:
                raise RuntimeError(f"Failed to connect to PostgreSQL: {e}")
        
        if self.dimension is None:
            raise RuntimeError("Index not built yet")
        
        cursor = self._thread_local.cursor
        query_vector = query.tolist()
        # Convert to string format for vector casting
        query_vector_str = str(query_vector).replace(' ', '')
        
        cursor.execute(f"""
            SELECT original_index 
            FROM {self.table_name} 
            WHERE tenant_id = %s
            ORDER BY vector <-> %s::vector 
            LIMIT %s
        """, (self.tenant_id, query_vector_str, k))
        
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