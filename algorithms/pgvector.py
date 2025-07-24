import numpy as np
import threading
import psycopg2
import psycopg2.extras
from typing import Dict, Any, Optional
from concurrent_benchmark import VectorIndex


class PgVector(VectorIndex):
    def __init__(self, build_params: Dict[str, Any], search_params: Dict[str, Any]):
        self.build_params = build_params
        self.search_params = search_params
        self.connection_params = build_params.get('connection_params', {})
        self.table_name = build_params.get('table_name', 'vectors')
        self.index_name = build_params.get('index_name', 'vectors_hnsw_idx')
        self.tenant_id = build_params.get('tenant_id', '550e8400-e29b-41d4-a716-446655440000')
        self.dimension = None
        self.lock = threading.RLock()
        self._connection = None
        self._inserted_vectors = []
        self._id_counter = 0
        
    def _get_connection(self):
        """Get a database connection with proper error handling"""
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
    
    def build(self, vectors: np.ndarray) -> None:
        with self.lock:
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
    
    def insert(self, vectors: np.ndarray) -> None:
        with self.lock:
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
    
    def search(self, query: np.ndarray, k: int) -> np.ndarray:
        with self.lock:
            if self.dimension is None:
                raise RuntimeError("Index not built yet")
            
            conn = self._get_connection()
            with conn.cursor() as cursor:
                query_vector = query.tolist()
                # Convert to string format for vector casting
                query_vector_str = str(query_vector).replace(' ', '')
                
                # Use HNSW search with ef parameter for better recall
                ef = self.search_params.get('ef', 100)
                
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
        """Clean up database connection"""
        if self._connection and not self._connection.closed:
            self._connection.close() 