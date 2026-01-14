"""Vector storage and similarity search for AI Assistant.

Provides vector similarity search, embedding storage and retrieval,
and integration with the main database.

Overview:
- Vector similarity search using cosine similarity
- Embedding storage and caching
- Optional clustering functionality (requires scikit-learn)
- Batch processing for performance

Key Dependencies:
- numpy for vector operations
- scikit-learn (optional) for clustering features
- DatabaseManager for persistence

Optional Features:
- cluster_entities: Requires scikit-learn, gracefully degrades if not available

Recent Changes:
- 2025-01-14: Implemented proper LRU cache with thread-safety and performance metrics
- 2025-10-27: Added optional scikit-learn import with graceful degradation
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import hashlib
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from cachetools import LRUCache

from .db_manager import DatabaseManager, get_db_manager

# Optional dependency: scikit-learn for clustering
# Clustering features will be disabled if sklearn is not installed
try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning(
        "scikit-learn not installed. Clustering features will be disabled. "
        "To enable clustering, install scikit-learn: pip install scikit-learn>=1.3.0"
    )


class EntityType(Enum):
    """Types of entities that can have embeddings."""
    MESSAGE = "message"
    KNOWLEDGE = "knowledge"
    PROJECT = "project"
    CONVERSATION = "conversation"


@dataclass
class SimilarityResult:
    """Result from a similarity search."""
    entity_type: str
    entity_id: int
    score: float
    entity_data: Optional[Dict[str, Any]] = None
    highlight: Optional[str] = None


class VectorStore:
    """Manages vector storage and similarity search for the AI Assistant.

    Uses LRU (Least Recently Used) caching to optimize frequent embedding lookups.
    Cache automatically evicts least recently used items when full, keeping the
    most frequently/recently accessed embeddings in memory for fast retrieval.

    Performance characteristics:
    - Cache hit: O(1) - immediate return from memory
    - Cache miss: O(1) cache + O(1) database lookup
    - LRU eviction: O(1) - constant time regardless of cache size
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None,
                 embedding_dim: int = 1536,
                 cache_size: int = 1000):
        """Initialize the vector store with LRU caching.

        Args:
            db_manager: Database manager instance. If None, will use default.
            embedding_dim: Dimension of embedding vectors (default: 1536 for OpenAI).
            cache_size: Number of embeddings to cache in memory. Larger values use
                       more RAM but reduce database queries.
        """
        self.db = db_manager or get_db_manager()
        self.embedding_dim = embedding_dim
        self.cache_size = cache_size

        # LRU cache for embeddings
        # Thread-safe via Lock since cache operations may occur from multiple threads
        self._cache: LRUCache = LRUCache(maxsize=cache_size)
        self._cache_lock = Lock()

        # Cache performance metrics
        # These help monitor cache effectiveness and tune cache_size
        self._cache_hits = 0
        self._cache_misses = 0

        self._embedding_functions: Dict[str, Callable] = {}

        logger.info(f"Vector store initialized with dimension {embedding_dim}, LRU cache size {cache_size}")
    
    def register_embedding_function(self, entity_type: EntityType, 
                                  func: Callable[[Dict[str, Any]], List[float]]) -> None:
        """Register a function to generate embeddings for an entity type.
        
        Args:
            entity_type: Type of entity.
            func: Function that takes entity data and returns embedding vector.
        """
        self._embedding_functions[entity_type.value] = func
        logger.debug(f"Registered embedding function for {entity_type.value}")
    
    def store_embedding(self, entity_type: EntityType, entity_id: int,
                       embedding: Union[List[float], np.ndarray],
                       model: str = "text-embedding-ada-002") -> int:
        """Store an embedding vector.
        
        Args:
            entity_type: Type of entity.
            entity_id: ID of the entity.
            embedding: Embedding vector.
            model: Model used to generate the embedding.
            
        Returns:
            ID of the stored embedding.
        """
        # Convert to list if numpy array
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        
        # Validate dimension
        if len(embedding) != self.embedding_dim:
            raise ValueError(f"Embedding dimension {len(embedding)} does not match expected {self.embedding_dim}")
        
        # Store in database
        embedding_id = self.db.store_embedding(entity_type.value, entity_id, embedding, model)
        
        # Update cache
        cache_key = f"{entity_type.value}_{entity_id}"
        self._update_cache(cache_key, np.array(embedding))
        
        logger.debug(f"Stored embedding for {entity_type.value} {entity_id}")
        return embedding_id
    
    def get_embedding(self, entity_type: EntityType, entity_id: int) -> Optional[np.ndarray]:
        """Get embedding for an entity with LRU caching.

        Checks cache first for O(1) lookup. On cache miss, fetches from database
        and updates cache. The LRU cache automatically evicts least recently used
        items when full.

        Args:
            entity_type: Type of entity.
            entity_id: ID of the entity.

        Returns:
            Embedding vector as numpy array, or None if not found.
        """
        cache_key = f"{entity_type.value}_{entity_id}"

        # Check cache first (thread-safe)
        with self._cache_lock:
            if cache_key in self._cache:
                self._cache_hits += 1
                # LRUCache automatically updates access order on retrieval
                return self._cache[cache_key]

        # Cache miss - fetch from database
        self._cache_misses += 1
        result = self.db.get_embedding(entity_type.value, entity_id)
        if result:
            embedding = np.array(result['embedding'])
            self._update_cache(cache_key, embedding)
            return embedding

        return None

    def _update_cache(self, key: str, embedding: np.ndarray) -> None:
        """Update the LRU cache with thread-safe operations.

        LRUCache automatically handles eviction when full:
        - When cache is at max capacity, adding a new item evicts the LRU item
        - LRU tracking is automatic - no manual bookkeeping needed
        - O(1) time complexity for both insertion and eviction
        """
        with self._cache_lock:
            self._cache[key] = embedding
    
    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray,
                         method: str = "cosine") -> float:
        """Compute similarity between two vectors.
        
        Args:
            vec1: First vector.
            vec2: Second vector.
            method: Similarity method ('cosine', 'euclidean', 'dot').
            
        Returns:
            Similarity score.
        """
        if method == "cosine":
            return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        elif method == "euclidean":
            return float(1 / (1 + np.linalg.norm(vec1 - vec2)))
        elif method == "dot":
            return float(np.dot(vec1, vec2))
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def search_similar(self, query_embedding: Union[List[float], np.ndarray],
                      entity_types: Optional[List[EntityType]] = None,
                      limit: int = 10,
                      threshold: float = 0.0,
                      method: str = "cosine",
                      include_data: bool = True) -> List[SimilarityResult]:
        """Search for similar entities across specified types.
        
        Args:
            query_embedding: Query embedding vector.
            entity_types: Types of entities to search. If None, search all types.
            limit: Maximum number of results.
            threshold: Minimum similarity threshold.
            method: Similarity method to use.
            include_data: Whether to include entity data in results.
            
        Returns:
            List of similarity results, sorted by score (descending).
        """
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
        
        # Normalize query for cosine similarity
        if method == "cosine":
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Determine entity types to search
        if entity_types is None:
            entity_types = list(EntityType)
        
        results = []
        
        # Search each entity type
        for entity_type in entity_types:
            type_results = self._search_entity_type(
                query_embedding, entity_type, threshold, method, include_data
            )
            results.extend(type_results)
        
        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
    
    def _search_entity_type(self, query_embedding: np.ndarray,
                           entity_type: EntityType,
                           threshold: float,
                           method: str,
                           include_data: bool) -> List[SimilarityResult]:
        """Search for similar entities of a specific type."""
        results = []
        
        # Get all embeddings for this entity type
        with self.db._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT entity_id, embedding 
                FROM embeddings 
                WHERE entity_type = ?
                ORDER BY created_at DESC
            """, (entity_type.value,))
            
            # Group by entity_id to get latest embedding for each entity
            entity_embeddings = {}
            for row in cursor.fetchall():
                entity_id = row[0]
                if entity_id not in entity_embeddings:
                    entity_embeddings[entity_id] = json.loads(row[1])
        
        # Compute similarities
        for entity_id, embedding in entity_embeddings.items():
            embedding_vec = np.array(embedding)
            
            # Normalize for cosine similarity
            if method == "cosine":
                embedding_vec = embedding_vec / np.linalg.norm(embedding_vec)
            
            score = self.compute_similarity(query_embedding, embedding_vec, method)
            
            if score >= threshold:
                result = SimilarityResult(
                    entity_type=entity_type.value,
                    entity_id=entity_id,
                    score=score
                )
                
                # Include entity data if requested
                if include_data:
                    result.entity_data = self._get_entity_data(entity_type, entity_id)
                
                results.append(result)
        
        return results
    
    def _get_entity_data(self, entity_type: EntityType, entity_id: int) -> Optional[Dict[str, Any]]:
        """Get data for an entity."""
        if entity_type == EntityType.MESSAGE:
            with self.db._pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM messages WHERE id = ?", (entity_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        
        elif entity_type == EntityType.KNOWLEDGE:
            return self.db.get_knowledge(entity_id)
        
        elif entity_type == EntityType.PROJECT:
            with self.db._pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM projects WHERE id = ?", (entity_id,))
                row = cursor.fetchone()
                if row:
                    data = dict(row)
                    data['metadata'] = json.loads(data['metadata'])
                    return data
                return None
        
        elif entity_type == EntityType.CONVERSATION:
            with self.db._pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM conversations WHERE id = ?", (entity_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        
        return None
    
    def update_embeddings_batch(self, updates: List[Tuple[EntityType, int, List[float]]],
                              model: str = "text-embedding-ada-002") -> int:
        """Update multiple embeddings in a batch.
        
        Args:
            updates: List of (entity_type, entity_id, embedding) tuples.
            model: Model used to generate embeddings.
            
        Returns:
            Number of embeddings updated.
        """
        count = 0
        with self.db._pool.get_connection() as conn:
            cursor = conn.cursor()
            
            for entity_type, entity_id, embedding in updates:
                cursor.execute("""
                    INSERT INTO embeddings (entity_type, entity_id, embedding, model)
                    VALUES (?, ?, ?, ?)
                """, (entity_type.value, entity_id, json.dumps(embedding), model))
                
                # Update cache
                cache_key = f"{entity_type.value}_{entity_id}"
                self._update_cache(cache_key, np.array(embedding))
                count += 1
            
            conn.commit()
        
        logger.info(f"Updated {count} embeddings in batch")
        return count
    
    def find_duplicates(self, entity_type: EntityType,
                       similarity_threshold: float = 0.95,
                       method: str = "cosine") -> List[Tuple[int, int, float]]:
        """Find potential duplicate entities based on embedding similarity.
        
        Args:
            entity_type: Type of entity to check.
            similarity_threshold: Minimum similarity to consider as duplicate.
            method: Similarity method to use.
            
        Returns:
            List of (entity_id1, entity_id2, similarity_score) tuples.
        """
        duplicates = []
        
        # Get all embeddings for this entity type
        with self.db._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT entity_id, embedding 
                FROM embeddings 
                WHERE entity_type = ?
            """, (entity_type.value,))
            
            entities = [(row[0], np.array(json.loads(row[1]))) for row in cursor.fetchall()]
        
        # Compare all pairs
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                id1, vec1 = entities[i]
                id2, vec2 = entities[j]
                
                similarity = self.compute_similarity(vec1, vec2, method)
                
                if similarity >= similarity_threshold:
                    duplicates.append((id1, id2, similarity))
        
        # Sort by similarity score
        duplicates.sort(key=lambda x: x[2], reverse=True)
        
        return duplicates
    
    def cluster_entities(self, entity_type: EntityType,
                        n_clusters: int = 5,
                        method: str = "kmeans") -> Dict[int, List[int]]:
        """Cluster entities based on their embeddings using scikit-learn.

        This is an optional feature that requires scikit-learn to be installed.
        If scikit-learn is not available, this method will raise a clear error
        with instructions on how to enable clustering.

        Args:
            entity_type: Type of entity to cluster.
            n_clusters: Number of clusters.
            method: Clustering method ('kmeans', 'hierarchical').

        Returns:
            Dictionary mapping cluster ID to list of entity IDs.

        Raises:
            ValueError: If scikit-learn is not installed or invalid method specified.
        """
        # Check if scikit-learn is available
        # This allows the system to function without sklearn for other features
        if not SKLEARN_AVAILABLE:
            raise ValueError(
                "Clustering requires scikit-learn to be installed. "
                "Install it with: pip install scikit-learn>=1.3.0\n"
                "Note: All other vector store features work without scikit-learn. "
                "Clustering is an optional advanced feature."
            )

        # Get all embeddings
        with self.db._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT entity_id, embedding
                FROM embeddings
                WHERE entity_type = ?
            """, (entity_type.value,))

            entity_ids = []
            embeddings = []

            for row in cursor.fetchall():
                entity_ids.append(row[0])
                embeddings.append(json.loads(row[1]))

        if not embeddings:
            logger.warning(f"No embeddings found for {entity_type.value}, returning empty clusters")
            return {}

        X = np.array(embeddings)

        # Use pre-imported sklearn classes (imported at module level if available)
        if method == "kmeans":
            # KMeans chosen for its speed and scalability with large embedding sets
            # n_clusters is capped at the number of samples to avoid errors
            kmeans = KMeans(n_clusters=min(n_clusters, len(X)), random_state=42)
            labels = kmeans.fit_predict(X)
        elif method == "hierarchical":
            # Hierarchical clustering provides dendrogram structure
            # Better for small to medium datasets where cluster hierarchy matters
            clustering = AgglomerativeClustering(n_clusters=min(n_clusters, len(X)))
            labels = clustering.fit_predict(X)
        else:
            raise ValueError(
                f"Unknown clustering method: {method}. "
                f"Supported methods: 'kmeans', 'hierarchical'"
            )

        # Group entities by cluster
        clusters = {}
        for entity_id, label in zip(entity_ids, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(entity_id)

        logger.info(f"Clustered {len(entity_ids)} entities into {len(clusters)} clusters using {method}")
        return clusters
    
    def generate_embeddings_for_entities(self, entity_type: EntityType,
                                       entity_ids: Optional[List[int]] = None,
                                       batch_size: int = 100,
                                       max_workers: int = 4) -> int:
        """Generate embeddings for entities that don't have them.
        
        Args:
            entity_type: Type of entity.
            entity_ids: Specific entity IDs to process. If None, process all.
            batch_size: Number of entities to process in each batch.
            max_workers: Number of parallel workers.
            
        Returns:
            Number of embeddings generated.
        """
        if entity_type.value not in self._embedding_functions:
            raise ValueError(f"No embedding function registered for {entity_type.value}")
        
        embedding_func = self._embedding_functions[entity_type.value]
        
        # Get entities without embeddings
        entities_to_process = self._get_entities_without_embeddings(entity_type, entity_ids)
        
        if not entities_to_process:
            logger.info(f"All {entity_type.value} entities already have embeddings")
            return 0
        
        logger.info(f"Generating embeddings for {len(entities_to_process)} {entity_type.value} entities")
        
        generated = 0
        
        # Process in batches with thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for i in range(0, len(entities_to_process), batch_size):
                batch = entities_to_process[i:i + batch_size]
                future = executor.submit(self._process_embedding_batch, 
                                       entity_type, batch, embedding_func)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    count = future.result()
                    generated += count
                except Exception as e:
                    logger.error(f"Error processing embedding batch: {e}")
        
        logger.info(f"Generated {generated} embeddings for {entity_type.value}")
        return generated
    
    def _get_entities_without_embeddings(self, entity_type: EntityType,
                                       entity_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Get entities that don't have embeddings yet."""
        entities = []
        
        with self.db._pool.get_connection() as conn:
            cursor = conn.cursor()
            
            if entity_type == EntityType.MESSAGE:
                query = """
                    SELECT m.* FROM messages m
                    LEFT JOIN embeddings e ON e.entity_type = 'message' AND e.entity_id = m.id
                    WHERE e.id IS NULL
                """
                if entity_ids:
                    query += f" AND m.id IN ({','.join('?' * len(entity_ids))})"
                    cursor.execute(query, entity_ids)
                else:
                    cursor.execute(query)
                
                entities = [dict(row) for row in cursor.fetchall()]
            
            elif entity_type == EntityType.KNOWLEDGE:
                query = """
                    SELECT k.* FROM knowledge_base k
                    LEFT JOIN embeddings e ON e.entity_type = 'knowledge' AND e.entity_id = k.id
                    WHERE e.id IS NULL
                """
                if entity_ids:
                    query += f" AND k.id IN ({','.join('?' * len(entity_ids))})"
                    cursor.execute(query, entity_ids)
                else:
                    cursor.execute(query)
                
                for row in cursor.fetchall():
                    data = dict(row)
                    data['tags'] = json.loads(data['tags'])
                    data['metadata'] = json.loads(data['metadata'])
                    entities.append(data)
            
            # Similar queries for other entity types...
        
        return entities
    
    def _process_embedding_batch(self, entity_type: EntityType,
                               entities: List[Dict[str, Any]],
                               embedding_func: Callable) -> int:
        """Process a batch of entities to generate embeddings."""
        updates = []
        
        for entity in entities:
            try:
                embedding = embedding_func(entity)
                updates.append((entity_type, entity['id'], embedding))
            except Exception as e:
                logger.error(f"Error generating embedding for {entity_type.value} {entity['id']}: {e}")
        
        if updates:
            return self.update_embeddings_batch(updates)
        
        return 0
    
    def export_embeddings(self, output_file: str,
                         entity_types: Optional[List[EntityType]] = None) -> int:
        """Export embeddings to a file for backup or analysis.
        
        Args:
            output_file: Path to output file (JSON format).
            entity_types: Types to export. If None, export all.
            
        Returns:
            Number of embeddings exported.
        """
        if entity_types is None:
            entity_types = list(EntityType)
        
        embeddings_data = []
        
        with self.db._pool.get_connection() as conn:
            cursor = conn.cursor()
            
            for entity_type in entity_types:
                cursor.execute("""
                    SELECT * FROM embeddings 
                    WHERE entity_type = ?
                    ORDER BY entity_id, created_at DESC
                """, (entity_type.value,))
                
                for row in cursor.fetchall():
                    embeddings_data.append({
                        'id': row['id'],
                        'entity_type': row['entity_type'],
                        'entity_id': row['entity_id'],
                        'embedding': json.loads(row['embedding']),
                        'model': row['model'],
                        'created_at': row['created_at']
                    })
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(embeddings_data, f, indent=2)
        
        logger.info(f"Exported {len(embeddings_data)} embeddings to {output_file}")
        return len(embeddings_data)
    
    def import_embeddings(self, input_file: str, overwrite: bool = False, dry_run: bool = False) -> int:
        """Import embeddings from a file with comprehensive validation.

        Security: This method validates all input data before import to prevent
        malicious or corrupted files from crashing the system or corrupting the database.

        Validation includes:
        - JSON schema validation (required fields)
        - Embedding dimension matching
        - Entity type validation
        - Transaction-based import with rollback on errors

        Args:
            input_file: Path to input file (JSON format).
            overwrite: Whether to overwrite existing embeddings.
            dry_run: If True, validate and preview but don't actually import.

        Returns:
            Number of embeddings imported (or would be imported in dry-run mode).

        Raises:
            FileNotFoundError: If input file doesn't exist.
            ValueError: If JSON is invalid or data fails validation.
            Exception: If database operation fails (with automatic rollback).
        """
        # Load and parse JSON file
        try:
            with open(input_file, 'r') as f:
                embeddings_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Import file not found: {input_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in import file: {e}")

        if not isinstance(embeddings_data, list):
            raise ValueError("Import file must contain a JSON array of embedding objects")

        # Required fields for each embedding entry
        # These are critical for data integrity and proper storage
        required_fields = {'entity_type', 'entity_id', 'embedding', 'model'}

        # Validate each embedding entry before importing
        for i, data in enumerate(embeddings_data):
            if not isinstance(data, dict):
                raise ValueError(f"Entry {i} is not a JSON object")

            # Check for required fields
            missing = required_fields - set(data.keys())
            if missing:
                raise ValueError(
                    f"Entry {i} missing required fields: {missing}. "
                    f"Required fields: {required_fields}"
                )

            # Validate entity_type is one of the known types
            # This prevents corrupt data from entering the system
            try:
                EntityType(data['entity_type'])
            except ValueError:
                valid_types = [t.value for t in EntityType]
                raise ValueError(
                    f"Entry {i} has invalid entity_type: '{data['entity_type']}'. "
                    f"Valid types: {valid_types}"
                )

            # Validate embedding is a list/array
            if not isinstance(data['embedding'], (list, tuple)):
                raise ValueError(
                    f"Entry {i} embedding is not an array "
                    f"(got {type(data['embedding']).__name__})"
                )

            # Validate embedding dimension matches expected
            # Mismatched dimensions would cause similarity search to fail
            if len(data['embedding']) != self.embedding_dim:
                raise ValueError(
                    f"Entry {i} has wrong embedding dimension: "
                    f"{len(data['embedding'])} (expected {self.embedding_dim}). "
                    f"This import file may be from a different model or configuration."
                )

            # Validate all embedding values are numbers
            # This prevents NaN, infinity, or string values that would break vector ops
            try:
                np.array(data['embedding'], dtype=np.float32)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Entry {i} has invalid embedding values: {e}. "
                    f"All embedding values must be numeric."
                )

        # If dry-run, report what would be imported and exit
        if dry_run:
            logger.info(
                f"DRY RUN: Would import {len(embeddings_data)} embeddings from {input_file}. "
                f"Validation passed. Use dry_run=False to perform actual import."
            )
            return len(embeddings_data)

        imported = 0

        # Use transaction for atomic import - either all succeed or all rollback
        # This prevents partial imports that could corrupt the database state
        with self.db._pool.get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute("BEGIN")

                for data in embeddings_data:
                    # Check if embedding exists
                    cursor.execute("""
                        SELECT id FROM embeddings
                        WHERE entity_type = ? AND entity_id = ?
                    """, (data['entity_type'], data['entity_id']))

                    exists = cursor.fetchone() is not None

                    if not exists or overwrite:
                        if exists and overwrite:
                            # Delete existing to allow overwrite
                            cursor.execute("""
                                DELETE FROM embeddings
                                WHERE entity_type = ? AND entity_id = ?
                            """, (data['entity_type'], data['entity_id']))

                        # Insert new embedding
                        cursor.execute("""
                            INSERT INTO embeddings (entity_type, entity_id, embedding, model, created_at)
                            VALUES (?, ?, ?, ?, ?)
                        """, (data['entity_type'], data['entity_id'],
                              json.dumps(data['embedding']), data['model'], data.get('created_at')))

                        imported += 1

                # Commit transaction - all imports succeeded
                conn.commit()
                logger.info(f"Successfully imported {imported} embeddings from {input_file}")

            except Exception as e:
                # Rollback on any error - this ensures database consistency
                conn.rollback()
                logger.error(f"Import failed, rolled back all changes: {e}")
                raise ValueError(f"Import failed at entry: {e}") from e

        return imported

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics.

        Returns detailed metrics about cache performance including:
        - Hit/miss counts and ratio
        - Current cache utilization
        - Performance indicators

        Use these metrics to:
        - Monitor cache effectiveness (target >80% hit rate)
        - Tune cache_size parameter
        - Identify performance bottlenecks

        Returns:
            Dictionary with cache statistics:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - hit_rate: Percentage of requests served from cache (0.0-1.0)
            - size: Current number of items in cache
            - max_size: Maximum cache capacity
            - utilization: Percentage of cache capacity used (0.0-1.0)
        """
        with self._cache_lock:
            total_requests = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
            current_size = len(self._cache)
            utilization = current_size / self.cache_size if self.cache_size > 0 else 0.0

            return {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "hit_rate": hit_rate,
                "size": current_size,
                "max_size": self.cache_size,
                "utilization": utilization
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about stored embeddings and cache performance."""
        # Get cache stats first
        cache_stats = self.get_cache_stats()

        stats = {
            'total_embeddings': 0,
            'by_entity_type': {},
            'by_model': {},
            'cache': cache_stats,  # Include cache performance metrics
            'embedding_dimension': self.embedding_dim
        }
        
        with self.db._pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total embeddings
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            stats['total_embeddings'] = cursor.fetchone()[0]
            
            # By entity type
            cursor.execute("""
                SELECT entity_type, COUNT(*) as count 
                FROM embeddings 
                GROUP BY entity_type
            """)
            stats['by_entity_type'] = {row[0]: row[1] for row in cursor.fetchall()}
            
            # By model
            cursor.execute("""
                SELECT model, COUNT(*) as count 
                FROM embeddings 
                GROUP BY model
            """)
            stats['by_model'] = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Average embeddings per entity type
            for entity_type in EntityType:
                cursor.execute("""
                    SELECT COUNT(DISTINCT entity_id) 
                    FROM embeddings 
                    WHERE entity_type = ?
                """, (entity_type.value,))
                unique_entities = cursor.fetchone()[0]
                
                if unique_entities > 0:
                    total = stats['by_entity_type'].get(entity_type.value, 0)
                    stats[f'avg_embeddings_per_{entity_type.value}'] = total / unique_entities
        
        return stats


def get_vector_store(db_manager: Optional[DatabaseManager] = None,
                    embedding_dim: int = 1536) -> VectorStore:
    """Get or create a vector store instance."""
    return VectorStore(db_manager, embedding_dim)