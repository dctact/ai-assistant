"""Database manager for AI Assistant memory storage.

Provides SQLite database management with connection pooling, schema creation,
CRUD operations, FTS5 full-text search, and vector storage capabilities.

Overview:
- Thread-safe connection pooling for concurrent access
- Transaction management with automatic rollback on errors
- Full-text search via SQLite FTS5
- Vector embedding storage and retrieval

Key Features:
- Explicit transaction management prevents partial updates
- Retry logic with exponential backoff for write conflicts
- Connection pooling for optimal resource usage
- WAL mode for better concurrent read performance

Transaction Management:
- Use transaction() context manager for multi-step operations
- Automatic BEGIN/COMMIT on success
- Automatic ROLLBACK on error
- Prevents data corruption from partial updates

Recent Changes:
- 2025-01-14: Added transaction context manager with retry logic and rollback
"""

import sqlite3
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from contextlib import contextmanager
from queue import Queue
import numpy as np
from loguru import logger

class ConnectionPool:
    """SQLite connection pool for thread-safe database access."""
    
    def __init__(self, db_path: str, max_connections: int = 5):
        self.db_path = db_path
        self.max_connections = max_connections
        self._pool = Queue(maxsize=max_connections)
        self._lock = threading.Lock()
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """Initialize connection pool with connections."""
        for _ in range(self.max_connections):
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._pool.put(conn)
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        conn = self._pool.get()
        try:
            yield conn
        finally:
            self._pool.put(conn)
    
    def close_all(self) -> None:
        """Close all connections in the pool."""
        while not self._pool.empty():
            conn = self._pool.get()
            conn.close()


class DatabaseManager:
    """Main database manager for AI Assistant memory."""
    
    def __init__(self, db_path: str = "~/ai-assistant/memory/assistant.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._pool = ConnectionPool(str(self.db_path))
        self._initialize_database()
        logger.info(f"Database initialized at {self.db_path}")
    
    def _initialize_database(self) -> None:
        """Initialize database schema."""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT UNIQUE NOT NULL,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                )
            """)
            
            # Knowledge base table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_base (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT,
                    tags TEXT DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Projects table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    path TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Embeddings table for vector storage
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_type TEXT NOT NULL,
                    entity_id INTEGER NOT NULL,
                    embedding TEXT NOT NULL,
                    model TEXT DEFAULT 'text-embedding-ada-002',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create FTS5 virtual tables for full-text search
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    content, 
                    conversation_id UNINDEXED,
                    content=messages,
                    content_rowid=id
                )
            """)
            
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                    title, 
                    content,
                    content=knowledge_base,
                    content_rowid=id
                )
            """)
            
            # Create triggers to keep FTS tables in sync
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                    INSERT INTO messages_fts(rowid, content, conversation_id) 
                    VALUES (new.id, new.content, new.conversation_id);
                END
            """)
            
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                    DELETE FROM messages_fts WHERE rowid = old.id;
                END
            """)
            
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
                    UPDATE messages_fts SET content = new.content 
                    WHERE rowid = new.id;
                END
            """)
            
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge_base BEGIN
                    INSERT INTO knowledge_fts(rowid, title, content) 
                    VALUES (new.id, new.title, new.content);
                END
            """)
            
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge_base BEGIN
                    DELETE FROM knowledge_fts WHERE rowid = old.id;
                END
            """)
            
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS knowledge_au AFTER UPDATE ON knowledge_base BEGIN
                    UPDATE knowledge_fts SET title = new.title, content = new.content 
                    WHERE rowid = new.id;
                END
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_entity ON embeddings(entity_type, entity_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge_base(category)")
            
            conn.commit()

    @contextmanager
    def transaction(self, max_retries: int = 3):
        """Transaction context manager with automatic rollback and retry logic.

        Provides explicit transaction management for multi-step database operations.
        Automatically begins a transaction, commits on success, and rolls back on error.

        Features:
        - Automatic BEGIN/COMMIT/ROLLBACK
        - Retry logic with exponential backoff for SQLITE_BUSY errors
        - Thread-safe via connection pool
        - Prevents partial updates that could corrupt data

        Args:
            max_retries: Maximum number of retry attempts for busy database (default: 3)

        Usage:
            with db_manager.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO ...")
                cursor.execute("UPDATE ...")
                # Both operations commit together or both roll back on error

        Yields:
            SQLite connection with active transaction

        Raises:
            Exception: Re-raises any exception after rollback, allowing caller to handle

        Examples:
            # Atomic multi-step operation
            with self.transaction() as conn:
                cursor = conn.cursor()
                # Insert message
                cursor.execute("INSERT INTO messages (...) VALUES (...)")
                message_id = cursor.lastrowid
                # Update conversation timestamp
                cursor.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", ...)
                # Both succeed or both fail together
        """
        import time

        retry_count = 0
        last_error = None

        while retry_count <= max_retries:
            conn = self._pool._pool.get()
            try:
                # Begin transaction explicitly
                # IMMEDIATE mode locks database for writes immediately
                # This prevents SQLITE_BUSY errors during the transaction
                conn.execute("BEGIN IMMEDIATE")

                yield conn

                # Commit transaction if no exceptions
                conn.commit()
                logger.debug("Transaction committed successfully")
                return

            except sqlite3.OperationalError as e:
                # Handle database busy/locked errors with retry logic
                conn.rollback()

                if "database is locked" in str(e).lower() or "busy" in str(e).lower():
                    retry_count += 1
                    last_error = e

                    if retry_count <= max_retries:
                        # Exponential backoff: 0.1s, 0.2s, 0.4s
                        wait_time = 0.1 * (2 ** (retry_count - 1))
                        logger.warning(
                            f"Database busy, retrying ({retry_count}/{max_retries}) "
                            f"after {wait_time}s: {e}"
                        )
                        self._pool._pool.put(conn)
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Transaction failed after {max_retries} retries: {e}")
                        self._pool._pool.put(conn)
                        raise
                else:
                    # Non-busy operational error - don't retry
                    logger.error(f"Database operational error: {e}")
                    self._pool._pool.put(conn)
                    raise

            except Exception as e:
                # Rollback on any error to maintain consistency
                conn.rollback()
                logger.error(f"Transaction rolled back due to error: {e}")
                self._pool._pool.put(conn)
                raise
            finally:
                # Ensure connection always returns to pool
                if conn:
                    try:
                        self._pool._pool.put(conn)
                    except:
                        pass

        # If we exhausted all retries
        if last_error:
            raise last_error

    # Conversation CRUD operations
    def create_conversation(self, conversation_id: str, title: Optional[str] = None, 
                          metadata: Optional[Dict[str, Any]] = None) -> int:
        """Create a new conversation."""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO conversations (conversation_id, title, metadata)
                    VALUES (?, ?, ?)
                """, (conversation_id, title, json.dumps(metadata or {})))
                conn.commit()
                logger.info(f"Created conversation: {conversation_id}")
                return cursor.lastrowid
            except sqlite3.IntegrityError as e:
                logger.error(f"Failed to create conversation {conversation_id}: {e}")
                raise
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID."""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM conversations WHERE conversation_id = ?
            """, (conversation_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def update_conversation(self, conversation_id: str, title: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update a conversation."""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            updates = []
            params = []
            
            if title is not None:
                updates.append("title = ?")
                params.append(title)
            
            if metadata is not None:
                updates.append("metadata = ?")
                params.append(json.dumps(metadata))
            
            if updates:
                updates.append("updated_at = CURRENT_TIMESTAMP")
                params.append(conversation_id)
                
                cursor.execute(f"""
                    UPDATE conversations 
                    SET {', '.join(updates)}
                    WHERE conversation_id = ?
                """, params)
                conn.commit()
                return cursor.rowcount > 0
            return False
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its messages atomically.

        Uses transaction to ensure both messages and conversation are deleted
        together. If either deletion fails, both are rolled back.

        This prevents:
        - Orphaned messages without parent conversation
        - Conversation records without their messages
        - Partial deletions that corrupt referential integrity
        """
        with self.transaction() as conn:
            cursor = conn.cursor()

            # Delete all messages first (child records)
            cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            messages_deleted = cursor.rowcount

            # Delete conversation (parent record)
            cursor.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
            conversation_deleted = cursor.rowcount > 0

            if conversation_deleted:
                logger.info(f"Deleted conversation {conversation_id} with {messages_deleted} messages")

            return conversation_deleted
    
    def list_conversations(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List conversations with pagination."""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM conversations 
                ORDER BY updated_at DESC 
                LIMIT ? OFFSET ?
            """, (limit, offset))
            return [dict(row) for row in cursor.fetchall()]
    
    # Message CRUD operations
    def add_message(self, conversation_id: str, role: str, content: str,
                   metadata: Optional[Dict[str, Any]] = None) -> int:
        """Add a message to a conversation with transactional consistency.

        Uses transaction to ensure message insertion and conversation update
        happen atomically. If either operation fails, both are rolled back.

        This prevents:
        - Messages being added without updating conversation timestamp
        - Partial updates that could leave database in inconsistent state
        - Race conditions where timestamp update fails silently
        """
        with self.transaction() as conn:
            cursor = conn.cursor()

            # Insert message
            cursor.execute("""
                INSERT INTO messages (conversation_id, role, content, metadata)
                VALUES (?, ?, ?, ?)
            """, (conversation_id, role, content, json.dumps(metadata or {})))

            message_id = cursor.lastrowid

            # Update conversation timestamp
            # This must succeed for the transaction to commit
            cursor.execute("""
                UPDATE conversations
                SET updated_at = CURRENT_TIMESTAMP
                WHERE conversation_id = ?
            """, (conversation_id,))

            logger.debug(f"Added message {message_id} to conversation {conversation_id}")
            return message_id
    
    def get_messages(self, conversation_id: str, limit: Optional[int] = None,
                    offset: int = 0) -> List[Dict[str, Any]]:
        """Get messages for a conversation."""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT * FROM messages 
                WHERE conversation_id = ? 
                ORDER BY timestamp ASC
            """
            params = [conversation_id]
            
            if limit:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def search_messages(self, query: str, conversation_id: Optional[str] = None,
                       limit: int = 20) -> List[Dict[str, Any]]:
        """Search messages using FTS5."""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            
            if conversation_id:
                cursor.execute("""
                    SELECT m.*, highlight(messages_fts, 0, '[', ']') as highlighted
                    FROM messages m
                    JOIN messages_fts ON m.id = messages_fts.rowid
                    WHERE messages_fts MATCH ? AND m.conversation_id = ?
                    ORDER BY rank
                    LIMIT ?
                """, (query, conversation_id, limit))
            else:
                cursor.execute("""
                    SELECT m.*, highlight(messages_fts, 0, '[', ']') as highlighted
                    FROM messages m
                    JOIN messages_fts ON m.id = messages_fts.rowid
                    WHERE messages_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, (query, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    # Knowledge base operations
    def add_knowledge(self, title: str, content: str, category: Optional[str] = None,
                     tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Add knowledge to the knowledge base."""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO knowledge_base (title, content, category, tags, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (title, content, category, json.dumps(tags or []), json.dumps(metadata or {})))
            conn.commit()
            logger.info(f"Added knowledge: {title}")
            return cursor.lastrowid
    
    def get_knowledge(self, knowledge_id: int) -> Optional[Dict[str, Any]]:
        """Get knowledge by ID."""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM knowledge_base WHERE id = ?", (knowledge_id,))
            row = cursor.fetchone()
            if row:
                data = dict(row)
                data['tags'] = json.loads(data['tags'])
                data['metadata'] = json.loads(data['metadata'])
                return data
            return None
    
    def update_knowledge(self, knowledge_id: int, title: Optional[str] = None,
                        content: Optional[str] = None, category: Optional[str] = None,
                        tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update knowledge entry."""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            updates = []
            params = []
            
            if title is not None:
                updates.append("title = ?")
                params.append(title)
            
            if content is not None:
                updates.append("content = ?")
                params.append(content)
            
            if category is not None:
                updates.append("category = ?")
                params.append(category)
            
            if tags is not None:
                updates.append("tags = ?")
                params.append(json.dumps(tags))
            
            if metadata is not None:
                updates.append("metadata = ?")
                params.append(json.dumps(metadata))
            
            if updates:
                updates.append("updated_at = CURRENT_TIMESTAMP")
                params.append(knowledge_id)
                
                cursor.execute(f"""
                    UPDATE knowledge_base 
                    SET {', '.join(updates)}
                    WHERE id = ?
                """, params)
                conn.commit()
                return cursor.rowcount > 0
            return False
    
    def delete_knowledge(self, knowledge_id: int) -> bool:
        """Delete knowledge entry."""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM knowledge_base WHERE id = ?", (knowledge_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def search_knowledge(self, query: str, category: Optional[str] = None,
                        limit: int = 20) -> List[Dict[str, Any]]:
        """Search knowledge base using FTS5."""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            
            if category:
                cursor.execute("""
                    SELECT k.*, 
                           highlight(knowledge_fts, 0, '[', ']') as highlighted_title,
                           highlight(knowledge_fts, 1, '[', ']') as highlighted_content
                    FROM knowledge_base k
                    JOIN knowledge_fts ON k.id = knowledge_fts.rowid
                    WHERE knowledge_fts MATCH ? AND k.category = ?
                    ORDER BY rank
                    LIMIT ?
                """, (query, category, limit))
            else:
                cursor.execute("""
                    SELECT k.*, 
                           highlight(knowledge_fts, 0, '[', ']') as highlighted_title,
                           highlight(knowledge_fts, 1, '[', ']') as highlighted_content
                    FROM knowledge_base k
                    JOIN knowledge_fts ON k.id = knowledge_fts.rowid
                    WHERE knowledge_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, (query, limit))
            
            results = []
            for row in cursor.fetchall():
                data = dict(row)
                data['tags'] = json.loads(data['tags'])
                data['metadata'] = json.loads(data['metadata'])
                results.append(data)
            
            return results
    
    # Project operations
    def create_project(self, project_id: str, name: str, description: Optional[str] = None,
                      path: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Create a new project."""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO projects (project_id, name, description, path, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (project_id, name, description, path, json.dumps(metadata or {})))
            conn.commit()
            logger.info(f"Created project: {name}")
            return cursor.lastrowid
    
    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get project by ID."""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects WHERE project_id = ?", (project_id,))
            row = cursor.fetchone()
            if row:
                data = dict(row)
                data['metadata'] = json.loads(data['metadata'])
                return data
            return None
    
    def update_project(self, project_id: str, name: Optional[str] = None,
                      description: Optional[str] = None, path: Optional[str] = None,
                      status: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update project information."""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            updates = []
            params = []
            
            if name is not None:
                updates.append("name = ?")
                params.append(name)
            
            if description is not None:
                updates.append("description = ?")
                params.append(description)
            
            if path is not None:
                updates.append("path = ?")
                params.append(path)
            
            if status is not None:
                updates.append("status = ?")
                params.append(status)
            
            if metadata is not None:
                updates.append("metadata = ?")
                params.append(json.dumps(metadata))
            
            if updates:
                updates.append("updated_at = CURRENT_TIMESTAMP")
                params.append(project_id)
                
                cursor.execute(f"""
                    UPDATE projects 
                    SET {', '.join(updates)}
                    WHERE project_id = ?
                """, params)
                conn.commit()
                return cursor.rowcount > 0
            return False
    
    def list_projects(self, status: Optional[str] = None, limit: int = 50,
                     offset: int = 0) -> List[Dict[str, Any]]:
        """List projects with optional filtering."""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            
            if status:
                cursor.execute("""
                    SELECT * FROM projects 
                    WHERE status = ?
                    ORDER BY updated_at DESC 
                    LIMIT ? OFFSET ?
                """, (status, limit, offset))
            else:
                cursor.execute("""
                    SELECT * FROM projects 
                    ORDER BY updated_at DESC 
                    LIMIT ? OFFSET ?
                """, (limit, offset))
            
            results = []
            for row in cursor.fetchall():
                data = dict(row)
                data['metadata'] = json.loads(data['metadata'])
                results.append(data)
            
            return results
    
    # Embedding operations
    def store_embedding(self, entity_type: str, entity_id: int, embedding: List[float],
                       model: str = "text-embedding-ada-002") -> int:
        """Store an embedding vector."""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO embeddings (entity_type, entity_id, embedding, model)
                VALUES (?, ?, ?, ?)
            """, (entity_type, entity_id, json.dumps(embedding), model))
            conn.commit()
            return cursor.lastrowid
    
    def get_embedding(self, entity_type: str, entity_id: int) -> Optional[Dict[str, Any]]:
        """Get embedding for an entity."""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM embeddings 
                WHERE entity_type = ? AND entity_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (entity_type, entity_id))
            row = cursor.fetchone()
            if row:
                data = dict(row)
                data['embedding'] = json.loads(data['embedding'])
                return data
            return None
    
    def search_similar_embeddings(self, query_embedding: List[float], entity_type: str,
                                 limit: int = 10, threshold: float = 0.8) -> List[Tuple[int, float]]:
        """Search for similar embeddings using cosine similarity."""
        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)
        
        results = []
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT entity_id, embedding 
                FROM embeddings 
                WHERE entity_type = ?
            """, (entity_type,))
            
            for row in cursor.fetchall():
                entity_id = row[0]
                embedding = np.array(json.loads(row[1]))
                
                # Compute cosine similarity
                similarity = np.dot(query_vec, embedding) / (query_norm * np.linalg.norm(embedding))
                
                if similarity >= threshold:
                    results.append((entity_id, float(similarity)))
            
            # Sort by similarity score
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:limit]
    
    # Context retrieval methods
    def get_conversation_context(self, conversation_id: str, max_messages: int = 10) -> Dict[str, Any]:
        """Get conversation context including recent messages."""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return {}
        
        messages = self.get_messages(conversation_id, limit=max_messages)
        
        return {
            "conversation": conversation,
            "messages": messages,
            "message_count": len(messages)
        }
    
    def get_conversation_messages(self, conversation_id: str, limit: Optional[int] = None, 
                                offset: int = 0) -> List[Dict[str, Any]]:
        """Get messages for a conversation (alias for get_messages)."""
        return self.get_messages(conversation_id, limit=limit, offset=offset)
    
    def get_relevant_knowledge(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get relevant knowledge based on query."""
        return self.search_knowledge(query, limit=limit)
    
    def get_projects(self, limit: Optional[int] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all projects with optional filtering."""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM projects"
            params = []
            
            if status:
                query += " WHERE status = ?"
                params.append(status)
            
            query += " ORDER BY updated_at DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get a single project by ID."""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_project_context(self, project_id: str) -> Dict[str, Any]:
        """Get project context including related conversations."""
        project = self.get_project(project_id)
        if not project:
            return {}
        
        # Find conversations related to this project
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT c.* 
                FROM conversations c
                WHERE json_extract(c.metadata, '$.project_id') = ?
                ORDER BY c.updated_at DESC
                LIMIT 10
            """, (project_id,))
            
            conversations = [dict(row) for row in cursor.fetchall()]
        
        return {
            "project": project,
            "conversations": conversations,
            "conversation_count": len(conversations)
        }
    
    def get_full_context(self, conversation_id: Optional[str] = None,
                        query: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive context for the assistant."""
        context = {}
        
        if conversation_id:
            context["conversation"] = self.get_conversation_context(conversation_id)
        
        if query:
            context["relevant_knowledge"] = self.get_relevant_knowledge(query)
            context["relevant_messages"] = self.search_messages(query, limit=10)
        
        # Get active projects
        context["active_projects"] = self.get_projects(status="active", limit=5)
        
        return context
    
    def archive_old_conversations(self, cutoff_date: datetime) -> int:
        """Archive conversations older than cutoff_date."""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE conversations 
                SET metadata = json_set(COALESCE(metadata, '{}'), '$.archived', 'true')
                WHERE updated_at < ?
            """, (cutoff_date.isoformat(),))
            conn.commit()
            archived_count = cursor.rowcount
            logger.info(f"Archived {archived_count} conversations older than {cutoff_date}")
            return archived_count

    # Utility methods
    def vacuum(self) -> None:
        """Vacuum the database to reclaim space."""
        with self._pool.get_connection() as conn:
            conn.execute("VACUUM")
            logger.info("Database vacuumed")
    
    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            tables = ['conversations', 'messages', 'knowledge_base', 'projects', 'embeddings']
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]
            
            return stats
    
    def close(self) -> None:
        """Close database connections."""
        self._pool.close_all()
        logger.info("Database connections closed")


# Singleton instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager(db_path: Optional[str] = None) -> DatabaseManager:
    """Get or create the database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(db_path or "~/ai-assistant/memory/assistant.db")
    return _db_manager