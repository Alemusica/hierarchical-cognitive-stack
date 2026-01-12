"""
LONG-TERM MEMORY (LTM)
======================
Graph-Vector DB per storage fingerprint consolidati.
DNA-like: pattern profondi, knowledge encoded.

Storage:
- SurrealDB con HNSW index per retrieval veloce
- Fallback in-memory per testing

Flussi:
- Instinctive → LTM: retrieval diretto (con routing da Dispatcher)
- Dispatcher → LTM: injection embedding distillati
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np
import asyncio
import json


@dataclass
class LTMEntry:
    """Entry nel Long-Term Memory"""
    id: str
    vector: np.ndarray
    created_at: datetime
    updated_at: datetime
    access_count: int = 0
    importance: float = 0.5
    topics: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "access_count": self.access_count,
            "importance": self.importance,
            "topics": self.topics,
            "metadata": self.metadata,
            "vector_dim": len(self.vector)
        }


class LTMBackend(ABC):
    """Backend astratto per LTM storage"""

    @abstractmethod
    async def store(
        self,
        id: str,
        vector: np.ndarray,
        metadata: Dict[str, Any]
    ) -> bool:
        pass

    @abstractmethod
    async def search(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        pass

    @abstractmethod
    async def get(self, id: str) -> Optional[LTMEntry]:
        pass

    @abstractmethod
    async def delete(self, id: str) -> bool:
        pass

    @abstractmethod
    async def update_metadata(
        self,
        id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        pass


# ==================== IN-MEMORY BACKEND ====================

class InMemoryLTMBackend(LTMBackend):
    """
    Backend in-memory per testing/development.
    Usa numpy per cosine similarity search.
    """

    def __init__(self):
        self._entries: Dict[str, LTMEntry] = {}
        self._vectors: Optional[np.ndarray] = None
        self._ids: List[str] = []
        self._dirty = True

    async def store(
        self,
        id: str,
        vector: np.ndarray,
        metadata: Dict[str, Any]
    ) -> bool:
        now = datetime.now()

        entry = LTMEntry(
            id=id,
            vector=vector.astype(np.float32),
            created_at=now,
            updated_at=now,
            importance=metadata.get("importance", 0.5),
            topics=metadata.get("topics", []),
            metadata=metadata
        )

        self._entries[id] = entry
        self._dirty = True
        return True

    async def search(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        if not self._entries:
            return []

        # Rebuild index se dirty
        if self._dirty:
            self._rebuild_index()

        # Cosine similarity
        query = vector.astype(np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        query = query / query_norm

        # Batch similarity
        similarities = self._vectors @ query

        # Top-k
        indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in indices:
            sim = float(similarities[idx])
            if sim >= threshold:
                entry_id = self._ids[idx]
                # Update access count
                self._entries[entry_id].access_count += 1
                self._entries[entry_id].updated_at = datetime.now()
                results.append((entry_id, sim))

        return results

    def _rebuild_index(self):
        """Ricostruisce matrice vettori per search"""
        if not self._entries:
            self._vectors = None
            self._ids = []
            return

        self._ids = list(self._entries.keys())
        vectors = [self._entries[id].vector for id in self._ids]
        self._vectors = np.stack(vectors)

        # Normalizza per cosine similarity veloce
        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
        self._vectors = self._vectors / (norms + 1e-8)

        self._dirty = False

    async def get(self, id: str) -> Optional[LTMEntry]:
        return self._entries.get(id)

    async def delete(self, id: str) -> bool:
        if id in self._entries:
            del self._entries[id]
            self._dirty = True
            return True
        return False

    async def update_metadata(
        self,
        id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        if id not in self._entries:
            return False

        entry = self._entries[id]
        entry.metadata.update(metadata)
        entry.updated_at = datetime.now()

        if "importance" in metadata:
            entry.importance = metadata["importance"]
        if "topics" in metadata:
            entry.topics = metadata["topics"]

        return True

    def get_stats(self) -> Dict[str, Any]:
        return {
            "entries_count": len(self._entries),
            "index_dirty": self._dirty,
            "total_accesses": sum(e.access_count for e in self._entries.values())
        }


# ==================== SURREALDB BACKEND ====================

class SurrealDBLTMBackend(LTMBackend):
    """
    Backend SurrealDB con HNSW vector index.
    Per production use.
    """

    def __init__(
        self,
        url: str = "ws://localhost:8000/rpc",
        namespace: str = "hierarchical_memory",
        database: str = "ltm",
        embedding_dim: int = 128
    ):
        self.url = url
        self.namespace = namespace
        self.database = database
        self.embedding_dim = embedding_dim
        self._client = None
        self._initialized = False

    async def _get_client(self):
        """Lazy init del client SurrealDB"""
        if self._client is None:
            try:
                from surrealdb import Surreal
                self._client = Surreal(self.url)
                await self._client.connect()
                await self._client.use(self.namespace, self.database)

                if not self._initialized:
                    await self._init_schema()
                    self._initialized = True

            except Exception as e:
                print(f"SurrealDB connection error: {e}")
                print("Falling back to in-memory backend")
                self._client = "fallback"

        return self._client

    async def _init_schema(self):
        """Inizializza schema e indici"""
        client = self._client
        if client == "fallback":
            return

        try:
            # Crea tabella con HNSW index
            await client.query(f"""
                DEFINE TABLE ltm_entry SCHEMAFULL;
                DEFINE FIELD vector ON ltm_entry TYPE array;
                DEFINE FIELD created_at ON ltm_entry TYPE datetime;
                DEFINE FIELD updated_at ON ltm_entry TYPE datetime;
                DEFINE FIELD access_count ON ltm_entry TYPE int DEFAULT 0;
                DEFINE FIELD importance ON ltm_entry TYPE float DEFAULT 0.5;
                DEFINE FIELD topics ON ltm_entry TYPE array DEFAULT [];
                DEFINE FIELD metadata ON ltm_entry TYPE object DEFAULT {{}};

                DEFINE INDEX ltm_vector_idx ON ltm_entry
                    FIELDS vector HNSW DIMENSION {self.embedding_dim}
                    DIST COSINE TYPE F32;
            """)
        except Exception as e:
            print(f"Schema init error (may already exist): {e}")

    async def store(
        self,
        id: str,
        vector: np.ndarray,
        metadata: Dict[str, Any]
    ) -> bool:
        client = await self._get_client()

        if client == "fallback":
            return False

        try:
            now = datetime.now().isoformat()

            await client.create(f"ltm_entry:{id}", {
                "vector": vector.tolist(),
                "created_at": now,
                "updated_at": now,
                "access_count": 0,
                "importance": metadata.get("importance", 0.5),
                "topics": metadata.get("topics", []),
                "metadata": metadata
            })
            return True
        except Exception as e:
            print(f"SurrealDB store error: {e}")
            return False

    async def search(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        client = await self._get_client()

        if client == "fallback":
            return []

        try:
            # HNSW vector search
            result = await client.query(f"""
                SELECT id, vector::similarity::cosine(vector, $query_vec) AS similarity
                FROM ltm_entry
                WHERE vector <|{top_k}|> $query_vec
                ORDER BY similarity DESC
            """, {"query_vec": vector.tolist()})

            matches = []
            for row in result[0]["result"]:
                sim = row["similarity"]
                if sim >= threshold:
                    entry_id = row["id"].split(":")[1]
                    matches.append((entry_id, sim))

                    # Update access count
                    await client.query("""
                        UPDATE $id SET
                            access_count += 1,
                            updated_at = time::now()
                    """, {"id": row["id"]})

            return matches

        except Exception as e:
            print(f"SurrealDB search error: {e}")
            return []

    async def get(self, id: str) -> Optional[LTMEntry]:
        client = await self._get_client()

        if client == "fallback":
            return None

        try:
            result = await client.select(f"ltm_entry:{id}")
            if not result:
                return None

            data = result[0]
            return LTMEntry(
                id=id,
                vector=np.array(data["vector"], dtype=np.float32),
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"]),
                access_count=data["access_count"],
                importance=data["importance"],
                topics=data["topics"],
                metadata=data["metadata"]
            )
        except Exception as e:
            print(f"SurrealDB get error: {e}")
            return None

    async def delete(self, id: str) -> bool:
        client = await self._get_client()

        if client == "fallback":
            return False

        try:
            await client.delete(f"ltm_entry:{id}")
            return True
        except Exception as e:
            print(f"SurrealDB delete error: {e}")
            return False

    async def update_metadata(
        self,
        id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        client = await self._get_client()

        if client == "fallback":
            return False

        try:
            await client.query("""
                UPDATE $id SET
                    metadata = $metadata,
                    importance = $importance,
                    topics = $topics,
                    updated_at = time::now()
            """, {
                "id": f"ltm_entry:{id}",
                "metadata": metadata,
                "importance": metadata.get("importance", 0.5),
                "topics": metadata.get("topics", [])
            })
            return True
        except Exception as e:
            print(f"SurrealDB update error: {e}")
            return False


# ==================== LTM MAIN CLASS ====================

class LTM:
    """
    Long-Term Memory.

    Graph-Vector DB per fingerprint consolidati.
    Supporta backend multipli (in-memory, SurrealDB).
    """

    def __init__(
        self,
        backend: Optional[LTMBackend] = None,
        embedding_dim: int = 128
    ):
        self.embedding_dim = embedding_dim
        self.backend = backend or InMemoryLTMBackend()

        self._store_count = 0
        self._search_count = 0

    @classmethod
    def with_surrealdb(
        cls,
        url: str = "ws://localhost:8000/rpc",
        namespace: str = "hierarchical_memory",
        database: str = "ltm",
        embedding_dim: int = 128
    ) -> 'LTM':
        """Factory per LTM con SurrealDB backend"""
        backend = SurrealDBLTMBackend(
            url=url,
            namespace=namespace,
            database=database,
            embedding_dim=embedding_dim
        )
        return cls(backend=backend, embedding_dim=embedding_dim)

    @classmethod
    def in_memory(cls, embedding_dim: int = 128) -> 'LTM':
        """Factory per LTM in-memory (testing)"""
        return cls(backend=InMemoryLTMBackend(), embedding_dim=embedding_dim)

    # ==================== CORE OPERATIONS ====================

    async def store(
        self,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None
    ) -> str:
        """
        Store fingerprint in LTM.
        Dispatcher → LTM injection.
        """
        import hashlib

        metadata = metadata or {}

        # Genera ID se non fornito
        if id is None:
            hash_input = f"{vector.tobytes().hex()[:32]}:{datetime.now().timestamp()}"
            id = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        await self.backend.store(id, vector, metadata)
        self._store_count += 1

        return id

    async def search(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
        routing_hint: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Search in LTM.
        Instinctive → LTM retrieval diretto.

        Args:
            vector: Query vector
            top_k: Numero risultati
            threshold: Soglia minima similarity
            routing_hint: Hint da Dispatcher (bucket/nodi)

        Returns:
            Lista di (id, similarity)
        """
        # routing_hint per ora ignorato (sarà usato con backend più sofisticati)
        results = await self.backend.search(vector, top_k, threshold)
        self._search_count += 1
        return results

    async def get(self, id: str) -> Optional[LTMEntry]:
        """Recupera entry per ID"""
        return await self.backend.get(id)

    async def delete(self, id: str) -> bool:
        """Elimina entry"""
        return await self.backend.delete(id)

    async def update(
        self,
        id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Aggiorna metadata entry"""
        return await self.backend.update_metadata(id, metadata)

    # ==================== BATCH OPERATIONS ====================

    async def store_batch(
        self,
        vectors: List[np.ndarray],
        metadata_list: Optional[List[Dict]] = None
    ) -> List[str]:
        """Store batch di vettori"""
        metadata_list = metadata_list or [{}] * len(vectors)
        ids = []

        for vec, meta in zip(vectors, metadata_list):
            id = await self.store(vec, meta)
            ids.append(id)

        return ids

    # ==================== STATS ====================

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "embedding_dim": self.embedding_dim,
            "store_count": self._store_count,
            "search_count": self._search_count,
            "backend_type": type(self.backend).__name__
        }

        if hasattr(self.backend, 'get_stats'):
            stats["backend"] = self.backend.get_stats()

        return stats
