"""
DISPATCHER (Medium-Term Memory)
===============================
Ibrido LLM + NN:

LLM Component:
- Distilla/sintetizza knowledge
- Valuta match e genera risposte
- Semantic compression

NN Component:
- Router: fornisce mappa/routing per LTM query
- Embedder: crea embedding per injection in LTM
- Sintetizza regole e pesi

Flussi:
- Instinctive → Dispatcher NN: "dove cerco?" → mappa
- Instinctive → Dispatcher LLM: "valuta questi risultati"
- STM → Dispatcher: dati da distillare
- Dispatcher → LTM: injection embedding distillati
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Callable
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime
import asyncio

from .instinctive import Fingerprint, RetrievalResult


# ==================== INTERFACES ====================

@dataclass
class RoutingMap:
    """Mappa routing per query LTM"""
    buckets: List[str]           # Bucket/cluster suggeriti
    node_ids: List[str]          # Nodi specifici nel grafo
    confidence: float            # Confidenza nel routing
    strategy: str = "hnsw"       # Strategia: hnsw, exact, hybrid


@dataclass
class EvaluationResult:
    """Risultato valutazione Dispatcher LLM"""
    should_respond: bool
    response: Optional[str]
    confidence: float
    reasoning: Optional[str]
    action: str = "none"  # none, respond, escalate, store


@dataclass
class DistillationResult:
    """Risultato distillazione per injection in LTM"""
    embedding: np.ndarray
    summary: str
    topics: List[str]
    importance: float
    metadata: Dict[str, Any]


# ==================== NN COMPONENT ====================

class DispatcherNN:
    """
    Componente NN del Dispatcher.

    Funzioni:
    1. Router: fingerprint → mappa routing per LTM
    2. Embedder: distillato LLM → embedding per LTM
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_buckets: int = 16,
        use_learned_routing: bool = False
    ):
        self.embedding_dim = embedding_dim
        self.num_buckets = num_buckets
        self.use_learned_routing = use_learned_routing

        # Centroidi per routing (inizializzati random, poi learned)
        self._bucket_centroids: Optional[np.ndarray] = None
        self._routing_history: List[Tuple[np.ndarray, List[str]]] = []

    async def get_routing(
        self,
        fingerprint: Fingerprint,
        top_k: int = 3
    ) -> RoutingMap:
        """
        Dato un fingerprint, ritorna mappa di routing.
        "Dove cercare in LTM?"
        """
        vector = fingerprint.vector

        if self._bucket_centroids is None:
            # Prima query: inizializza centroidi
            self._init_centroids(vector.shape[0])

        # Calcola distanze dai centroidi
        distances = np.linalg.norm(
            self._bucket_centroids - vector,
            axis=1
        )

        # Top-k bucket più vicini
        nearest_indices = np.argsort(distances)[:top_k]
        buckets = [f"bucket_{i}" for i in nearest_indices]

        # Confidenza basata su distanza
        min_dist = distances[nearest_indices[0]]
        confidence = float(np.exp(-min_dist))

        # Aggiungi alla history per learning futuro
        self._routing_history.append((vector, buckets))

        return RoutingMap(
            buckets=buckets,
            node_ids=[],  # Sarà popolato da LTM
            confidence=confidence,
            strategy="hnsw"
        )

    def _init_centroids(self, dim: int) -> None:
        """Inizializza centroidi per routing"""
        self._bucket_centroids = np.random.randn(
            self.num_buckets, dim
        ).astype(np.float32)
        # Normalizza
        norms = np.linalg.norm(self._bucket_centroids, axis=1, keepdims=True)
        self._bucket_centroids /= (norms + 1e-8)

    def update_centroids(
        self,
        vectors: List[np.ndarray],
        labels: List[int]
    ) -> None:
        """Aggiorna centroidi con nuovi dati (online learning)"""
        if self._bucket_centroids is None:
            return

        for vec, label in zip(vectors, labels):
            if 0 <= label < self.num_buckets:
                # Moving average update
                alpha = 0.1
                self._bucket_centroids[label] = (
                    (1 - alpha) * self._bucket_centroids[label] +
                    alpha * vec
                )

    async def create_embedding(
        self,
        text: str,
        context: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Crea embedding da testo distillato.
        Per injection in LTM.

        TODO: Integrare sentence-transformers.
        """
        # Placeholder: hash-based
        import hashlib
        hash_bytes = hashlib.sha256(text.encode()).digest()
        np.random.seed(int.from_bytes(hash_bytes[:4], 'big'))
        return np.random.randn(self.embedding_dim).astype(np.float32)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "num_buckets": self.num_buckets,
            "embedding_dim": self.embedding_dim,
            "routing_history_size": len(self._routing_history),
            "centroids_initialized": self._bucket_centroids is not None
        }


# ==================== LLM COMPONENT ====================

class DispatcherLLM(ABC):
    """
    Componente LLM del Dispatcher (Abstract).

    Funzioni:
    1. Valuta risultati retrieval e genera risposte
    2. Distilla conoscenza da STM
    """

    @abstractmethod
    async def evaluate(
        self,
        fingerprint: Fingerprint,
        retrieval_result: RetrievalResult,
        context: Optional[Dict] = None
    ) -> EvaluationResult:
        """Valuta match e decide se/come rispondere"""
        pass

    @abstractmethod
    async def distill(
        self,
        data: List[Dict[str, Any]],
        topic: str
    ) -> DistillationResult:
        """Distilla dati da STM in summary + embedding"""
        pass

    @abstractmethod
    async def generate_response(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> str:
        """Genera risposta dato contesto"""
        pass


class LocalDispatcherLLM(DispatcherLLM):
    """
    Implementazione locale del Dispatcher LLM.
    Usa Ollama per inference locale.
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 512
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    async def _get_client(self):
        """Lazy init del client Ollama"""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.AsyncClient(host=self.base_url)
            except ImportError:
                print("Warning: ollama not installed, using mock")
                self._client = "mock"
        return self._client

    async def evaluate(
        self,
        fingerprint: Fingerprint,
        retrieval_result: RetrievalResult,
        context: Optional[Dict] = None
    ) -> EvaluationResult:
        """
        Valuta risultati retrieval.
        Decide se rispondere e con cosa.
        """
        # Se nessun match, non rispondere
        if not retrieval_result.matches:
            return EvaluationResult(
                should_respond=False,
                response=None,
                confidence=0.0,
                reasoning="No matches found",
                action="none"
            )

        best_match = retrieval_result.best_match
        if best_match is None:
            return EvaluationResult(
                should_respond=False,
                response=None,
                confidence=0.0,
                reasoning="No best match",
                action="none"
            )

        match_id, similarity = best_match

        # Threshold per risposta istintiva
        if similarity >= 0.85:
            # Match forte → risposta istintiva
            return EvaluationResult(
                should_respond=True,
                response=f"[Instinctive match: {match_id}]",
                confidence=similarity,
                reasoning=f"Strong match (sim={similarity:.3f})",
                action="respond"
            )
        elif similarity >= 0.7:
            # Match moderato → genera risposta con LLM
            client = await self._get_client()
            if client == "mock":
                response = f"[Mock response for match: {match_id}]"
            else:
                # Chiedi a LLM di elaborare
                response = await self._generate_with_context(
                    match_id=match_id,
                    similarity=similarity,
                    topic=fingerprint.topic,
                    context=context
                )

            return EvaluationResult(
                should_respond=True,
                response=response,
                confidence=similarity,
                reasoning=f"Moderate match, LLM elaborated",
                action="respond"
            )
        else:
            # Match debole → store per futuro
            return EvaluationResult(
                should_respond=False,
                response=None,
                confidence=similarity,
                reasoning=f"Weak match (sim={similarity:.3f}), storing",
                action="store"
            )

    async def distill(
        self,
        data: List[Dict[str, Any]],
        topic: str
    ) -> DistillationResult:
        """
        Distilla dati accumulati in STM.
        Crea summary + embedding per injection in LTM.
        """
        if not data:
            return DistillationResult(
                embedding=np.zeros(128, dtype=np.float32),
                summary="",
                topics=[topic],
                importance=0.0,
                metadata={}
            )

        # Crea prompt per distillazione
        data_str = "\n".join([str(d) for d in data[:10]])  # Limita

        client = await self._get_client()

        if client == "mock":
            summary = f"[Distilled {len(data)} items for topic: {topic}]"
        else:
            try:
                response = await client.chat(
                    model=self.model,
                    messages=[{
                        "role": "system",
                        "content": "Distill the following data into a concise summary. Extract key patterns and insights."
                    }, {
                        "role": "user",
                        "content": f"Topic: {topic}\n\nData:\n{data_str}"
                    }],
                    options={"temperature": 0.3}
                )
                summary = response["message"]["content"]
            except Exception as e:
                summary = f"[Distillation error: {e}]"

        # Crea embedding del summary
        embedding = await self._text_to_embedding(summary)

        return DistillationResult(
            embedding=embedding,
            summary=summary,
            topics=[topic],
            importance=min(len(data) / 10, 1.0),
            metadata={"source_count": len(data)}
        )

    async def generate_response(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> str:
        """Genera risposta con contesto"""
        client = await self._get_client()

        if client == "mock":
            return f"[Mock response for: {query[:50]}...]"

        try:
            response = await client.chat(
                model=self.model,
                messages=[{
                    "role": "system",
                    "content": f"Context: {context}"
                }, {
                    "role": "user",
                    "content": query
                }],
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            )
            return response["message"]["content"]
        except Exception as e:
            return f"[Error: {e}]"

    async def _generate_with_context(
        self,
        match_id: str,
        similarity: float,
        topic: str,
        context: Optional[Dict]
    ) -> str:
        """Genera risposta elaborata con contesto del match"""
        prompt = f"""Based on a memory match:
- Match ID: {match_id}
- Similarity: {similarity:.3f}
- Topic: {topic}
- Context: {context or 'None'}

Generate an appropriate response."""

        return await self.generate_response(prompt, context or {})

    async def _text_to_embedding(self, text: str) -> np.ndarray:
        """Converte testo in embedding"""
        # Placeholder: hash-based
        import hashlib
        hash_bytes = hashlib.sha256(text.encode()).digest()
        np.random.seed(int.from_bytes(hash_bytes[:4], 'big'))
        return np.random.randn(128).astype(np.float32)


# ==================== DISPATCHER (Combined) ====================

class Dispatcher:
    """
    Dispatcher completo: LLM + NN.
    Medium-Term Memory del sistema.
    """

    def __init__(
        self,
        llm: Optional[DispatcherLLM] = None,
        nn: Optional[DispatcherNN] = None,
        embedding_dim: int = 128
    ):
        self.llm = llm or LocalDispatcherLLM()
        self.nn = nn or DispatcherNN(embedding_dim=embedding_dim)
        self.ltm = None  # Verrà iniettato

        self._evaluation_history: List[EvaluationResult] = []
        self._distillation_queue: List[Dict] = []

    def set_ltm(self, ltm) -> None:
        """Inietta riferimento a LTM"""
        self.ltm = ltm

    # ==================== NN Functions ====================

    async def get_routing(self, fingerprint: Fingerprint) -> List[str]:
        """
        Instinctive chiede: "dove cerco?"
        Ritorna lista di bucket/nodi.
        """
        routing_map = await self.nn.get_routing(fingerprint)
        return routing_map.buckets

    # ==================== LLM Functions ====================

    async def evaluate(
        self,
        fingerprint: Fingerprint,
        retrieval_result: RetrievalResult
    ) -> EvaluationResult:
        """
        Valuta risultati retrieval.
        Instinctive passa risultati, Dispatcher decide.
        """
        result = await self.llm.evaluate(fingerprint, retrieval_result)
        self._evaluation_history.append(result)

        # Se action=store, aggiungi a coda distillazione
        if result.action == "store":
            self._distillation_queue.append({
                "fingerprint": fingerprint.to_dict(),
                "retrieval": {
                    "matches": retrieval_result.matches,
                    "confidence": result.confidence
                }
            })

        return result

    async def distill_and_inject(self, topic: str) -> Optional[str]:
        """
        Distilla dati in coda e inietta in LTM.
        STM → Dispatcher → LTM
        """
        if not self._distillation_queue:
            return None

        # Filtra per topic
        topic_data = [
            d for d in self._distillation_queue
            if d.get("fingerprint", {}).get("topic") == topic
        ]

        if not topic_data:
            return None

        # Distilla
        result = await self.llm.distill(topic_data, topic)

        # Inietta in LTM
        if self.ltm is not None and result.importance > 0.3:
            entry_id = await self.ltm.store(
                vector=result.embedding,
                metadata={
                    "summary": result.summary,
                    "topics": result.topics,
                    "importance": result.importance,
                    **result.metadata
                }
            )

            # Rimuovi dalla coda
            self._distillation_queue = [
                d for d in self._distillation_queue
                if d not in topic_data
            ]

            return entry_id

        return None

    # ==================== Stats ====================

    def get_stats(self) -> Dict[str, Any]:
        return {
            "nn": self.nn.get_stats(),
            "evaluations_count": len(self._evaluation_history),
            "distillation_queue_size": len(self._distillation_queue),
            "ltm_connected": self.ltm is not None
        }
