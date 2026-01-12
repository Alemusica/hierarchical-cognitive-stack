"""
INSTINCTIVE LAYER
=================
Primo contatto con input. Crea fingerprint e orchestra retrieval.

Funzioni:
1. Riceve input multipli
2. Buffer in STM (bidirezionale)
3. Quando STM ready → crea fingerprint
4. Chiede mappa routing a Dispatcher NN
5. Retrieval diretto da LTM
6. Passa risultati a Dispatcher LLM

NON fa valutazione/decisione - quello è Dispatcher.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import numpy as np
from datetime import datetime
import hashlib

from .stm import STM, STMState, TopicBuffer


@dataclass
class Fingerprint:
    """
    Fingerprint percettivo (Shazam-like).
    Hash semantico dell'input per matching veloce.
    """
    id: str
    vector: np.ndarray
    topic: str
    created_at: datetime
    source_vectors_count: int
    quality: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def similarity(self, other: 'Fingerprint') -> float:
        """Cosine similarity con altro fingerprint"""
        dot = np.dot(self.vector, other.vector)
        norm = np.linalg.norm(self.vector) * np.linalg.norm(other.vector)
        if norm == 0:
            return 0.0
        return float(dot / norm)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "created_at": self.created_at.isoformat(),
            "quality": self.quality,
            "vector_dim": len(self.vector),
            "metadata": self.metadata
        }


@dataclass
class RetrievalRequest:
    """Richiesta di retrieval per Dispatcher/LTM"""
    fingerprint: Fingerprint
    routing_hint: Optional[List[str]] = None  # bucket/nodi suggeriti
    top_k: int = 5
    threshold: float = 0.7


@dataclass
class RetrievalResult:
    """Risultato retrieval da LTM"""
    fingerprint_id: str
    matches: List[Tuple[str, float]]  # [(match_id, similarity), ...]
    best_match: Optional[Tuple[str, float]] = None
    routing_used: Optional[List[str]] = None
    latency_ms: float = 0.0


@dataclass
class InstinctiveState:
    """Stato del layer instinctive"""
    pending_fingerprints: int
    processed_count: int
    last_fingerprint_time: Optional[datetime]
    stm_state: Optional[STMState]


class InstinctiveLayer:
    """
    Layer Istintivo - Primo contatto con input.

    Crea fingerprint e orchestra il flusso, ma NON decide.
    La decisione/valutazione è del Dispatcher.
    """

    def __init__(
        self,
        stm: STM,
        dispatcher=None,  # Verrà iniettato
        ltm=None,         # Verrà iniettato
        embedding_dim: int = 128
    ):
        self.stm = stm
        self.dispatcher = dispatcher
        self.ltm = ltm
        self.embedding_dim = embedding_dim

        self._processed_count = 0
        self._last_fingerprint_time: Optional[datetime] = None
        self._pending_fingerprints: List[Fingerprint] = []

        # Subscribe a STM per notifiche (bidirezionalità)
        self.stm.subscribe(self._on_stm_state_change)

    # ==================== INPUT PROCESSING ====================

    async def process_input(
        self,
        input_data: Any,
        topic: str,
        metadata: Optional[Dict] = None
    ) -> Optional[RetrievalResult]:
        """
        Processa singolo input.

        1. Embedding dell'input
        2. Buffer in STM
        3. Se topic ready → fingerprint → retrieval
        """
        # 1. Crea embedding (placeholder - userà sentence-transformers)
        vector = self._create_embedding(input_data)

        # 2. Buffer in STM
        self.stm.buffer(vector, topic, metadata)

        # 3. Check se topic ready (bidirezionale query a STM)
        if self.stm.is_topic_ready(topic):
            return await self._process_ready_topic(topic)

        return None

    async def process_batch(
        self,
        inputs: List[Tuple[Any, str, Optional[Dict]]]
    ) -> List[Optional[RetrievalResult]]:
        """Processa batch di input: [(data, topic, meta), ...]"""
        results = []
        for data, topic, meta in inputs:
            result = await self.process_input(data, topic, meta)
            results.append(result)
        return results

    # ==================== FINGERPRINTING ====================

    async def _process_ready_topic(self, topic: str) -> Optional[RetrievalResult]:
        """
        Topic ha segnale sufficiente → crea fingerprint e retrieval.
        """
        # 1. Crea fingerprint
        fingerprint = self._create_fingerprint(topic)
        if fingerprint is None:
            return None

        self._pending_fingerprints.append(fingerprint)
        self._last_fingerprint_time = datetime.now()

        # 2. Chiedi routing a Dispatcher NN (se disponibile)
        routing = await self._get_routing(fingerprint)

        # 3. Retrieval da LTM
        result = await self._retrieve_from_ltm(fingerprint, routing)

        # 4. Feed risultato a STM (per contesto)
        if result and result.matches:
            self._feed_result_to_stm(result, topic)

        # 5. Passa a Dispatcher LLM per valutazione
        if self.dispatcher and result:
            await self.dispatcher.evaluate(fingerprint, result)

        self._processed_count += 1
        return result

    def _create_fingerprint(self, topic: str) -> Optional[Fingerprint]:
        """
        Crea fingerprint da dati aggregati in STM per topic.
        """
        # Consuma dati da STM
        topic_data = self.stm.consume_topic(topic)
        if topic_data is None or not topic_data.vectors:
            return None

        # Aggrega vettori (media pesata per recency)
        aggregated = self._aggregate_vectors(topic_data.vectors)

        # Genera ID univoco
        fp_id = self._generate_fingerprint_id(aggregated, topic)

        return Fingerprint(
            id=fp_id,
            vector=aggregated,
            topic=topic,
            created_at=datetime.now(),
            source_vectors_count=len(topic_data.vectors),
            quality=topic_data.quality,
            metadata={
                "noise_level": topic_data.noise_level,
                "timestamps": [t.isoformat() for t in topic_data.timestamps[-5:]]
            }
        )

    def _aggregate_vectors(
        self,
        vectors: List[np.ndarray],
        recency_weight: float = 0.3
    ) -> np.ndarray:
        """
        Aggrega vettori con peso per recency.
        Vettori più recenti pesano di più.
        """
        n = len(vectors)
        if n == 1:
            return vectors[0]

        # Pesi esponenziali per recency
        weights = np.array([
            (1 - recency_weight) + recency_weight * (i / (n - 1))
            for i in range(n)
        ])
        weights /= weights.sum()

        # Media pesata
        return np.average(vectors, axis=0, weights=weights)

    def _generate_fingerprint_id(
        self,
        vector: np.ndarray,
        topic: str
    ) -> str:
        """Genera ID univoco per fingerprint"""
        data = f"{topic}:{vector.tobytes().hex()[:32]}:{datetime.now().timestamp()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    # ==================== ROUTING & RETRIEVAL ====================

    async def _get_routing(
        self,
        fingerprint: Fingerprint
    ) -> Optional[List[str]]:
        """
        Chiede mappa/routing a Dispatcher NN.
        "Ho questo fingerprint, dove cerco in LTM?"
        """
        if self.dispatcher is None:
            return None

        try:
            routing = await self.dispatcher.get_routing(fingerprint)
            return routing
        except Exception as e:
            print(f"Routing error: {e}")
            return None

    async def _retrieve_from_ltm(
        self,
        fingerprint: Fingerprint,
        routing: Optional[List[str]] = None
    ) -> RetrievalResult:
        """
        Retrieval DIRETTO da LTM con routing.
        """
        import time
        start = time.perf_counter()

        matches = []

        if self.ltm is not None:
            try:
                # Query LTM con fingerprint
                matches = await self.ltm.search(
                    vector=fingerprint.vector,
                    top_k=5,
                    routing_hint=routing
                )
            except Exception as e:
                print(f"LTM retrieval error: {e}")

        latency = (time.perf_counter() - start) * 1000

        best_match = matches[0] if matches else None

        return RetrievalResult(
            fingerprint_id=fingerprint.id,
            matches=matches,
            best_match=best_match,
            routing_used=routing,
            latency_ms=latency
        )

    def _feed_result_to_stm(
        self,
        result: RetrievalResult,
        original_topic: str
    ) -> None:
        """
        Feed risultati retrieval a STM per contesto.
        Crea un "echo" del match trovato.
        """
        if not result.matches:
            return

        # Crea vettore "echo" dai match (per contesto futuro)
        # Questo arricchisce STM con informazioni dal retrieval
        echo_meta = {
            "type": "ltm_echo",
            "original_topic": original_topic,
            "match_id": result.best_match[0] if result.best_match else None,
            "similarity": result.best_match[1] if result.best_match else 0.0
        }

        # L'echo va in un topic speciale "_context"
        # (non triggera nuovo fingerprinting)
        # Per ora skip, sarà implementato con logica più sofisticata

    # ==================== STM OBSERVER ====================

    def _on_stm_state_change(self, state: STMState) -> None:
        """
        Callback quando STM notifica cambio stato.
        Bidirezionalità: STM → Instinctive.
        """
        # Check se ci sono nuovi topic ready
        for topic in state.ready_topics:
            # Potrebbe triggerare processing automatico
            # Per ora solo logging
            pass

    # ==================== EMBEDDING ====================

    def _create_embedding(self, input_data: Any) -> np.ndarray:
        """
        Crea embedding dall'input.

        TODO: Integrare sentence-transformers o altro encoder.
        Per ora: placeholder con hash-based embedding.
        """
        if isinstance(input_data, np.ndarray):
            # Già un vettore
            if len(input_data) == self.embedding_dim:
                return input_data
            # Resize/project
            return self._project_vector(input_data)

        if isinstance(input_data, str):
            # Placeholder: hash-based embedding per testo
            return self._text_to_vector(input_data)

        if isinstance(input_data, (list, tuple)):
            return np.array(input_data, dtype=np.float32)[:self.embedding_dim]

        # Fallback
        return np.random.randn(self.embedding_dim).astype(np.float32)

    def _text_to_vector(self, text: str) -> np.ndarray:
        """
        Placeholder: converte testo in vettore.
        TODO: Usare sentence-transformers.
        """
        # Hash-based per ora (deterministic ma non semantico)
        hash_bytes = hashlib.sha256(text.encode()).digest()
        # Espandi hash a embedding_dim
        np.random.seed(int.from_bytes(hash_bytes[:4], 'big'))
        return np.random.randn(self.embedding_dim).astype(np.float32)

    def _project_vector(self, vector: np.ndarray) -> np.ndarray:
        """Proietta vettore a embedding_dim"""
        if len(vector) >= self.embedding_dim:
            return vector[:self.embedding_dim].astype(np.float32)
        # Pad con zeri
        result = np.zeros(self.embedding_dim, dtype=np.float32)
        result[:len(vector)] = vector
        return result

    # ==================== STATE ====================

    def get_state(self) -> InstinctiveState:
        """Ritorna stato corrente"""
        return InstinctiveState(
            pending_fingerprints=len(self._pending_fingerprints),
            processed_count=self._processed_count,
            last_fingerprint_time=self._last_fingerprint_time,
            stm_state=self.stm.get_state()
        )

    def get_stats(self) -> Dict[str, Any]:
        """Statistiche layer"""
        return {
            "processed_count": self._processed_count,
            "pending_fingerprints": len(self._pending_fingerprints),
            "last_fingerprint": self._last_fingerprint_time.isoformat() if self._last_fingerprint_time else None,
            "embedding_dim": self.embedding_dim,
            "stm_stats": self.stm.get_stats()
        }
