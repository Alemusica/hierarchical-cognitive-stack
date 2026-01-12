"""
SHORT-TERM MEMORY (STM)
=======================
Vector DB raw + labeling per topic.
RAM organizzata come appoggio per Instinctive.

Bidirezionalità (pattern da Phonon-UI):
- Instinctive buffer dati → STM
- STM notifica stato → Instinctive
- "Ho abbastanza segnale per fingerprint?"

Analogia Shazam:
- Se rumore → più tempo, più memoria
- STM comunica quando topic è "ready"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import numpy as np
from datetime import datetime
import asyncio


class STMPhase(Enum):
    """Fasi dello STM (come PourState.phase in Phonon-UI)"""
    BUFFERING = "buffering"
    READY = "ready"
    PROCESSING = "processing"
    IDLE = "idle"


@dataclass
class TopicBuffer:
    """Buffer per singolo topic con metriche qualità"""
    topic: str
    vectors: List[np.ndarray] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    metadata: List[Dict[str, Any]] = field(default_factory=list)

    # Metriche qualità (per decidere se ready)
    quality: float = 0.0          # 0-1, qualità segnale
    noise_level: float = 0.0      # 0-1, rumore stimato
    ready: bool = False           # fits equivalent
    min_samples: int = 5          # minimo per ready
    quality_threshold: float = 0.7

    def add(self, vector: np.ndarray, meta: Optional[Dict] = None) -> None:
        """Aggiunge vettore al buffer"""
        self.vectors.append(vector)
        self.timestamps.append(datetime.now())
        self.metadata.append(meta or {})
        self._update_quality()

    def _update_quality(self) -> None:
        """Aggiorna metriche qualità"""
        n = len(self.vectors)
        if n == 0:
            self.quality = 0.0
            self.noise_level = 1.0
            self.ready = False
            return

        # Qualità = f(numero samples, varianza, consistenza)
        sample_score = min(n / self.min_samples, 1.0)

        if n >= 2:
            # Stima rumore dalla varianza tra vettori consecutivi
            diffs = [np.linalg.norm(self.vectors[i] - self.vectors[i-1])
                     for i in range(1, n)]
            self.noise_level = min(np.mean(diffs) / 2.0, 1.0)
        else:
            self.noise_level = 0.5

        consistency_score = 1.0 - self.noise_level
        self.quality = (sample_score * 0.6 + consistency_score * 0.4)
        self.ready = (n >= self.min_samples and
                      self.quality >= self.quality_threshold)

    def get_aggregated_vector(self) -> Optional[np.ndarray]:
        """Ritorna vettore aggregato per fingerprinting"""
        if not self.vectors:
            return None
        return np.mean(self.vectors, axis=0)

    def clear(self) -> None:
        """Svuota buffer dopo fingerprinting"""
        self.vectors.clear()
        self.timestamps.clear()
        self.metadata.clear()
        self.quality = 0.0
        self.noise_level = 0.0
        self.ready = False


@dataclass
class STMState:
    """Stato completo STM (per notifiche bidirezionali)"""
    phase: STMPhase
    topics: Dict[str, TopicBuffer]
    ready_topics: List[str]
    total_vectors: int
    last_update: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "ready_topics": self.ready_topics,
            "total_vectors": self.total_vectors,
            "topic_stats": {
                topic: {
                    "count": len(buf.vectors),
                    "quality": buf.quality,
                    "noise": buf.noise_level,
                    "ready": buf.ready
                }
                for topic, buf in self.topics.items()
            }
        }


class STM:
    """
    Short-Term Memory

    Vector DB raw con labeling per topic.
    Bidirezionale: Instinctive chiede stato, STM notifica.
    """

    def __init__(
        self,
        min_samples_per_topic: int = 5,
        quality_threshold: float = 0.7,
        max_buffer_size: int = 1000
    ):
        self.min_samples = min_samples_per_topic
        self.quality_threshold = quality_threshold
        self.max_buffer_size = max_buffer_size

        self._topics: Dict[str, TopicBuffer] = {}
        self._phase = STMPhase.IDLE
        self._observers: List[Callable[[STMState], None]] = []
        self._async_observers: List[Callable[[STMState], Any]] = []

    # ==================== BUFFER (Instinctive → STM) ====================

    def buffer(
        self,
        vector: np.ndarray,
        topic: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Instinctive manda dati a STM.

        Args:
            vector: Vettore da bufferizzare
            topic: Topic/categoria semantica
            metadata: Metadati opzionali
        """
        if topic not in self._topics:
            self._topics[topic] = TopicBuffer(
                topic=topic,
                min_samples=self.min_samples,
                quality_threshold=self.quality_threshold
            )

        self._topics[topic].add(vector, metadata)
        self._phase = STMPhase.BUFFERING

        # Notifica observers del cambio stato
        self._notify_observers()

    def buffer_batch(
        self,
        vectors: List[np.ndarray],
        topics: List[str],
        metadata: Optional[List[Dict]] = None
    ) -> None:
        """Buffer multipli vettori"""
        metadata = metadata or [{}] * len(vectors)
        for vec, topic, meta in zip(vectors, topics, metadata):
            self.buffer(vec, topic, meta)

    # ==================== STATE QUERY (STM → Instinctive) ====================

    def get_state(self) -> STMState:
        """
        Instinctive chiede stato a STM.
        Bidirezionalità: "Hai abbastanza segnale?"
        """
        ready_topics = [
            topic for topic, buf in self._topics.items()
            if buf.ready
        ]

        total_vectors = sum(len(buf.vectors) for buf in self._topics.values())

        return STMState(
            phase=self._phase,
            topics=self._topics.copy(),
            ready_topics=ready_topics,
            total_vectors=total_vectors,
            last_update=datetime.now()
        )

    def is_topic_ready(self, topic: str) -> bool:
        """Check rapido se topic specifico è ready"""
        if topic not in self._topics:
            return False
        return self._topics[topic].ready

    def get_topic_quality(self, topic: str) -> float:
        """Ritorna qualità segnale per topic (0-1)"""
        if topic not in self._topics:
            return 0.0
        return self._topics[topic].quality

    def get_ready_topics(self) -> List[str]:
        """Lista topic pronti per fingerprinting"""
        return [t for t, buf in self._topics.items() if buf.ready]

    # ==================== DATA RETRIEVAL ====================

    def get_topic_data(self, topic: str) -> Optional[TopicBuffer]:
        """Ritorna buffer completo per topic"""
        return self._topics.get(topic)

    def get_aggregated_vector(self, topic: str) -> Optional[np.ndarray]:
        """Ritorna vettore aggregato per fingerprinting"""
        if topic not in self._topics:
            return None
        return self._topics[topic].get_aggregated_vector()

    def consume_topic(self, topic: str) -> Optional[TopicBuffer]:
        """
        Consuma e ritorna dati topic (dopo fingerprinting).
        Svuota il buffer per quel topic.
        """
        if topic not in self._topics:
            return None

        buffer = self._topics[topic]
        data = TopicBuffer(
            topic=topic,
            vectors=buffer.vectors.copy(),
            timestamps=buffer.timestamps.copy(),
            metadata=buffer.metadata.copy(),
            quality=buffer.quality,
            noise_level=buffer.noise_level,
            ready=buffer.ready
        )

        buffer.clear()
        self._notify_observers()

        return data

    # ==================== OBSERVER PATTERN (Notifiche) ====================

    def subscribe(self, callback: Callable[[STMState], None]) -> Callable[[], None]:
        """
        Sottoscrivi a notifiche cambio stato.
        Ritorna funzione unsubscribe.
        """
        self._observers.append(callback)
        return lambda: self._observers.remove(callback)

    def subscribe_async(
        self,
        callback: Callable[[STMState], Any]
    ) -> Callable[[], None]:
        """Sottoscrivi con callback async"""
        self._async_observers.append(callback)
        return lambda: self._async_observers.remove(callback)

    def _notify_observers(self) -> None:
        """Notifica tutti gli observers del nuovo stato"""
        state = self.get_state()

        for callback in self._observers:
            try:
                callback(state)
            except Exception as e:
                print(f"STM observer error: {e}")

        # Async observers
        for callback in self._async_observers:
            try:
                result = callback(state)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                print(f"STM async observer error: {e}")

    # ==================== MANAGEMENT ====================

    def clear_topic(self, topic: str) -> None:
        """Svuota buffer per topic specifico"""
        if topic in self._topics:
            self._topics[topic].clear()
            self._notify_observers()

    def clear_all(self) -> None:
        """Svuota tutti i buffer"""
        for buf in self._topics.values():
            buf.clear()
        self._phase = STMPhase.IDLE
        self._notify_observers()

    def get_stats(self) -> Dict[str, Any]:
        """Statistiche globali STM"""
        return {
            "phase": self._phase.value,
            "num_topics": len(self._topics),
            "total_vectors": sum(len(b.vectors) for b in self._topics.values()),
            "ready_topics": len(self.get_ready_topics()),
            "topics": {
                t: {
                    "vectors": len(b.vectors),
                    "quality": round(b.quality, 3),
                    "noise": round(b.noise_level, 3),
                    "ready": b.ready
                }
                for t, b in self._topics.items()
            }
        }
