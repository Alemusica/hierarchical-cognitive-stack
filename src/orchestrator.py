"""
ORCHESTRATOR
============
Coordina il flusso completo del sistema.
Connette tutti i componenti e gestisce il ciclo.

Flusso:
1. Input → Instinctive
2. Instinctive ↔ STM (bidirezionale)
3. Instinctive → Dispatcher NN (routing)
4. Instinctive → LTM (retrieval)
5. Instinctive → Dispatcher LLM (valutazione)
6. STM → Dispatcher → LTM (distillazione periodica)
7. Output
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, AsyncIterator
from enum import Enum
import asyncio
from datetime import datetime
import time

from .stm import STM, STMState
from .ltm import LTM
from .dispatcher import Dispatcher, EvaluationResult
from .instinctive import InstinctiveLayer, Fingerprint, RetrievalResult


class SystemPhase(Enum):
    """Fasi del sistema"""
    IDLE = "idle"
    RECEIVING = "receiving"
    PROCESSING = "processing"
    RESPONDING = "responding"
    DISTILLING = "distilling"


@dataclass
class ProcessingResult:
    """Risultato processing completo"""
    input_id: str
    topic: str
    fingerprint_created: bool
    fingerprint_id: Optional[str]
    retrieval_result: Optional[RetrievalResult]
    evaluation: Optional[EvaluationResult]
    response: Optional[str]
    latency_ms: float
    phase_timings: Dict[str, float]


@dataclass
class SystemStats:
    """Statistiche sistema"""
    phase: SystemPhase
    total_inputs: int
    total_fingerprints: int
    total_retrievals: int
    total_responses: int
    avg_latency_ms: float
    uptime_seconds: float


class Orchestrator:
    """
    Orchestratore del sistema Hierarchical Memory.

    Gestisce il flusso completo:
    Input → Instinctive ↔ STM → Dispatcher → LTM → Output
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        use_surrealdb: bool = False,
        surrealdb_url: str = "ws://localhost:8000/rpc",
        distillation_interval: float = 60.0,  # secondi
        auto_distill: bool = True
    ):
        self.embedding_dim = embedding_dim

        # Inizializza componenti
        self.stm = STM(
            min_samples_per_topic=3,
            quality_threshold=0.6
        )

        if use_surrealdb:
            self.ltm = LTM.with_surrealdb(
                url=surrealdb_url,
                embedding_dim=embedding_dim
            )
        else:
            self.ltm = LTM.in_memory(embedding_dim=embedding_dim)

        self.dispatcher = Dispatcher(embedding_dim=embedding_dim)
        self.dispatcher.set_ltm(self.ltm)

        self.instinctive = InstinctiveLayer(
            stm=self.stm,
            dispatcher=self.dispatcher,
            ltm=self.ltm,
            embedding_dim=embedding_dim
        )

        # State
        self._phase = SystemPhase.IDLE
        self._start_time = datetime.now()
        self._total_inputs = 0
        self._total_fingerprints = 0
        self._total_retrievals = 0
        self._total_responses = 0
        self._latencies: List[float] = []

        # Distillation
        self.distillation_interval = distillation_interval
        self.auto_distill = auto_distill
        self._distillation_task: Optional[asyncio.Task] = None
        self._last_distillation = datetime.now()

        # Callbacks
        self._on_response_callbacks: List[Callable] = []
        self._on_fingerprint_callbacks: List[Callable] = []

    # ==================== MAIN API ====================

    async def process(
        self,
        input_data: Any,
        topic: str,
        metadata: Optional[Dict] = None
    ) -> ProcessingResult:
        """
        Processa singolo input attraverso il sistema completo.

        Args:
            input_data: Input (testo, vettore, etc.)
            topic: Topic/categoria semantica
            metadata: Metadati opzionali

        Returns:
            ProcessingResult con tutti i dettagli
        """
        start_time = time.perf_counter()
        phase_timings = {}

        input_id = f"input_{self._total_inputs}_{int(time.time()*1000)}"
        self._total_inputs += 1
        self._phase = SystemPhase.RECEIVING

        # 1. Instinctive processa input
        t0 = time.perf_counter()
        self._phase = SystemPhase.PROCESSING

        retrieval_result = await self.instinctive.process_input(
            input_data, topic, metadata
        )

        phase_timings["instinctive"] = (time.perf_counter() - t0) * 1000

        # Check se è stato creato fingerprint
        fingerprint_created = retrieval_result is not None
        fingerprint_id = None
        evaluation = None
        response = None

        if fingerprint_created:
            self._total_fingerprints += 1
            fingerprint_id = retrieval_result.fingerprint_id

            if retrieval_result.matches:
                self._total_retrievals += 1

            # 2. Dispatcher ha già valutato (chiamato da Instinctive)
            # Recupera ultima evaluation
            if self.dispatcher._evaluation_history:
                evaluation = self.dispatcher._evaluation_history[-1]

                if evaluation.should_respond:
                    self._phase = SystemPhase.RESPONDING
                    response = evaluation.response
                    self._total_responses += 1

                    # Notify callbacks
                    for cb in self._on_response_callbacks:
                        try:
                            cb(response, topic, retrieval_result)
                        except Exception as e:
                            print(f"Response callback error: {e}")

        # Calcola latenza totale
        total_latency = (time.perf_counter() - start_time) * 1000
        self._latencies.append(total_latency)

        self._phase = SystemPhase.IDLE

        return ProcessingResult(
            input_id=input_id,
            topic=topic,
            fingerprint_created=fingerprint_created,
            fingerprint_id=fingerprint_id,
            retrieval_result=retrieval_result,
            evaluation=evaluation,
            response=response,
            latency_ms=total_latency,
            phase_timings=phase_timings
        )

    async def process_stream(
        self,
        inputs: AsyncIterator[tuple],
    ) -> AsyncIterator[ProcessingResult]:
        """
        Processa stream di input.
        Yield risultati man mano.

        Args:
            inputs: AsyncIterator di (input_data, topic, metadata)
        """
        async for input_data, topic, metadata in inputs:
            result = await self.process(input_data, topic, metadata)
            yield result

    async def process_batch(
        self,
        inputs: List[tuple]
    ) -> List[ProcessingResult]:
        """
        Processa batch di input.

        Args:
            inputs: Lista di (input_data, topic, metadata)
        """
        results = []
        for input_data, topic, metadata in inputs:
            result = await self.process(input_data, topic, metadata)
            results.append(result)
        return results

    # ==================== DISTILLATION ====================

    async def distill_topic(self, topic: str) -> Optional[str]:
        """
        Forza distillazione per topic specifico.
        STM → Dispatcher → LTM
        """
        self._phase = SystemPhase.DISTILLING
        entry_id = await self.dispatcher.distill_and_inject(topic)
        self._last_distillation = datetime.now()
        self._phase = SystemPhase.IDLE
        return entry_id

    async def distill_all(self) -> Dict[str, Optional[str]]:
        """Distilla tutti i topic in STM"""
        state = self.stm.get_state()
        results = {}

        for topic in state.topics.keys():
            entry_id = await self.distill_topic(topic)
            results[topic] = entry_id

        return results

    async def _distillation_loop(self):
        """Loop automatico di distillazione"""
        while True:
            await asyncio.sleep(self.distillation_interval)

            try:
                await self.distill_all()
            except Exception as e:
                print(f"Distillation error: {e}")

    def start_auto_distillation(self):
        """Avvia distillazione automatica"""
        if self._distillation_task is None:
            self._distillation_task = asyncio.create_task(
                self._distillation_loop()
            )

    def stop_auto_distillation(self):
        """Ferma distillazione automatica"""
        if self._distillation_task:
            self._distillation_task.cancel()
            self._distillation_task = None

    # ==================== CALLBACKS ====================

    def on_response(self, callback: Callable) -> Callable[[], None]:
        """
        Registra callback per quando viene generata risposta.
        callback(response, topic, retrieval_result)
        """
        self._on_response_callbacks.append(callback)
        return lambda: self._on_response_callbacks.remove(callback)

    def on_fingerprint(self, callback: Callable) -> Callable[[], None]:
        """
        Registra callback per quando viene creato fingerprint.
        callback(fingerprint)
        """
        self._on_fingerprint_callbacks.append(callback)
        return lambda: self._on_fingerprint_callbacks.remove(callback)

    # ==================== STATE & STATS ====================

    def get_state(self) -> Dict[str, Any]:
        """Ritorna stato completo del sistema"""
        return {
            "phase": self._phase.value,
            "stm": self.stm.get_stats(),
            "ltm": self.ltm.get_stats(),
            "dispatcher": self.dispatcher.get_stats(),
            "instinctive": self.instinctive.get_stats()
        }

    def get_stats(self) -> SystemStats:
        """Ritorna statistiche sistema"""
        uptime = (datetime.now() - self._start_time).total_seconds()
        avg_latency = sum(self._latencies) / len(self._latencies) if self._latencies else 0

        return SystemStats(
            phase=self._phase,
            total_inputs=self._total_inputs,
            total_fingerprints=self._total_fingerprints,
            total_retrievals=self._total_retrievals,
            total_responses=self._total_responses,
            avg_latency_ms=avg_latency,
            uptime_seconds=uptime
        )

    # ==================== LIFECYCLE ====================

    async def start(self):
        """Avvia il sistema"""
        if self.auto_distill:
            self.start_auto_distillation()

    async def stop(self):
        """Ferma il sistema"""
        self.stop_auto_distillation()

        # Final distillation
        await self.distill_all()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


# ==================== FACTORY FUNCTIONS ====================

def create_system(
    embedding_dim: int = 128,
    use_surrealdb: bool = False,
    **kwargs
) -> Orchestrator:
    """
    Factory per creare sistema completo.

    Args:
        embedding_dim: Dimensione embedding
        use_surrealdb: Usa SurrealDB invece di in-memory
        **kwargs: Altri parametri per Orchestrator
    """
    return Orchestrator(
        embedding_dim=embedding_dim,
        use_surrealdb=use_surrealdb,
        **kwargs
    )


async def quick_test():
    """Test rapido del sistema"""
    system = create_system(embedding_dim=64)

    async with system:
        # Test input
        result = await system.process(
            "Come resetto la password?",
            topic="auth"
        )
        print(f"Result: {result}")

        # Più input stesso topic
        for i in range(5):
            await system.process(
                f"Password reset help {i}",
                topic="auth"
            )

        # Stats
        stats = system.get_stats()
        print(f"Stats: {stats}")


if __name__ == "__main__":
    asyncio.run(quick_test())
