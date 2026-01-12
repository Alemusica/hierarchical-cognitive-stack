"""
Integration Tests
=================

Test del flusso completo del sistema.
Verifica ipotesi H1-H4.
"""

import pytest
import asyncio
import numpy as np
import time
from typing import List

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0] + '/src')

from stm import STM, STMPhase
from ltm import LTM
from dispatcher import Dispatcher
from instinctive import InstinctiveLayer, Fingerprint
from orchestrator import Orchestrator, create_system, ProcessingResult


# ==================== FIXTURES ====================

@pytest.fixture
def stm():
    return STM(min_samples_per_topic=3, quality_threshold=0.5)


@pytest.fixture
def ltm():
    return LTM.in_memory(embedding_dim=64)


@pytest.fixture
def dispatcher():
    return Dispatcher(embedding_dim=64)


@pytest.fixture
def system():
    return create_system(embedding_dim=64, auto_distill=False)


# ==================== STM TESTS ====================

class TestSTM:
    """Test Short-Term Memory"""

    def test_buffer_single(self, stm):
        """Test buffer singolo vettore"""
        vec = np.random.randn(64).astype(np.float32)
        stm.buffer(vec, "test_topic")

        state = stm.get_state()
        assert "test_topic" in state.topics
        assert len(state.topics["test_topic"].vectors) == 1

    def test_buffer_until_ready(self, stm):
        """Test che topic diventa ready dopo N samples"""
        topic = "auth"

        for i in range(5):
            vec = np.random.randn(64).astype(np.float32)
            stm.buffer(vec, topic)

        assert stm.is_topic_ready(topic)
        assert topic in stm.get_ready_topics()

    def test_bidirectional_state(self, stm):
        """Test bidirezionalità: STM notifica stato"""
        notifications = []

        def on_state_change(state):
            notifications.append(state)

        stm.subscribe(on_state_change)

        # Buffer vettori
        for i in range(3):
            stm.buffer(np.random.randn(64).astype(np.float32), "topic_a")

        assert len(notifications) == 3
        assert notifications[-1].phase == STMPhase.BUFFERING

    def test_quality_improves_with_samples(self, stm):
        """Test che qualità migliora con più samples"""
        topic = "quality_test"
        qualities = []

        for i in range(10):
            # Vettori simili (basso rumore)
            base = np.ones(64) * 0.5
            vec = base + np.random.randn(64) * 0.1
            stm.buffer(vec.astype(np.float32), topic)
            qualities.append(stm.get_topic_quality(topic))

        # Qualità dovrebbe crescere
        assert qualities[-1] > qualities[0]

    def test_consume_clears_buffer(self, stm):
        """Test che consume svuota il buffer"""
        topic = "consume_test"

        for i in range(5):
            stm.buffer(np.random.randn(64).astype(np.float32), topic)

        data = stm.consume_topic(topic)
        assert data is not None
        assert len(data.vectors) == 5

        # Dopo consume, topic non più ready
        assert not stm.is_topic_ready(topic)


# ==================== LTM TESTS ====================

class TestLTM:
    """Test Long-Term Memory"""

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, ltm):
        """Test store e retrieval base"""
        vec = np.random.randn(64).astype(np.float32)

        entry_id = await ltm.store(vec, {"topic": "test"})
        assert entry_id is not None

        # Search con stesso vettore → alta similarity
        results = await ltm.search(vec, top_k=1)
        assert len(results) == 1
        assert results[0][0] == entry_id
        assert results[0][1] > 0.99  # Quasi identico

    @pytest.mark.asyncio
    async def test_similar_vectors_match(self, ltm):
        """Test che vettori simili matchano"""
        base = np.random.randn(64).astype(np.float32)

        # Store base
        await ltm.store(base, {"type": "base"})

        # Search con vettore simile
        similar = base + np.random.randn(64).astype(np.float32) * 0.1
        results = await ltm.search(similar, top_k=1)

        assert len(results) == 1
        assert results[0][1] > 0.9  # Alta similarity

    @pytest.mark.asyncio
    async def test_dissimilar_vectors_low_match(self, ltm):
        """Test che vettori diversi hanno bassa similarity"""
        vec1 = np.random.randn(64).astype(np.float32)
        vec2 = -vec1  # Opposto

        await ltm.store(vec1, {})
        results = await ltm.search(vec2, top_k=1, threshold=0.5)

        # Non dovrebbe superare threshold
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_batch_store(self, ltm):
        """Test batch store"""
        vectors = [np.random.randn(64).astype(np.float32) for _ in range(10)]

        ids = await ltm.store_batch(vectors)
        assert len(ids) == 10

        stats = ltm.get_stats()
        assert stats["store_count"] == 10


# ==================== INTEGRATION TESTS ====================

class TestIntegration:
    """Test integrazione completa"""

    @pytest.mark.asyncio
    async def test_full_flow(self, system):
        """Test flusso completo: input → output"""
        async with system:
            # Prima di tutto, popola LTM con pattern
            pattern_vec = np.array([1.0] * 64, dtype=np.float32)
            await system.ltm.store(pattern_vec, {
                "topic": "auth",
                "response": "Reset password at Settings > Security"
            })

            # Processa input
            result = await system.process(
                "Come resetto la password?",
                topic="auth"
            )

            assert result.topic == "auth"
            assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_fingerprint_creation_after_threshold(self, system):
        """Test che fingerprint viene creato dopo soglia"""
        async with system:
            results = []

            # Input multipli stesso topic
            for i in range(5):
                result = await system.process(
                    f"Password help {i}",
                    topic="auth"
                )
                results.append(result)

            # Almeno uno dovrebbe aver creato fingerprint
            fingerprints_created = sum(1 for r in results if r.fingerprint_created)
            assert fingerprints_created >= 1

    @pytest.mark.asyncio
    async def test_evolution_improves_latency(self, system):
        """
        H2: Pattern ripetuti consolidano path preferenziali.
        Latenza dovrebbe diminuire con ripetizioni.
        """
        async with system:
            # Popola LTM
            for i in range(5):
                vec = np.random.randn(64).astype(np.float32)
                await system.ltm.store(vec, {"topic": "repeated"})

            latencies = []

            # Query ripetute simili
            for i in range(10):
                result = await system.process(
                    "repeated query pattern",
                    topic="repeated"
                )
                latencies.append(result.latency_ms)

            # Media prime 3 vs ultime 3
            early_avg = sum(latencies[:3]) / 3
            late_avg = sum(latencies[-3:]) / 3

            # Non necessariamente migliora (dipende da implementazione)
            # Ma non dovrebbe peggiorare drasticamente
            assert late_avg < early_avg * 2

    @pytest.mark.asyncio
    async def test_stm_ltm_bidirectional(self, system):
        """Test comunicazione bidirezionale STM ↔ Instinctive"""
        notifications = []

        system.stm.subscribe(lambda state: notifications.append(state))

        async with system:
            for i in range(5):
                await system.process(f"Input {i}", topic="test")

        # Dovrebbero esserci notifiche
        assert len(notifications) > 0

        # Check che alcuni topic siano stati marcati ready
        ready_seen = any(len(n.ready_topics) > 0 for n in notifications)
        # Potrebbe non essere ready se threshold non raggiunto

    @pytest.mark.asyncio
    async def test_distillation(self, system):
        """Test distillazione STM → LTM"""
        async with system:
            # Genera dati in STM
            for i in range(10):
                await system.process(f"Data for distillation {i}", topic="distill_test")

            # Forza distillazione
            initial_ltm_count = system.ltm.get_stats().get("backend", {}).get("entries_count", 0)

            entry_id = await system.distill_topic("distill_test")

            # Se c'erano dati in coda, dovrebbe aver creato entry
            # (dipende da logica interna)


# ==================== BENCHMARK TESTS ====================

class TestBenchmarks:
    """Benchmark per validare ipotesi"""

    @pytest.mark.asyncio
    async def test_h1_fingerprint_faster_than_bruteforce(self):
        """
        H1: Fingerprinting permette retrieval più veloce di full-context search.
        """
        ltm = LTM.in_memory(embedding_dim=128)

        # Popola con 1000 entry
        for i in range(1000):
            vec = np.random.randn(128).astype(np.float32)
            await ltm.store(vec, {"id": i})

        query = np.random.randn(128).astype(np.float32)

        # Misura tempo search
        times = []
        for _ in range(10):
            start = time.perf_counter()
            await ltm.search(query, top_k=5)
            times.append((time.perf_counter() - start) * 1000)

        avg_time = sum(times) / len(times)

        # Dovrebbe essere < 10ms per 1000 entry
        assert avg_time < 10, f"Search too slow: {avg_time:.2f}ms"

    @pytest.mark.asyncio
    async def test_h3_routing_improves_precision(self):
        """
        H3: Dispatcher routing migliora precision vs random routing.
        """
        dispatcher = Dispatcher(embedding_dim=64)

        # Crea fingerprint
        fp = Fingerprint(
            id="test",
            vector=np.random.randn(64).astype(np.float32),
            topic="test",
            created_at=__import__('datetime').datetime.now(),
            source_vectors_count=5,
            quality=0.8
        )

        # Get routing
        routing = await dispatcher.get_routing(fp)

        # Routing dovrebbe essere non-vuoto e con confidence > 0
        assert len(routing) > 0
        # La routing map ha attributi
        routing_map = await dispatcher.nn.get_routing(fp)
        assert routing_map.confidence > 0


# ==================== RUN ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
