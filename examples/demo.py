"""
Demo: Hierarchical Memory System
================================

Esempio completo del flusso:
Input → Instinctive ↔ STM → Dispatcher → LTM → Output
"""

import asyncio
import numpy as np
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0] + '/src')

from orchestrator import create_system


async def demo_basic():
    """Demo base: singolo input"""
    print("=" * 60)
    print("DEMO 1: Basic Flow")
    print("=" * 60)

    system = create_system(embedding_dim=64, auto_distill=False)

    async with system:
        result = await system.process(
            "Come resetto la password?",
            topic="auth"
        )

        print(f"Input processato")
        print(f"  Topic: {result.topic}")
        print(f"  Fingerprint creato: {result.fingerprint_created}")
        print(f"  Latenza: {result.latency_ms:.2f}ms")

        if result.response:
            print(f"  Risposta: {result.response}")


async def demo_accumulation():
    """Demo: accumulo in STM fino a fingerprint"""
    print("\n" + "=" * 60)
    print("DEMO 2: STM Accumulation → Fingerprint")
    print("=" * 60)

    system = create_system(
        embedding_dim=64,
        auto_distill=False
    )

    # Abbassa soglia per demo
    system.stm.quality_threshold = 0.3
    system.stm.min_samples = 3

    async with system:
        inputs = [
            "Come resetto la password?",
            "Password dimenticata",
            "Non ricordo la mia password",
            "Reset credenziali",
            "Voglio cambiare password"
        ]

        print("\nProcesso input sequenziali (stesso topic):")
        for i, text in enumerate(inputs):
            result = await system.process(text, topic="auth")

            state = system.stm.get_state()
            quality = system.stm.get_topic_quality("auth")

            print(f"\n  [{i+1}] '{text[:30]}...'")
            print(f"      STM quality: {quality:.2f}")
            print(f"      STM ready: {system.stm.is_topic_ready('auth')}")
            print(f"      Fingerprint creato: {result.fingerprint_created}")

            if result.fingerprint_created:
                print(f"      → Fingerprint ID: {result.fingerprint_id}")


async def demo_ltm_matching():
    """Demo: popola LTM e verifica matching"""
    print("\n" + "=" * 60)
    print("DEMO 3: LTM Matching")
    print("=" * 60)

    system = create_system(embedding_dim=64, auto_distill=False)

    async with system:
        # Popola LTM con pattern noti
        print("\nPopolo LTM con pattern...")
        patterns = [
            (np.random.randn(64).astype(np.float32), {"topic": "auth", "response": "Reset at Settings > Security"}),
            (np.random.randn(64).astype(np.float32), {"topic": "billing", "response": "Check invoices at Account > Billing"}),
            (np.random.randn(64).astype(np.float32), {"topic": "support", "response": "Contact support@example.com"}),
        ]

        for vec, meta in patterns:
            await system.ltm.store(vec, meta)

        print(f"  LTM entries: {system.ltm.get_stats()['store_count']}")

        # Query
        print("\nQuery LTM con vettore simile al primo pattern...")
        query = patterns[0][0] + np.random.randn(64).astype(np.float32) * 0.1
        matches = await system.ltm.search(query, top_k=3)

        print(f"  Matches trovati: {len(matches)}")
        for match_id, sim in matches:
            print(f"    - {match_id}: similarity {sim:.3f}")


async def demo_bidirectional():
    """Demo: bidirezionalità STM ↔ Instinctive"""
    print("\n" + "=" * 60)
    print("DEMO 4: Bidirectional Communication")
    print("=" * 60)

    system = create_system(embedding_dim=64, auto_distill=False)

    notifications = []

    def on_stm_change(state):
        notifications.append({
            "phase": state.phase.value,
            "ready_topics": state.ready_topics.copy(),
            "total_vectors": state.total_vectors
        })

    system.stm.subscribe(on_stm_change)

    async with system:
        print("\nProcesso input e monitoro notifiche STM...")

        for i in range(5):
            await system.process(f"Input {i}", topic="test")

        print(f"\nNotifiche ricevute: {len(notifications)}")
        for i, n in enumerate(notifications[-3:]):
            print(f"  [{i}] Phase: {n['phase']}, Vectors: {n['total_vectors']}, Ready: {n['ready_topics']}")


async def demo_evolution():
    """Demo: evoluzione del sistema"""
    print("\n" + "=" * 60)
    print("DEMO 5: System Evolution")
    print("=" * 60)

    system = create_system(embedding_dim=64, auto_distill=False)

    async with system:
        # Simula evoluzione: stessi pattern rafforzano LTM
        print("\nSimulo 20 interazioni sullo stesso topic...")

        latencies = []
        for i in range(20):
            result = await system.process(
                f"Query about authentication {i}",
                topic="auth"
            )
            latencies.append(result.latency_ms)

        # Stats
        stats = system.get_stats()
        print(f"\nStatistiche finali:")
        print(f"  Input processati: {stats.total_inputs}")
        print(f"  Fingerprint creati: {stats.total_fingerprints}")
        print(f"  Latenza media: {stats.avg_latency_ms:.2f}ms")

        # Trend latenza
        early = sum(latencies[:5]) / 5
        late = sum(latencies[-5:]) / 5
        print(f"\n  Latenza prime 5: {early:.2f}ms")
        print(f"  Latenza ultime 5: {late:.2f}ms")
        print(f"  Variazione: {((late - early) / early * 100):+.1f}%")


async def demo_full_stats():
    """Demo: statistiche complete sistema"""
    print("\n" + "=" * 60)
    print("DEMO 6: Full System Stats")
    print("=" * 60)

    system = create_system(embedding_dim=64, auto_distill=False)

    async with system:
        # Genera attività
        for topic in ["auth", "billing", "support"]:
            for i in range(5):
                await system.process(f"Query {i} for {topic}", topic=topic)

        # Stats complete
        state = system.get_state()

        print("\n📊 System State:")
        print(f"  Phase: {state['phase']}")

        print("\n📦 STM:")
        for k, v in state['stm'].items():
            if k != 'topics':
                print(f"  {k}: {v}")

        print("\n🗄️ LTM:")
        for k, v in state['ltm'].items():
            print(f"  {k}: {v}")

        print("\n🔀 Dispatcher:")
        for k, v in state['dispatcher'].items():
            print(f"  {k}: {v}")


async def main():
    """Run all demos"""
    print("\n" + "🧠" * 30)
    print("HIERARCHICAL MEMORY MVP - DEMO")
    print("🧠" * 30)

    await demo_basic()
    await demo_accumulation()
    await demo_ltm_matching()
    await demo_bidirectional()
    await demo_evolution()
    await demo_full_stats()

    print("\n" + "=" * 60)
    print("✅ All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
