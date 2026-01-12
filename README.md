# Hierarchical Memory MVP

**4-Layer Memory Architecture for LLMs** — Sistema di memoria gerarchica ispirato a principi biologici, frattali e istintivi.

## Architettura

```
                              INPUT
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                                                                                │
│   ① INSTINCTIVE LAYER (Primo contatto)                                        │
│   • Riceve input, crea FINGERPRINT (Shazam-like)                              │
│   • Chiede routing a Dispatcher NN                                            │
│   • Retrieval diretto da LTM                                                  │
│   • Feed risultati a STM (bidirezionale)                                      │
│                                                                                │
│         │                    │                       │                         │
│         │ "mappa?"           │ retrieval            │ feed                    │
│         ▼                    │ diretto               ▼                         │
│   ┌───────────────┐          │              ┌─────────────────┐               │
│   │  DISPATCHER   │          │              │  SHORT-TERM     │               │
│   │  (LLM + NN)   │◄─────────┼──────────────│  MEMORY (STM)   │               │
│   │               │          │              │                 │               │
│   │ • NN: Router  │          │              │ Vector DB raw   │               │
│   │   (mappa)     │──────────┘              │ + labeling      │               │
│   │               │                         │ per topic       │               │
│   │ • LLM:        │                         │                 │               │
│   │   Distiller   │◄────────────────────────│ Bidirezionale:  │               │
│   │   Valutatore  │                         │ "Ho abbastanza  │               │
│   │               │                         │  segnale?"      │               │
│   │ • NN:         │                         └─────────────────┘               │
│   │   Embedder    │                                                           │
│   └───────────────┘                                                           │
│           │                                                                    │
│           │ inject                                                             │
│           ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────┐             │
│   │  LONG-TERM MEMORY (LTM)                                     │             │
│   │  Graph-Vector DB (SurrealDB + HNSW)                         │             │
│   │  • Fingerprint consolidati                                  │             │
│   │  • Pattern profondi (DNA-like)                              │             │
│   │  • Knowledge encoded                                        │             │
│   └─────────────────────────────────────────────────────────────┘             │
│                                                                                │
│   ⑧ Ogni ciclo = EVOLUZIONE                                                   │
│      I fingerprint si rafforzano, i path diventano più veloci                 │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                             OUTPUT
```

## Concetti Chiave

### Analogia Shazam
Come Shazam crea fingerprint audio per match istantaneo, questo sistema:
- Crea fingerprint percettivi degli input
- Li matcha con pattern consolidati in LTM
- Se match forte → risposta istintiva

### Bidirezionalità (da Phonon-UI)
- **Instinctive ↔ STM**: "Hai abbastanza segnale per fingerprint?"
- STM notifica quando topic è "ready" (come buffer audio di Shazam)
- Se rumore → più tempo, più samples necessari

### Evoluzione
- Ogni ciclo rafforza i fingerprint
- Path preferenziali emergono (come fulmine che segue resistenza minima)
- Sistema "impara" e diventa più veloce

## Installazione

```bash
# Clone
git clone https://github.com/alemusica/hierarchical-memory-mvp
cd hierarchical-memory-mvp

# Install
pip install -e ".[dev]"

# Test
pytest tests/ -v
```

## Quick Start

```python
import asyncio
from src import create_system

async def main():
    # Crea sistema
    system = create_system(embedding_dim=128)

    async with system:
        # Processa input
        result = await system.process(
            "Come resetto la password?",
            topic="auth"
        )

        print(f"Fingerprint creato: {result.fingerprint_created}")
        print(f"Risposta: {result.response}")
        print(f"Latenza: {result.latency_ms:.2f}ms")

asyncio.run(main())
```

## Componenti

### STM (Short-Term Memory)
```python
from src import STM

stm = STM(min_samples_per_topic=5, quality_threshold=0.7)

# Buffer input
stm.buffer(vector, topic="auth")

# Check stato (bidirezionale)
state = stm.get_state()
if stm.is_topic_ready("auth"):
    # Pronto per fingerprint
    data = stm.consume_topic("auth")
```

### Instinctive Layer
```python
from src import InstinctiveLayer

instinctive = InstinctiveLayer(stm=stm, dispatcher=dispatcher, ltm=ltm)

# Processa input
result = await instinctive.process_input("query", topic="auth")
```

### Dispatcher (LLM + NN)
```python
from src import Dispatcher

dispatcher = Dispatcher(embedding_dim=128)

# NN: routing
routing = await dispatcher.get_routing(fingerprint)

# LLM: valutazione
evaluation = await dispatcher.evaluate(fingerprint, retrieval_result)
```

### LTM (Long-Term Memory)
```python
from src import LTM

# In-memory (testing)
ltm = LTM.in_memory(embedding_dim=128)

# SurrealDB (production)
ltm = LTM.with_surrealdb(url="ws://localhost:8000/rpc")

# Store
entry_id = await ltm.store(vector, metadata={"topic": "auth"})

# Search
matches = await ltm.search(query_vector, top_k=5)
```

## Test

```bash
# Tutti i test
pytest tests/ -v

# Solo integration
pytest tests/test_integration.py -v

# Con coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Ipotesi da Validare

| ID | Ipotesi | Metrica |
|----|---------|---------|
| H1 | Fingerprinting più veloce di full-context | Latency ratio < 0.1 |
| H2 | Pattern ripetuti riducono latenza | Evolution rate > 0.3 |
| H3 | Routing migliora precision | Precision > 0.8 |
| H4 | Distillazione preserva semantica | Cosine sim > 0.7 |

## Struttura

```
hierarchical-memory-mvp/
├── src/
│   ├── __init__.py
│   ├── stm.py          # Short-Term Memory
│   ├── ltm.py          # Long-Term Memory
│   ├── dispatcher.py   # Dispatcher (LLM + NN)
│   ├── instinctive.py  # Instinctive Layer
│   └── orchestrator.py # Orchestratore
├── tests/
│   └── test_integration.py
├── pyproject.toml
└── README.md
```

## Riferimenti

- **Phonon-UI**: Pattern bidirezionalità (cooperative refinement)
- **Nico/surge_shazam**: Fingerprinting Shazam-like (MiniRocket)
- **Rememberance**: Concetti memoria gerarchica

## Autore

Alessio (@FluturArt)

## License

MIT
