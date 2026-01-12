# Hierarchical Cognitive Stack

> **⚠️ WIP - Work In Progress**

**4-Layer Memory Architecture for LLMs** — A hierarchical memory system inspired by biological, fractal, and instinctive principles.

## Architecture

```
                              INPUT
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                                                                                │
│   ① INSTINCTIVE LAYER (First Contact)                                         │
│   • Receives input, creates FINGERPRINT (Shazam-like)                         │
│   • Asks routing from Dispatcher NN                                           │
│   • Direct retrieval from LTM                                                 │
│   • Feeds results to STM (bidirectional)                                      │
│                                                                                │
│         │                    │                       │                         │
│         │ "routing?"         │ direct               │ feed                    │
│         ▼                    │ retrieval             ▼                         │
│   ┌───────────────┐          │              ┌─────────────────┐               │
│   │  DISPATCHER   │          │              │  SHORT-TERM     │               │
│   │  (LLM + NN)   │◄─────────┼──────────────│  MEMORY (STM)   │               │
│   │  "MTM Fabric" │          │              │                 │               │
│   │               │          │              │ Raw Vector DB   │               │
│   │ • NN: Router  │──────────┘              │ + topic labels  │               │
│   │   (map)       │                         │                 │               │
│   │               │                         │ Bidirectional:  │               │
│   │ • LLM:        │◄────────────────────────│ "Do I have      │               │
│   │   Distiller   │                         │  enough signal?"│               │
│   │   Evaluator   │                         └─────────────────┘               │
│   │               │                                                           │
│   │ • NN:         │                                                           │
│   │   Embedder    │                                                           │
│   └───────────────┘                                                           │
│           │                                                                    │
│           │ inject                                                             │
│           ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────┐             │
│   │  LONG-TERM MEMORY (LTM)                                     │             │
│   │  Graph-Vector DB (SurrealDB + HNSW)                         │             │
│   │  • Consolidated fingerprints                                │             │
│   │  • Deep patterns (DNA-like)                                 │             │
│   │  • Encoded knowledge                                        │             │
│   └─────────────────────────────────────────────────────────────┘             │
│                                                                                │
│   ⑧ Every cycle = EVOLUTION                                                   │
│      Fingerprints strengthen, paths become faster                             │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                             OUTPUT
```

## Core Concepts

> **Deep dive**: See [docs/THEORY.md](docs/THEORY.md) for theoretical foundations (EN/IT)

### Shazam Analogy
Like Shazam creates audio fingerprints for instant matching, this system:
- Creates perceptual fingerprints of inputs
- Matches them against consolidated patterns in LTM
- If strong match → instinctive response

### Bidirectionality (from Phonon-UI)
- **Instinctive ↔ STM**: "Do you have enough signal for fingerprinting?"
- STM notifies when topic is "ready" (like Shazam's audio buffer)
- If noise → more time, more samples needed

### Evolution
- Each cycle strengthens fingerprints
- Preferred paths emerge (like lightning following least resistance)
- System "learns" and becomes faster

### Medium-Term Fabric
The Dispatcher is not just a layer but the **connective fabric** that ties everything:
- **NN Router**: Maps fingerprints to LTM locations
- **LLM Distiller**: Compresses knowledge from STM
- **NN Embedder**: Injects distilled knowledge into LTM

## Installation

```bash
# Clone
git clone https://github.com/Alemusica/hierarchical-cognitive-stack
cd hierarchical-cognitive-stack

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
    system = create_system(embedding_dim=128)

    async with system:
        result = await system.process(
            "How do I reset my password?",
            topic="auth"
        )

        print(f"Fingerprint created: {result.fingerprint_created}")
        print(f"Response: {result.response}")
        print(f"Latency: {result.latency_ms:.2f}ms")

asyncio.run(main())
```

## Components

### STM (Short-Term Memory)
```python
from src import STM

stm = STM(min_samples_per_topic=5, quality_threshold=0.7)

# Buffer input
stm.buffer(vector, topic="auth")

# Check state (bidirectional)
state = stm.get_state()
if stm.is_topic_ready("auth"):
    data = stm.consume_topic("auth")
```

### Instinctive Layer
```python
from src import InstinctiveLayer

instinctive = InstinctiveLayer(stm=stm, dispatcher=dispatcher, ltm=ltm)
result = await instinctive.process_input("query", topic="auth")
```

### Dispatcher (Medium-Term Fabric)
```python
from src import Dispatcher

dispatcher = Dispatcher(embedding_dim=128)

# NN: routing
routing = await dispatcher.get_routing(fingerprint)

# LLM: evaluation
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

## Run Demo

```bash
python examples/demo.py
```

## Testing

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Hypotheses to Validate

| ID | Hypothesis | Metric | Target |
|----|------------|--------|--------|
| H1 | Fingerprinting faster than full-context | Latency ratio | < 0.1 |
| H2 | Repeated patterns reduce latency | Evolution rate | > 30% |
| H3 | Routing improves precision | Precision | > 0.8 |
| H4 | Distillation preserves semantics | Cosine similarity | > 0.7 |

## Project Structure

```
hierarchical-cognitive-stack/
├── src/
│   ├── __init__.py
│   ├── stm.py          # Short-Term Memory
│   ├── ltm.py          # Long-Term Memory
│   ├── dispatcher.py   # Dispatcher (LLM + NN) - MTM Fabric
│   ├── instinctive.py  # Instinctive Layer
│   └── orchestrator.py # System Orchestrator
├── tests/
│   └── test_integration.py
├── examples/
│   └── demo.py
├── pyproject.toml
└── README.md
```

## Related Projects

- **[Phonon-UI](https://github.com/alemusica/phonon-ui)**: Bidirectional patterns (cooperative refinement)
- **[Nico](https://github.com/alemusica/nico)**: Shazam-like fingerprinting (surge_shazam/MiniRocket)
- **[Rememberance](https://github.com/alemusica/Rememberance)**: Hierarchical memory concepts

## Author

Alessio ([@FluturArt](https://x.com/FluturArt))

## License

MIT
