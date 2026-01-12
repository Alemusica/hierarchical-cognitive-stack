"""
Hierarchical Memory MVP
=======================

4-Layer Memory Architecture for LLMs:
1. Instinctive Layer - Fingerprinting, primo contatto
2. Short-Term Memory (STM) - Buffer multidimensionale
3. Dispatcher (Medium-Term) - LLM + NN hybrid
4. Long-Term Memory (LTM) - Graph-Vector DB

Flusso:
    INPUT → Instinctive ↔ STM (bidirezionale)
          → Dispatcher NN (routing)
          → LTM (retrieval)
          → Dispatcher LLM (valutazione)
          → OUTPUT

Ogni ciclo = Evoluzione (fingerprint si rafforzano)
"""

__version__ = "0.1.0"

from .stm import STM, STMState, TopicBuffer, STMPhase
from .ltm import LTM, LTMEntry, InMemoryLTMBackend, SurrealDBLTMBackend
from .dispatcher import (
    Dispatcher,
    DispatcherNN,
    DispatcherLLM,
    LocalDispatcherLLM,
    EvaluationResult,
    DistillationResult,
    RoutingMap
)
from .instinctive import (
    InstinctiveLayer,
    Fingerprint,
    RetrievalRequest,
    RetrievalResult
)
from .orchestrator import (
    Orchestrator,
    ProcessingResult,
    SystemStats,
    SystemPhase,
    create_system
)

__all__ = [
    # Core classes
    "STM",
    "LTM",
    "Dispatcher",
    "InstinctiveLayer",
    "Orchestrator",

    # Data classes
    "STMState",
    "TopicBuffer",
    "STMPhase",
    "LTMEntry",
    "Fingerprint",
    "RetrievalRequest",
    "RetrievalResult",
    "EvaluationResult",
    "DistillationResult",
    "RoutingMap",
    "ProcessingResult",
    "SystemStats",
    "SystemPhase",

    # Backends
    "InMemoryLTMBackend",
    "SurrealDBLTMBackend",

    # Components
    "DispatcherNN",
    "DispatcherLLM",
    "LocalDispatcherLLM",

    # Factory
    "create_system",
]
