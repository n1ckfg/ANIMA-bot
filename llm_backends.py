"""
LLM Backend Factory for ANIMA-bot

Provides a unified interface to create LLM instances from different backends.
Supported backends: ollama, llamacpp, openai
"""

import os
from typing import Any


def create_llm(config: dict) -> Any:
    """
    Create an LLM instance based on configuration.

    Args:
        config: Full configuration dict (expects config["llm"] structure)

    Returns:
        A LlamaIndex-compatible LLM instance

    Raises:
        ValueError: If backend is unknown
        ImportError: If required package for backend is not installed
    """
    llm_config = config.get("llm", {})
    backend = llm_config.get("backend", "ollama").lower()
    model = llm_config.get("model", "llama3.1:8b")
    timeout = llm_config.get("request_timeout", 360)

    if backend == "ollama":
        return _create_ollama(model, timeout, llm_config.get("ollama", {}))

    elif backend == "llamacpp":
        return _create_llamacpp(timeout, llm_config.get("llamacpp", {}))

    elif backend == "openai":
        return _create_openai(model, timeout, llm_config.get("openai", {}))

    else:
        raise ValueError(
            f"Unknown LLM backend: '{backend}'. "
            f"Supported backends: ollama, llamacpp, openai"
        )


def _create_ollama(model: str, timeout: float, settings: dict) -> Any:
    """Create Ollama LLM instance."""
    try:
        from llama_index.llms.ollama import Ollama
    except ImportError:
        raise ImportError(
            "Ollama backend requires llama-index-llms-ollama. "
            "Install with: pip install llama-index-llms-ollama"
        )

    base_url = settings.get("base_url", "http://localhost:11434")

    return Ollama(
        model=model,
        base_url=base_url,
        request_timeout=timeout
    )


def _create_llamacpp(timeout: float, settings: dict) -> Any:
    """Create llama.cpp LLM instance."""
    try:
        from llama_index.llms.llama_cpp import LlamaCPP
    except ImportError:
        raise ImportError(
            "llama.cpp backend requires llama-index-llms-llama-cpp. "
            "Install with: pip install llama-index-llms-llama-cpp"
        )

    model_path = settings.get("model_path")
    if not model_path:
        raise ValueError("llamacpp backend requires 'model_path' in config")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    n_ctx = settings.get("n_ctx", 4096)
    n_gpu_layers = settings.get("n_gpu_layers", -1)

    return LlamaCPP(
        model_path=model_path,
        context_window=n_ctx,
        n_gpu_layers=n_gpu_layers,
        generate_kwargs={
            "temperature": 0.7,
            "top_p": 0.9,
        },
    )


def _create_openai(model: str, timeout: float, settings: dict) -> Any:
    """Create OpenAI-compatible LLM instance."""
    try:
        from llama_index.llms.openai import OpenAI
    except ImportError:
        raise ImportError(
            "OpenAI backend requires llama-index-llms-openai. "
            "Install with: pip install llama-index-llms-openai"
        )

    api_key = settings.get("api_key") or os.environ.get("OPENAI_API_KEY")
    api_base = settings.get("api_base")

    kwargs = {
        "model": model,
        "timeout": timeout,
    }

    if api_key:
        kwargs["api_key"] = api_key

    if api_base:
        kwargs["api_base"] = api_base

    return OpenAI(**kwargs)


def get_backend_info(config: dict) -> str:
    """Return a human-readable string describing the configured backend."""
    llm_config = config.get("llm", {})
    backend = llm_config.get("backend", "ollama").lower()
    model = llm_config.get("model", "unknown")

    if backend == "ollama":
        base_url = llm_config.get("ollama", {}).get("base_url", "localhost:11434")
        return f"Ollama ({model}) at {base_url}"

    elif backend == "llamacpp":
        model_path = llm_config.get("llamacpp", {}).get("model_path", "unknown")
        return f"llama.cpp ({os.path.basename(model_path)})"

    elif backend == "openai":
        api_base = llm_config.get("openai", {}).get("api_base")
        if api_base:
            return f"OpenAI-compatible ({model}) at {api_base}"
        return f"OpenAI ({model})"

    return f"Unknown backend: {backend}"
