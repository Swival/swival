import os
from importlib import metadata as _metadata

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from .session import Result as Result
from .session import Session as Session
from .report import AgentError as AgentError
from .report import ConfigError as ConfigError
from .report import ContextOverflowError as ContextOverflowError
from .report import LifecycleError as LifecycleError

try:
    __version__ = _metadata.version("swival")
except _metadata.PackageNotFoundError:
    __version__ = "unknown"


def run(question: str, *, base_dir: str = ".", **kwargs) -> str:
    """One-call convenience. Returns the answer string or raises AgentError."""
    session = Session(base_dir=base_dir, **kwargs)
    result = session.run(question)
    answer = result.answer
    if answer is None:
        raise AgentError(
            "Agent exhausted max turns without producing an answer"
            if result.exhausted
            else "Agent returned no answer"
        )
    return answer
