from .session import Result as Result
from .session import Session as Session
from .report import AgentError as AgentError
from .report import ConfigError as ConfigError


def run(question: str, *, base_dir: str = ".", **kwargs) -> str:
    """One-call convenience. Returns the answer string or raises AgentError."""
    session = Session(base_dir=base_dir, **kwargs)
    result = session.run(question)
    if result.answer is None:
        raise AgentError(
            "Agent exhausted max turns without producing an answer"
            if result.exhausted
            else "Agent returned no answer"
        )
    return result.answer
