from .session import Result as Result
from .session import Session as Session
from .report import AgentError as AgentError
from .report import ConfigError as ConfigError
from .report import ContextOverflowError as ContextOverflowError
from .report import LifecycleError as LifecycleError

from . import agent as _agent
from ._chatgpt_compat import (
    patch_chatgpt_responses_empty_output as _patch_chatgpt_responses_empty_output,
)

_agent._patch_chatgpt_responses_empty_output = _patch_chatgpt_responses_empty_output

del _agent, _patch_chatgpt_responses_empty_output


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
