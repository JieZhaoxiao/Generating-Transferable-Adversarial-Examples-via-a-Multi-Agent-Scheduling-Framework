"""Public entry point for the released method.

The attack uses a role-specialized multi-agent scheduling framework:

1. Transferability Agent: schedules transformation operators and neighbor samples.
2. Iterative Agent: schedules perturbation budget, step scale, decay strength, and direction mixing.
3. Quality Agent: schedules texture-aware perceptual budget preservation.
4. Coordinator Agent: fuses the scheduled action applied at the next iteration.

The implementation records the fused action in the paper format
{n_op, n_nei, epsilon, alpha, gamma, lambda, eta}.
"""

from .agent import AgentAction, LLMAgentController
from .input_transformation.agent_quality_mdcsops import AgentQualityMDCSOPS
from .quality import QualityPreserver


class AgenticScheduleAttack(AgentQualityMDCSOPS):
    """Agent-scheduled transferable adversarial attack used by the paper."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("attack", "AgenticScheduleAttack")
        super().__init__(*args, **kwargs)


__all__ = [
    "AgentAction",
    "AgenticScheduleAttack",
    "LLMAgentController",
    "QualityPreserver",
]
