from .termination import (
    TerminateAfterNEvaluations,
    TerminateAfterNSeconds,
    TerminateAfterNSteps,
    TerminateAll,
    TerminateAny,
    TerminateByGradientNorm,
    TerminateByUpdateNorm,
    TerminateOnLossReached,
    TerminateOnNoImprovement,
    TerminationCriteriaBase,
    TerminateNever,
    make_termination_criteria
)
