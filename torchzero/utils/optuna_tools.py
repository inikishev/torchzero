import optuna

from ..core import Chain

from ..modules import (
    EMA,
    NAG,
    Cautious,
    ClipNorm,
    ClipNormGrowth,
    ClipValue,
    ClipValueGrowth,
    Debias1,
    Normalize,
)


def get_momentum(trial: optuna.Trial, prefix: str, conditional: bool=True):
    cond = trial.suggest_categorical(f'{prefix}_use_momentum', [True,False]) if conditional else True
    if cond:
        beta = trial.suggest_categorical(f'{prefix}_beta', [True, False])
        dampening = trial.suggest_categorical(f'{prefix}_dampening', [True, False])
        lerp = trial.suggest_categorical(f'{prefix}_use_lerp', [True, False])
        nag = trial.suggest_categorical(f'{prefix}_use_NAG', [True, False])
        debiased = trial.suggest_categorical(f'{prefix}_debiased', [True, False])
        if nag:
            m = NAG(beta, dampening, lerp)
            if debiased: m = Chain(m, Debias1(beta))
        else:
            m = EMA(beta, dampening, debiased=debiased, lerp=lerp)
        return [m]
    return []

def get_clip_value(trial: optuna.Trial, prefix: str, conditional: bool=True):
    cond = trial.suggest_categorical(f'{prefix}_use_clip_value', [True,False]) if conditional else True
    if cond:
        return [ClipValue(value = trial.suggest_float(f'{prefix}_clip_value', 0, 10))]
    return []


