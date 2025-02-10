import random
from collections.abc import Iterable
from typing import Any, Literal

from ...core import OptimizerModule, _Chainable


class Alternate(OptimizerModule):
    """Alternates stepping with multiple modules.

    Args:
        modules (Iterable[OptimizerModule  |  Iterable[OptimizerModule]]): modules to alternate between.
        mode (int | list[int] | tuple[int] | "random"], optional):
            can be integer - number of repeats for all modules;
            list or tuple of integers per each module with number of repeats;
            "random" to pick module randomly each time. Defaults to 1.
        seed (int | None, optional): seed for "random" mode. Defaults to None.
    """
    def __init__(
        self,
        modules: Iterable[_Chainable],
        mode: int | list[int] | tuple[int] | Literal["random"] = 1,
        seed: int | None = None
    ):
        super().__init__({})
        modules = list(modules)

        for i,m in enumerate(modules):
            self._set_child_(i, m)

        self.random = random.Random(seed)

        if isinstance(mode, int): mode = [mode for _ in modules]
        self.mode: list[int] | tuple[int] | Literal['random'] = mode

        self.cur = 0
        if self.mode == 'random': self.remaining = 0
        else:
            self.remaining = self.mode[0]
            if len(self.mode) != len(self.children):
                raise ValueError(f"got {len(self.children)} modules but {len(mode)} repeats, they should be the same")

    def step(self, vars):
        if self.mode == 'random':
            module = self.random.choice(list(self.children.values()))

        else:
            if self.remaining == 0:
                self.cur += 1

            if self.cur >= len(self.mode):
                self.cur = 0

            if self.remaining == 0: self.remaining = self.mode[self.cur]

            module = self.children[self.cur]

            self.remaining -= 1

        if self.next_module is None:
            return module.step(vars)

        vars.ascent = module.return_ascent(vars)
        return self._update_params_or_step_with_next(vars)

