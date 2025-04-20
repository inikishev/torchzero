from collections.abc import Callable, Iterable
import torch

class RFFTWrapper(torch.optim.Optimizer):
    """use any optimizer in fft space, for a more memory efficient version use torchzeros RFFTProject module."""
    def __init__(
        self,
        params,
        optimizer_fn: Callable[[list[torch.Tensor]], torch.optim.Optimizer],
        one_d: bool = False,
        norm=None,
    ):
        defaults = dict(optimizer_fn = optimizer_fn, one_d = one_d, norm = norm)
        super().__init__(params, defaults)

        self.optimizer: torch.optim.Optimizer | None = None

    def project(self, grads: list[torch.Tensor]) -> list[torch.Tensor]:
        one_d = self.param_groups[0]['one_d']
        norm = self.param_groups[0]['norm']

        if one_d:
            vec = torch.cat([g.view(-1) for g in grads])
            self._original_length = len(vec)
            vec = torch.view_as_real(torch.fft.rfft(vec, norm = norm)) # pylint:disable=not-callable
            return [vec]

        self._original_shapes = [g.shape for g in grads]
        projected = [torch.view_as_real(torch.fft.rfftn(g, norm = norm)) for g in grads] # pylint:disable=not-callable
        return projected


    def unproject(self, update: list[torch.Tensor]) -> list[torch.Tensor]:
        one_d = self.param_groups[0]['one_d']
        norm = self.param_groups[0]['norm']

        if one_d:
            vec = torch.view_as_complex(update[0])
            unprojected_vec = torch.fft.irfft(vec, n=self._original_length, norm=norm) # pylint:disable=not-callable
            torch.nn.utils.vector_to_parameters(unprojected_vec, update)
            return update

        return [torch.fft.irfftn(torch.view_as_complex(u), s=s, norm=norm) for u, s in zip(update, self._original_shapes)] # pylint:disable=not-callable

    @torch.no_grad
    def step(self, closure=None):
        params: list[torch.Tensor] = [p for g in self.param_groups for p in g['params'] if p.requires_grad]

        # -------------------------------- no closure -------------------------------- #
        if closure is None:

            # project grads
            projected = self.project([p.grad if p.grad is not None else torch.zeros_like(p) for p in params])
            projected = [p.contiguous() for p in projected]

            # initialize optimizer if None
            if self.optimizer is None:
                # fake project params by just setting to ones
                fake_params = [torch.ones_like(p, requires_grad=True, memory_format=torch.contiguous_format) for p in projected]
                self.optimizer = self.param_groups[0]['optimizer_fn'](fake_params)
                assert self.optimizer is not None

            else:
                fake_params = self.optimizer.param_groups[0]['params']

            # step
            initial_params = [p.clone() for p in fake_params]

            # set projected grads to fake optimizer params
            for fake_p, proj in zip(fake_params, projected):
                fake_p.grad = proj

            self.optimizer.step()
            projected_update = [p - initial_p for p, initial_p in zip(fake_params, initial_params)]

            # unproject
            update = self.unproject(projected_update)
            torch._foreach_add_(params,update)

            return None

        # ---------------------------------- closure --------------------------------- #
        # evaluate closure initially to initialize optimizer with fake projected params
        if self.optimizer is None:
            with torch.enable_grad(): loss = closure()
            projected = self.project([p.grad if p.grad is not None else torch.zeros_like(p) for p in params])
            projected = [p.contiguous() for p in projected]

            fake_params = [torch.ones_like(p, requires_grad=True, memory_format=torch.contiguous_format) for p in projected]
            self.optimizer = self.param_groups[0]['optimizer_fn'](fake_params)
            assert self.optimizer is not None

        else:
            fake_params = self.optimizer.param_groups[0]['params']

        initial_params = [p.clone() for p in fake_params]

        # modify closure
        def projected_closure(*args, **kwargs):
            with torch.enable_grad(): loss = closure(*args, **kwargs)

            # handle "backward" arg
            if (len(args) > 0 and args[0] is False) or kwargs.get('backward', False):
                return loss

            projected = self.project([p.grad if p.grad is not None else torch.zeros_like(p) for p in params])
            projected = [p.contiguous() for p in projected]

            # set projected grads to fake optimizer params
            for fake_p, proj in zip(fake_params, projected):
                fake_p.grad = proj

            return loss

        loss = self.optimizer.step(projected_closure)
        projected_update = [p - initial_p for p, initial_p in zip(fake_params, initial_params)]

        # unproject
        update = self.unproject(projected_update)
        torch._foreach_add_(params,update)

        return loss
