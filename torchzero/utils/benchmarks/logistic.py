from functools import partial
from typing import Any, cast

import numpy as np
import torch
import tqdm

def generate_correlated_logistic_data(
    n_samples=100_000,
    n_features=32,
    n_classes=10,
    n_correlated=768,
    correlation=0.99,
    seed=0
) -> tuple[np.ndarray, np.ndarray]:
    assert n_classes >= 2
    generator = np.random.default_rng(seed)

    X = generator.standard_normal(size=(n_samples, n_features))
    weights = generator.uniform(-2, 2, size=(n_features, n_classes))

    used_pairs = set()
    n_correlated = min(n_correlated, n_features * (n_features - 1) // 2)

    for _ in range(n_correlated):
        idxs = None
        while idxs is None or idxs in used_pairs:
            pair = generator.choice(n_features, size=2, replace=False)
            pair.sort()
            idxs = tuple(pair)

        used_pairs.add(idxs)
        idx1, idx2 = idxs

        noise = generator.standard_normal(n_samples) * np.sqrt(1 - correlation**2)
        X[:, idx2] = correlation * X[:, idx1] + noise

        w = generator.integers(1, 51)
        cls = generator.integers(0, n_classes)
        weights[idx1, cls] = w
        weights[idx2, cls] = -w

    logits = X @ weights

    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    y_one_hot = generator.multinomial(1, pvals=probabilities)
    y = np.argmax(y_one_hot, axis=1)

    X -= X.mean(0, keepdims=True)
    X /= X.std(0, keepdims=True)

    return X, y.astype(np.int64)


# if __name__ == '__main__':
#     X, y = generate_correlated_logistic_data()

#     plt.figure(figsize=(10, 8))
#     sns.heatmap(pl.DataFrame(X).corr(), annot=True, cmap='coolwarm', fmt=".2f")
#     plt.show()




def _tensorlist_equal(t1, t2):
    return all(a == b for a, b in zip(t1, t2))

_placeholder = cast(Any, ...)

def run_logistic_regression(X: torch.Tensor, y: torch.Tensor, opt_fn, max_steps: int, tol:float=0, l1:float=0, l2:float=0, pbar:bool=False, *, _assert_on_evaluated_same_params: bool = False):
    # ------------------------------- verify inputs ------------------------------ #
    n_samples, n_features = X.size()

    if y.ndim != 1: raise ValueError(f"y should be 1d, got {y.shape}")
    if y.size(0) != n_samples: raise ValueError(f"y should have {n_samples} elements, got {y.shape}")
    if y.device != X.device: raise ValueError(f"X and y should be on same device, got {X.device = }, {y.device = }")
    device = X.device
    dtype = X.dtype

    # ---------------------------- model and criterion --------------------------- #
    n_targets = int(y.amax()) + 1
    binary = n_targets == 2

    if binary:
        criterion = torch.nn.functional.binary_cross_entropy_with_logits
        model = torch.nn.Linear(n_features, 1).to(device=device, dtype=dtype)
        y = y.to(dtype=dtype)
    else:
        model = torch.nn.Linear(n_features, n_targets).to(device=device, dtype=dtype)
        criterion = torch.nn.functional.cross_entropy
        y = y.long()

    optimizer = opt_fn(list(model.parameters()))

    # ---------------------------------- closure --------------------------------- #
    def _l1_penalty():
        return sum(p.abs().sum() for p in model.parameters())
    def _l2_penalty():
        return sum(p.square().sum() for p in model.parameters())

    def closure(backward=True, evaluated_params: list = _placeholder, epoch: int = _placeholder):
        y_hat = model(X)
        loss = criterion(y_hat.squeeze(), y)

        if l1 > 0: loss += _l1_penalty() * l1
        if l2 > 0: loss += _l2_penalty() * l2

        if backward:
            optimizer.zero_grad()
            loss.backward()

        # here I also test to make sure the optimizer doesn't evaluate same parameters twice per step
        # this is for tests
        if _assert_on_evaluated_same_params:
            for p in evaluated_params:
                assert not _tensorlist_equal(p, model.parameters()), f"{optimizer} evaluated same parameters on epoch {epoch}"

            evaluated_params.append([p.clone() for p in model.parameters()])

        return loss

    # --------------------------------- optimize --------------------------------- #
    losses = []
    epochs = tqdm.trange(max_steps, disable=not pbar)
    for epoch in epochs:
        evaluated_params = []
        loss = float(optimizer.step(partial(closure, evaluated_params=evaluated_params, epoch=epoch)))

        losses.append(loss)
        epochs.set_postfix_str(f"{loss:.5f}")
        if loss <= tol:
            break

    return losses
