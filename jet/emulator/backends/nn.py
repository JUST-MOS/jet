"""
Neural-network backend: PyTorch for training, NumPy for prediction.

The architecture is a plain multilayer perceptron, following the reference
implementation the project grew out of. What is *not* carried over is the
placement of preprocessing: that reference normalised parameters, standardised
targets and ran a PCA inside the model class, which couples the network to a
particular data layout. Here the network sees already-transformed arrays and
does nothing but regression.

Prediction is reimplemented in NumPy so that a saved bundle can be served
without PyTorch. That matters because the alternative -- requiring a multi-
gigabyte dependency to evaluate a model that has already been trained -- is
exactly the cost the Gaussian-process path exists to avoid, and there is no
reason for the two backends to differ on this point.

This backend reports no predictive uncertainty. ``predict`` returns ``None``
where a standard deviation would go, and :meth:`jet.emulator.Emulator.predict`
refuses a ``return_std`` request rather than handing back an array of zeros --
zeros downstream read as perfect certainty, which is the failure mode the
reference implementation had. Quantifying a network's uncertainty is a problem
in its own right, and is left for later rather than approximated badly now.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np

from .base import Backend, register_backend

__all__ = ["NNBackend"]

#: Supported hidden-layer activations. Names are stored in the bundle, so they
#: must stay stable.
_ACTIVATIONS = ("silu", "relu", "tanh")


def _silu(x: np.ndarray) -> np.ndarray:
    """Numerically stable ``x * sigmoid(x)``.

    The naive ``x / (1 + exp(-x))`` overflows for large negative ``x``; the
    two-branch form keeps every exponential's argument non-positive.
    """
    out = np.empty_like(x)
    positive = x >= 0.0
    out[positive] = x[positive] / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = x[~positive] * exp_x / (1.0 + exp_x)
    return out


def _activate(x: np.ndarray, name: str) -> np.ndarray:
    """Apply a named activation to a NumPy array."""
    if name == "silu":
        return _silu(x)
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "tanh":
        return np.tanh(x)
    raise ValueError(f"unknown activation {name!r}; supported: {list(_ACTIVATIONS)}")


@register_backend
class NNBackend(Backend):
    """An ensemble of multilayer perceptrons.

    Parameters
    ----------
    hidden_dims : sequence of int, optional
        Width of each hidden layer; the output layer is appended automatically.
    activation : str, optional
        One of ``"silu"``, ``"relu"``, ``"tanh"``.
    n_epochs : int, optional
        Maximum number of passes over the training set per ensemble member.
    batch_size : int, optional
        Mini-batch size.
    lr, weight_decay : float, optional
        AdamW hyperparameters.
    ensemble_size : int, optional
        Number of independently seeded networks, averaged at prediction time.
        Since no uncertainty is reported, the ensemble is purely an averaging
        device: independent initialisations make different errors, and their
        mean is usually more accurate than any single member.
    early_stopping_patience : int, optional
        Epochs without validation improvement before a member stops training.
        Ignored when ``validation_fraction`` is zero.
    validation_fraction : float, optional
        Fraction of samples held out for early stopping, taken from the end of
        a seeded permutation. Zero disables validation, in which case the
        learning-rate schedule tracks the training loss and no early stopping
        occurs.
    lr_scheduler_patience : int, optional
        Patience of the ``ReduceLROnPlateau`` schedule.
    device : str, optional
        Torch device string, e.g. ``"cpu"`` or ``"cuda"``. ``None`` picks CUDA
        when available.
    seed : int, optional
        Base seed; member ``i`` is seeded with ``seed + i``.
    use_amp : bool, optional
        Enable automatic mixed precision. Only takes effect on CUDA.
    use_compile : bool, optional
        Wrap the network in ``torch.compile``. Off by default: the compilation
        overhead is not repaid for the small training sets used here.
    verbose : bool, optional
        Print per-member progress to stdout.
    """

    name: ClassVar[str] = "nn"

    def __init__(
        self,
        hidden_dims: Sequence[int] = (512, 512, 256, 128),
        activation: str = "silu",
        n_epochs: int = 200,
        batch_size: int = 512,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        ensemble_size: int = 1,
        early_stopping_patience: int = 10,
        validation_fraction: float = 0.0,
        lr_scheduler_patience: int = 10,
        device: str | None = None,
        seed: int = 0,
        use_amp: bool = False,
        use_compile: bool = False,
        verbose: bool = False,
    ) -> None:
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; supported: {list(_ACTIVATIONS)}")
        if ensemble_size < 1:
            raise ValueError(f"ensemble_size must be at least 1, got {ensemble_size}")
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError(f"validation_fraction must lie in [0, 1), got {validation_fraction}")

        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer width")
        self.activation = activation
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.ensemble_size = int(ensemble_size)
        self.early_stopping_patience = int(early_stopping_patience)
        self.validation_fraction = float(validation_fraction)
        self.lr_scheduler_patience = int(lr_scheduler_patience)
        self.device = device
        self.seed = int(seed)
        self.use_amp = bool(use_amp)
        self.use_compile = bool(use_compile)
        self.verbose = bool(verbose)

        # Fitted state: one list of (weight, bias) pairs per ensemble member,
        # each ordered from the input layer outwards.
        self._weights: list[list[tuple[np.ndarray, np.ndarray]]] = []
        self._history: list[dict[str, list[float]]] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def n_members(self) -> int:
        """Number of fitted ensemble members."""
        return len(self._weights)

    @property
    def n_linears(self) -> int:
        """Number of affine layers per member, i.e. len(hidden_dims) + 1."""
        return len(self.hidden_dims) + 1

    @property
    def history(self) -> list[dict[str, list[float]]]:
        """Per-member training curves from the last :meth:`fit`."""
        return [dict(h) for h in self._history]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray) -> NNBackend:
        """Train the ensemble.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Transformed parameter points.
        Y : ndarray of shape (n_samples, n_targets)
            Transformed data vectors.

        Returns
        -------
        NNBackend
            ``self``, fitted.

        Raises
        ------
        ImportError
            If PyTorch is not installed; the message names the extra to install.
        """
        torch = _require_torch()

        X = self._as_2d(X, "X")
        Y = self._as_2d(Y, "Y")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X and Y must have the same number of samples, got {X.shape[0]} and {Y.shape[0]}"
            )
        if X.shape[0] < 2:
            raise ValueError("NNBackend needs at least two training samples")

        device = _resolve_device(torch, self.device)
        X_train, Y_train, X_val, Y_val = self._split_validation(X, Y)

        self._weights = []
        self._history = []
        for member in range(self.ensemble_size):
            if self.verbose:
                print(f"  NN member {member + 1}/{self.ensemble_size} on {device}")
            weights, history = self._train_member(
                torch, device, X_train, Y_train, X_val, Y_val, seed=self.seed + member
            )
            self._weights.append(weights)
            self._history.append(history)
        return self

    def _split_validation(
        self, X: np.ndarray, Y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Split off a validation set, or return ``None`` for it when disabled."""
        if self.validation_fraction <= 0.0:
            return X, Y, None, None

        n_val = int(round(self.validation_fraction * X.shape[0]))
        if n_val < 1 or n_val >= X.shape[0]:
            return X, Y, None, None

        rng = np.random.default_rng(self.seed)
        order = rng.permutation(X.shape[0])
        val_idx, train_idx = order[:n_val], order[n_val:]
        return X[train_idx], Y[train_idx], X[val_idx], Y[val_idx]

    def _train_member(
        self,
        torch: Any,
        device: Any,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray | None,
        Y_val: np.ndarray | None,
        seed: int,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, list[float]]]:
        """Train one ensemble member and return its weights and training curves."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        model = self._build_model(torch, X_train.shape[1], Y_train.shape[1])
        model = model.to(device)
        if self.use_compile and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as exc:  # pragma: no cover - backend-dependent
                if self.verbose:
                    print(f"  torch.compile unavailable, continuing eagerly: {exc}")

        criterion = torch.nn.MSELoss()
        optimiser = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=self.lr_scheduler_patience
        )

        amp_enabled = self.use_amp and device.type == "cuda"
        amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

        X_tensor = torch.as_tensor(X_train, dtype=torch.float32, device=device)
        Y_tensor = torch.as_tensor(Y_train, dtype=torch.float32, device=device)
        has_validation = X_val is not None and Y_val is not None
        if has_validation:
            X_val_tensor = torch.as_tensor(X_val, dtype=torch.float32, device=device)
            Y_val_tensor = torch.as_tensor(Y_val, dtype=torch.float32, device=device)

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        patience_left = self.early_stopping_patience
        n_samples = X_tensor.shape[0]
        generator = torch.Generator(device="cpu").manual_seed(seed)

        for epoch in range(self.n_epochs):
            model.train()
            permutation = torch.randperm(n_samples, generator=generator)
            running = 0.0
            for start in range(0, n_samples, self.batch_size):
                batch = permutation[start : start + self.batch_size]
                xb, yb = X_tensor[batch], Y_tensor[batch]

                optimiser.zero_grad(set_to_none=True)
                if amp_enabled:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        loss = criterion(model(xb).float(), yb)
                else:
                    loss = criterion(model(xb), yb)
                loss.backward()
                optimiser.step()
                running += loss.item() * xb.shape[0]

            train_loss = running / n_samples

            if has_validation:
                model.eval()
                with torch.no_grad():
                    if amp_enabled:
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            val_pred = model(X_val_tensor).float()
                    else:
                        val_pred = model(X_val_tensor)
                    val_loss = criterion(val_pred, Y_val_tensor).item()
                scheduler.step(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_left = self.early_stopping_patience
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                else:
                    patience_left -= 1
            else:
                val_loss = float("nan")
                scheduler.step(train_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["lr"].append(optimiser.param_groups[0]["lr"])

            if self.verbose and (epoch + 1) % 20 == 0:
                print(
                    f"    epoch {epoch + 1:4d}/{self.n_epochs}  "
                    f"train={train_loss:.3e}  val={val_loss:.3e}  "
                    f"lr={optimiser.param_groups[0]['lr']:.2e}"
                )

            if has_validation and patience_left <= 0:
                if self.verbose:
                    print(f"    early stop at epoch {epoch + 1}, best val loss {best_loss:.3e}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        return self._extract_weights(torch, model), history

    def _build_model(self, torch: Any, n_features: int, n_targets: int) -> Any:
        """Build the MLP as a ``nn.Sequential`` of alternating affine/activation layers."""
        layers: list[Any] = []
        previous = n_features
        for width in self.hidden_dims:
            layers.append(torch.nn.Linear(previous, width))
            layers.append(_torch_activation(torch, self.activation))
            previous = width
        layers.append(torch.nn.Linear(previous, n_targets))
        return torch.nn.Sequential(*layers)

    def _extract_weights(self, torch: Any, model: Any) -> list[tuple[np.ndarray, np.ndarray]]:
        """Pull the affine layers out of a fitted module as NumPy arrays.

        The model is a bare ``nn.Sequential`` alternating affine and activation
        layers, so the k-th affine layer lives at index ``2 * k``.
        """
        state = model.state_dict()
        weights: list[tuple[np.ndarray, np.ndarray]] = []
        for index in range(self.n_linears):
            weight_key = f"{2 * index}.weight"
            bias_key = f"{2 * index}.bias"
            # `torch.compile` wraps the module and prefixes every key with
            # `_orig_mod.`; strip it so compiled and eager models produce
            # identical bundles.
            if weight_key not in state:
                weight_key = f"_orig_mod.{weight_key}"
                bias_key = f"_orig_mod.{bias_key}"
            if weight_key not in state:
                raise KeyError(
                    f"expected {weight_key!r} in the trained model state; found {sorted(state)}"
                )
            weights.append(
                (
                    state[weight_key].detach().cpu().numpy().astype(float),
                    state[bias_key].detach().cpu().numpy().astype(float),
                )
            )
        return weights

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict with the fitted ensemble, using NumPy only.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Transformed parameter points.

        Returns
        -------
        mean : ndarray of shape (n_samples, n_targets)
            Mean over the ensemble members.
        std : None
            This backend does not report a predictive uncertainty. The slot is
            part of the backend protocol; ``None`` tells
            :meth:`jet.emulator.Emulator.predict` to refuse a ``return_std``
            request rather than fabricate zeros.
        """
        X = self._as_2d(X, "X")
        self._require_fitted()
        expected = self._weights[0][0][0].shape[1]
        if X.shape[1] != expected:
            raise ValueError(f"expected {expected} input features, got {X.shape[1]}")

        predictions = np.stack([self._forward(X, member) for member in self._weights])
        return predictions.mean(axis=0), None

    def _forward(self, X: np.ndarray, member: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Forward pass through one member, in NumPy."""
        a = X
        for index, (weight, bias) in enumerate(member):
            a = a @ weight.T + bias
            if index < len(member) - 1:
                a = _activate(a, self.activation)
        return a

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def state(self) -> dict[str, np.ndarray]:
        """Return the fitted weights as plain arrays."""
        self._require_fitted()
        state: dict[str, np.ndarray] = {
            "n_members": np.asarray(self.n_members),
            "n_linears": np.asarray(self.n_linears),
            "activation": np.asarray(self.activation),
        }
        for member_index, member in enumerate(self._weights):
            for layer_index, (weight, bias) in enumerate(member):
                state[f"m{member_index}.layer{layer_index}.weight"] = weight
                state[f"m{member_index}.layer{layer_index}.bias"] = bias
        return state

    @classmethod
    def from_state(cls, state: Mapping[str, np.ndarray]) -> NNBackend:
        """Rebuild a fitted backend from :meth:`state` output, using NumPy only."""
        if "n_members" not in state or "activation" not in state:
            raise KeyError("NN state must contain 'n_members' and 'activation'")

        n_members = int(state["n_members"])
        n_linears = int(state["n_linears"])
        activation = str(state["activation"])

        obj = cls(activation=activation, ensemble_size=n_members)
        weights: list[list[tuple[np.ndarray, np.ndarray]]] = []
        for member_index in range(n_members):
            member: list[tuple[np.ndarray, np.ndarray]] = []
            for layer_index in range(n_linears):
                weight_key = f"m{member_index}.layer{layer_index}.weight"
                bias_key = f"m{member_index}.layer{layer_index}.bias"
                if weight_key not in state:
                    raise KeyError(f"NN state is missing {weight_key!r}")
                member.append(
                    (
                        np.asarray(state[weight_key], dtype=float),
                        np.asarray(state[bias_key], dtype=float),
                    )
                )
            weights.append(member)

        # Recover the architecture so that repr and further training agree with
        # the loaded model rather than with the constructor defaults.
        obj.hidden_dims = tuple(weights[0][i][0].shape[0] for i in range(len(weights[0]) - 1))
        obj._weights = weights
        obj._history = []
        return obj

    def _require_fitted(self) -> None:
        if not self._weights:
            raise RuntimeError("NNBackend has not been fitted")

    def __repr__(self) -> str:
        if not self._weights:
            return f"NNBackend(hidden_dims={list(self.hidden_dims)}, unfitted)"
        return f"NNBackend(hidden_dims={list(self.hidden_dims)}, n_members={self.n_members})"


def _torch_activation(torch: Any, name: str) -> Any:
    """Return the PyTorch module implementing a named activation."""
    if name == "silu":
        return torch.nn.SiLU()
    if name == "relu":
        return torch.nn.ReLU()
    if name == "tanh":
        return torch.nn.Tanh()
    raise ValueError(f"unknown activation {name!r}; supported: {list(_ACTIVATIONS)}")


def _resolve_device(torch: Any, requested: str | None) -> Any:
    """Resolve the requested device string, defaulting to CUDA when present."""
    if requested is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _require_torch() -> Any:
    """Import PyTorch, or raise an error naming the extra to install."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "training a neural-network emulator requires PyTorch; "
            "install it with `pip install jet[nn]`"
        ) from exc
    return torch
