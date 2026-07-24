"""Per-source-frame Adam for trainable local projection fields."""

import torch

LOCAL_PROJECTION_FIELD_BETAS = (0.8, 0.95)
LOCAL_PROJECTION_FIELD_EPSILON = 2.0e-5
LOCAL_PROJECTION_FIELD_LR = 1.0e-3
LOCAL_PROJECTION_FIELD_SHRINKAGE = 0.1
LOCAL_PROJECTION_FIELD_WARMUP_UPDATES = 4


class LocalProjectionFieldAdam(torch.optim.Optimizer):
    """Update exactly one source-frame field with independent moments."""

    def __init__(
        self,
        parameter: torch.nn.Parameter,
        *,
        lr: float = LOCAL_PROJECTION_FIELD_LR,
        betas: tuple[float, float] = LOCAL_PROJECTION_FIELD_BETAS,
        eps: float = LOCAL_PROJECTION_FIELD_EPSILON,
        shrinkage: float = LOCAL_PROJECTION_FIELD_SHRINKAGE,
        warmup_updates: int = LOCAL_PROJECTION_FIELD_WARMUP_UPDATES,
    ) -> None:
        """Initialize per-frame moments for one field-table parameter."""
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "shrinkage": shrinkage,
            "warmup_updates": warmup_updates,
        }
        super().__init__([parameter], defaults)
        self._validate_parameters()
        self._validate_hyperparameters()

    @staticmethod
    def _parameter(group: dict[str, object]) -> torch.nn.Parameter:
        parameters = group["params"]
        if not isinstance(parameters, list) or len(parameters) != 1:
            raise ValueError(
                "Local projection field requires one parameter tensor."
            )
        parameter = parameters[0]
        if not isinstance(parameter, torch.nn.Parameter):
            raise TypeError("Local projection field requires a Parameter.")
        return parameter

    def _validate_parameters(self) -> None:
        parameter = self._parameter(self.param_groups[0])
        if parameter.ndim != 4 or parameter.shape[-1] != 2:
            raise ValueError(
                "Local projection field values must have shape (F, H, W, 2)."
            )
        if parameter.shape[0] <= 0 or parameter.shape[1] <= 1:
            raise ValueError("Local projection field must contain usable grids.")
        if parameter.shape[2] <= 1 or parameter.dtype is not torch.float32:
            raise ValueError(
                "Local projection field must use float32 grids larger than 1x1."
            )

    def _validate_hyperparameters(self) -> None:
        group = self.param_groups[0]
        lr = float(group["lr"])
        eps = float(group["eps"])
        shrinkage = float(group["shrinkage"])
        warmup_updates = int(group["warmup_updates"])
        betas = tuple(float(value) for value in group["betas"])
        if not torch.isfinite(torch.tensor((lr, eps, shrinkage))).all():
            raise ValueError("Local projection field hyperparameters must be finite.")
        if lr <= 0.0 or eps <= 0.0 or shrinkage < 0.0:
            raise ValueError("Local projection field hyperparameters are invalid.")
        if len(betas) != 2 or any(value < 0.0 or value >= 1.0 for value in betas):
            raise ValueError("Local projection field betas must be in [0, 1).")
        if warmup_updates <= 0:
            raise ValueError("Local projection field warmup_updates must be positive.")

    @staticmethod
    def _hyperparameter_signature(
        group: dict[str, object],
    ) -> tuple[float, tuple[float, float], float, float, int]:
        try:
            betas_value = group["betas"]
            if not isinstance(betas_value, (list, tuple)):
                raise TypeError("betas must be a sequence")
            betas = tuple(float(value) for value in betas_value)
            if len(betas) != 2:
                raise ValueError("betas must contain two values")
            return (
                float(group["lr"]),
                (betas[0], betas[1]),
                float(group["eps"]),
                float(group["shrinkage"]),
                int(group["warmup_updates"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Local projection field optimizer hyperparameters are invalid."
            ) from exc

    def _validate_checkpoint_hyperparameters(
        self,
        state_dict: dict[str, object],
    ) -> None:
        saved_groups = state_dict.get("param_groups")
        if not isinstance(saved_groups, list) or len(saved_groups) != 1:
            raise ValueError(
                "Local projection field optimizer checkpoint is invalid."
            )
        saved_group = saved_groups[0]
        if not isinstance(saved_group, dict):
            raise ValueError(
                "Local projection field optimizer checkpoint is invalid."
            )
        saved_signature = self._hyperparameter_signature(saved_group)
        current_signature = self._hyperparameter_signature(self.param_groups[0])
        if saved_signature != current_signature:
            raise ValueError(
                "Local projection field optimizer hyperparameters changed "
                "across checkpoint resume."
            )

    def _initialize_state(self) -> None:
        parameter = self._parameter(self.param_groups[0])
        state = self.state[parameter]
        if state:
            return
        state["source_frame_steps"] = torch.zeros(
            parameter.shape[0],
            dtype=torch.int32,
            device=parameter.device,
        )
        state["exp_avg"] = torch.zeros_like(parameter)
        state["exp_avg_sq"] = torch.zeros_like(parameter)

    def _restore_and_validate_state(self) -> None:
        self._validate_parameters()
        self._validate_hyperparameters()
        parameter = self._parameter(self.param_groups[0])
        state = self.state[parameter]
        expected = {
            "source_frame_steps": (parameter.shape[0],),
            "exp_avg": parameter.shape,
            "exp_avg_sq": parameter.shape,
        }
        for key, shape in expected.items():
            value = state.get(key)
            if not torch.is_tensor(value) or value.shape != shape:
                raise ValueError(
                    "Local projection field optimizer state is incompatible."
                )
            dtype = torch.int32 if key == "source_frame_steps" else torch.float32
            state[key] = value.to(device=parameter.device, dtype=dtype)

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore source-frame moments and independent update counters."""
        self._validate_checkpoint_hyperparameters(state_dict)
        super().load_state_dict(state_dict)
        self._restore_and_validate_state()

    @torch.no_grad()
    def step(self, source_frame_idx: int) -> None:
        """Apply the recovered update to one source-frame field row."""
        self._initialize_state()
        parameter = self._parameter(self.param_groups[0])
        if not 0 <= source_frame_idx < parameter.shape[0]:
            raise ValueError(
                f"Invalid local projection source frame: {source_frame_idx}."
            )
        group = self.param_groups[0]
        state = self.state[parameter]
        source_frame_steps = state["source_frame_steps"]
        update_step = int(source_frame_steps[source_frame_idx].item()) + 1
        gradient = (
            torch.zeros_like(parameter[source_frame_idx])
            if parameter.grad is None
            else parameter.grad[source_frame_idx]
        )
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("Local projection field gradient is non-finite.")
        first_moment = state["exp_avg"][source_frame_idx]
        second_moment = state["exp_avg_sq"][source_frame_idx]
        beta1, beta2 = (float(value) for value in group["betas"])
        first_moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        second_moment.mul_(beta2).addcmul_(
            gradient,
            gradient,
            value=1.0 - beta2,
        )
        normalized = first_moment / (1.0 - beta1**update_step)
        normalized = normalized / (
            torch.sqrt(second_moment / (1.0 - beta2**update_step))
            + float(group["eps"])
        )
        warmup = min(
            update_step / float(group["warmup_updates"]),
            1.0,
        )
        selected_parameter = parameter[source_frame_idx]
        selected_parameter.add_(
            normalized + float(group["shrinkage"]) * selected_parameter,
            alpha=-float(group["lr"]) * warmup,
        )
        source_frame_steps[source_frame_idx] = update_step
