"""Independent row-sparse Adam for geometry-only supervision passes."""

import math
from collections.abc import Iterable

import torch

GEOMETRY_GROUP_NAMES = frozenset(("positions", "density", "rotation", "scale"))


class SparseGeometryAdam(torch.optim.Optimizer):
    """Update only geometry rows reached by the current sparse loss."""

    def __init__(
        self,
        params: Iterable[dict[str, object]],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
    ) -> None:
        """Create independent moments for the four geometry groups."""
        parameter_groups = list(params)
        group_names = tuple(str(group.get("name")) for group in parameter_groups)
        if (
            len(group_names) != len(GEOMETRY_GROUP_NAMES)
            or len(set(group_names)) != len(group_names)
            or frozenset(group_names) != GEOMETRY_GROUP_NAMES
        ):
            msg = (
                "Sparse geometry Adam requires exactly the groups "
                f"{sorted(GEOMETRY_GROUP_NAMES)} once each; got "
                f"{list(group_names)}."
            )
            raise ValueError(msg)
        super().__init__(
            parameter_groups,
            {"lr": lr, "betas": betas, "eps": eps},
        )
        self._validate_groups()

    @classmethod
    def from_primary_optimizer(
        cls,
        optimizer: torch.optim.Optimizer,
        *,
        learning_rate_scale: float,
    ) -> "SparseGeometryAdam":
        """Copy geometry learning rates into independent sparse state."""
        if not 0.0 < learning_rate_scale <= 1.0:
            msg = "Geometry learning-rate scale must be in (0, 1]."
            raise ValueError(msg)
        groups: list[dict[str, object]] = []
        for primary_group in optimizer.param_groups:
            name = str(primary_group.get("name"))
            if name not in GEOMETRY_GROUP_NAMES:
                continue
            groups.append(
                {
                    "name": name,
                    "params": list(primary_group["params"]),
                    "lr": float(primary_group["lr"]) * learning_rate_scale,
                    "betas": tuple(primary_group["betas"]),
                    "eps": float(primary_group["eps"]),
                }
            )
        return cls(groups)

    @staticmethod
    def _parameter(group: dict[str, object]) -> torch.nn.Parameter:
        parameters = group["params"]
        if not isinstance(parameters, list) or len(parameters) != 1:
            msg = "Sparse geometry Adam requires one tensor per group."
            raise ValueError(msg)
        parameter = parameters[0]
        if not isinstance(parameter, torch.nn.Parameter):
            msg = "Sparse geometry Adam group value must be a Parameter."
            raise TypeError(msg)
        return parameter

    def _validate_groups(self) -> None:
        point_count: int | None = None
        group_names: list[str] = []
        for group in self.param_groups:
            group_names.append(str(group.get("name")))
            parameter = self._parameter(group)
            if point_count is None:
                point_count = parameter.shape[0]
            elif parameter.shape[0] != point_count:
                msg = "Sparse geometry Adam groups have different row counts."
                raise ValueError(msg)
            if float(group.get("weight_decay", 0.0)) != 0.0:
                msg = "Sparse geometry Adam does not admit weight decay."
                raise ValueError(msg)
            self._validate_hyperparameters(group)
            self._validate_parameter_state(parameter)
        if (
            len(group_names) != len(GEOMETRY_GROUP_NAMES)
            or len(set(group_names)) != len(group_names)
            or frozenset(group_names) != GEOMETRY_GROUP_NAMES
        ):
            msg = "Sparse geometry Adam group identity changed: " f"{group_names}."
            raise ValueError(msg)

    @staticmethod
    def _validate_hyperparameters(group: dict[str, object]) -> None:
        name = str(group.get("name"))
        learning_rate = float(group["lr"])
        epsilon = float(group["eps"])
        beta_values = tuple(float(beta) for beta in group["betas"])
        if not math.isfinite(learning_rate) or learning_rate < 0.0:
            msg = f"Sparse geometry Adam group {name!r} has invalid lr."
            raise ValueError(msg)
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            msg = f"Sparse geometry Adam group {name!r} has invalid eps."
            raise ValueError(msg)
        if len(beta_values) != 2 or any(not math.isfinite(beta) or not 0.0 <= beta < 1.0 for beta in beta_values):
            msg = f"Sparse geometry Adam group {name!r} has invalid betas."
            raise ValueError(msg)

    def _validate_parameter_state(
        self,
        parameter: torch.nn.Parameter,
    ) -> None:
        state = self.state[parameter]
        if not state:
            return
        expected_keys = {"exp_avg", "exp_avg_sq", "row_steps"}
        if set(state) != expected_keys:
            msg = "Sparse geometry Adam checkpoint has incomplete row state."
            raise ValueError(msg)
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key]
            if value.shape != parameter.shape:
                msg = (
                    "Sparse geometry Adam checkpoint moment shape differs "
                    f"for {key}: {tuple(value.shape)} versus "
                    f"{tuple(parameter.shape)}."
                )
                raise ValueError(msg)
            if not torch.all(torch.isfinite(value)):
                msg = f"Sparse geometry Adam checkpoint has non-finite {key}."
                raise ValueError(msg)
        row_steps = state["row_steps"]
        if row_steps.shape != (parameter.shape[0],):
            msg = "Sparse geometry Adam checkpoint row-step shape differs."
            raise ValueError(msg)
        if not torch.all(torch.isfinite(row_steps)):
            msg = "Sparse geometry Adam checkpoint row steps are non-finite."
            raise ValueError(msg)
        if not torch.all(row_steps >= 0) or not torch.all(row_steps == torch.round(row_steps)):
            msg = "Sparse geometry Adam checkpoint row steps are invalid."
            raise ValueError(msg)

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore state only when group identity and row shapes agree."""
        saved_groups = state_dict.get("param_groups")
        if not isinstance(saved_groups, list):
            msg = "Sparse geometry Adam checkpoint has no parameter groups."
            raise ValueError(msg)
        saved_names = tuple(str(group.get("name")) for group in saved_groups)
        current_names = tuple(str(group.get("name")) for group in self.param_groups)
        if saved_names != current_names:
            msg = (
                "Sparse geometry Adam checkpoint group order differs: "
                f"saved={list(saved_names)}, current={list(current_names)}."
            )
            raise ValueError(msg)
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            parameter = self._parameter(group)
            state = self.state[parameter]
            if not state:
                continue
            state["exp_avg"] = state["exp_avg"].to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            state["exp_avg_sq"] = state["exp_avg_sq"].to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            state["row_steps"] = state["row_steps"].to(
                device=parameter.device,
                dtype=torch.int64,
            )
        self._validate_groups()

    def _initialize_state(self, parameter: torch.nn.Parameter) -> dict[str, torch.Tensor]:
        state = self.state[parameter]
        if state:
            return state
        state["exp_avg"] = torch.zeros_like(
            parameter,
            memory_format=torch.preserve_format,
        )
        state["exp_avg_sq"] = torch.zeros_like(
            parameter,
            memory_format=torch.preserve_format,
        )
        state["row_steps"] = torch.zeros(
            parameter.shape[0],
            dtype=torch.int64,
            device=parameter.device,
        )
        return state

    def _active_rows(self) -> torch.Tensor:
        active: torch.Tensor | None = None
        for group in self.param_groups:
            parameter = self._parameter(group)
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                msg = "Sparse geometry Adam requires dense parameter gradients."
                raise TypeError(msg)
            if not torch.all(torch.isfinite(gradient)):
                name = str(group["name"])
                msg = "Sparse geometry Adam received a non-finite gradient in " f"group {name!r}."
                raise FloatingPointError(msg)
            group_active = (gradient != 0.0).reshape(gradient.shape[0], -1).any(dim=1)
            active = group_active if active is None else active | group_active
        if active is not None:
            return active
        parameter = self._parameter(self.param_groups[0])
        return torch.zeros(
            parameter.shape[0],
            dtype=torch.bool,
            device=parameter.device,
        )

    @torch.no_grad()
    def step(self, closure=None) -> None:
        """Apply Adam only to rows with finite nonzero sparse gradients."""
        if closure is not None:
            msg = "Sparse geometry Adam does not support closures."
            raise ValueError(msg)
        self._validate_groups()
        active_indices = torch.nonzero(
            self._active_rows(),
            as_tuple=False,
        ).squeeze(1)
        if active_indices.numel() == 0:
            return
        for group in self.param_groups:
            parameter = self._parameter(group)
            gradient = parameter.grad
            if gradient is None:
                continue
            selected_gradient = gradient.index_select(0, active_indices)
            if not torch.all(torch.isfinite(selected_gradient)):
                msg = "Sparse geometry Adam received a non-finite gradient."
                raise FloatingPointError(msg)
            state = self._initialize_state(parameter)
            self._step_group(
                group=group,
                parameter=parameter,
                gradient=selected_gradient,
                active_indices=active_indices,
                state=state,
            )

    def _step_group(
        self,
        *,
        group: dict[str, object],
        parameter: torch.nn.Parameter,
        gradient: torch.Tensor,
        active_indices: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> None:
        beta1, beta2 = group["betas"]
        learning_rate = float(group["lr"])
        epsilon = float(group["eps"])
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        row_steps = state["row_steps"]
        selected_avg = exp_avg.index_select(0, active_indices)
        selected_avg_sq = exp_avg_sq.index_select(0, active_indices)
        selected_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        selected_avg_sq.mul_(beta2).addcmul_(
            gradient,
            gradient,
            value=1.0 - beta2,
        )
        selected_steps = row_steps.index_select(0, active_indices) + 1
        step_shape = (selected_steps.shape[0],) + (1,) * (parameter.ndim - 1)
        steps = selected_steps.to(dtype=torch.float32).reshape(step_shape)
        bias_correction1 = 1.0 - torch.pow(beta1, steps)
        bias_correction2 = 1.0 - torch.pow(beta2, steps)
        denominator = (selected_avg_sq.sqrt() / bias_correction2.sqrt()).add_(epsilon)
        update = selected_avg / bias_correction1 / denominator
        selected_parameter = parameter.index_select(0, active_indices)
        selected_parameter.add_(update, alpha=-learning_rate)
        parameter.index_copy_(0, active_indices, selected_parameter)
        exp_avg.index_copy_(0, active_indices, selected_avg)
        exp_avg_sq.index_copy_(0, active_indices, selected_avg_sq)
        row_steps.index_copy_(0, active_indices, selected_steps)
