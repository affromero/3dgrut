"""Per-Gaussian Adam semantics recovered from the fused visibility-selective CUDA kernels."""

from collections.abc import Iterable

import torch

REQUIRED_GROUP_NAMES = frozenset(("positions", "density", "features_albedo", "rotation", "scale"))
OPTIONAL_GROUP_NAMES = frozenset(("features_specular",))
IMAGE_GROUP_NAMES = REQUIRED_GROUP_NAMES | OPTIONAL_GROUP_NAMES
GEOMETRY_GROUP_NAMES = frozenset(("positions", "density", "rotation", "scale"))
COUNTER_MODULUS = 65_536
POSITION_DECAY = 0.9998
POSITION_DECAY_FLOOR = 0.01
COLOR_DECAY = 0.95
COLOR_DECAY_FLOOR = 0.025
COLOR_EPSILON = 2.0e-5
OPACITY_REGULARIZER = 0.001
SCALE_REGULARIZER = 0.01
SCALE_RANGE_THRESHOLD = 8.0


def _is_color_group(group_name: str) -> bool:
    return group_name in {"features_albedo", "features_specular"}


def ordered_log_scale_regularizer(log_scales: torch.Tensor) -> torch.Tensor:
    """Return the recovered regularizer in normalized-Adam update space."""
    if log_scales.ndim != 2 or log_scales.shape[1] != 3:
        msg = "Log-scales must have shape (N, 3)."
        raise ValueError(msg)

    sorted_scales, sorted_indices = torch.sort(
        log_scales,
        dim=1,
        stable=True,
    )
    minimum = sorted_scales[:, 0]
    middle = sorted_scales[:, 1]
    maximum = sorted_scales[:, 2]
    maximum_middle_gap = maximum - middle
    maximum_minimum_gap = maximum - minimum
    all_equal = maximum_minimum_gap == 0.0

    minimum_regularizer = torch.where(
        maximum_minimum_gap > SCALE_RANGE_THRESHOLD,
        torch.full_like(minimum, -SCALE_REGULARIZER),
        torch.full_like(minimum, SCALE_REGULARIZER),
    )
    minimum_regularizer = torch.where(
        all_equal,
        torch.zeros_like(minimum_regularizer),
        minimum_regularizer,
    )
    sorted_regularizer = torch.stack(
        (
            minimum_regularizer,
            -SCALE_REGULARIZER * maximum_middle_gap,
            SCALE_REGULARIZER * maximum_middle_gap,
        ),
        dim=1,
    )
    regularizer = torch.zeros_like(log_scales)
    regularizer.scatter_(1, sorted_indices, sorted_regularizer)
    return regularizer


class VisibilityDecayedAdam(torch.optim.Optimizer):
    """Visibility-selective Adam with distinct image and geometry passes."""

    def __init__(
        self,
        params: Iterable[dict[str, object]],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1.0e-10,
    ) -> None:
        """Initialize the recovered per-Gaussian parameter groups."""
        parameter_groups = list(params)
        group_names = frozenset(str(group["name"]) for group in parameter_groups)
        if not REQUIRED_GROUP_NAMES.issubset(group_names) or not group_names.issubset(IMAGE_GROUP_NAMES):
            msg = (
                "Visibility-decayed Adam requires groups "
                f"{sorted(REQUIRED_GROUP_NAMES)} and supports "
                f"{sorted(OPTIONAL_GROUP_NAMES)}; got "
                f"{sorted(group_names)}."
            )
            raise ValueError(msg)
        for group in parameter_groups:
            if _is_color_group(str(group["name"])):
                group["eps"] = COLOR_EPSILON
        super().__init__(
            parameter_groups,
            {"lr": lr, "betas": betas, "eps": eps},
        )

    @staticmethod
    def _parameter(group: dict[str, object]) -> torch.nn.Parameter:
        parameters = group["params"]
        if not isinstance(parameters, list) or len(parameters) != 1:
            msg = "Visibility-decayed Adam requires one tensor per group."
            raise ValueError(msg)
        parameter = parameters[0]
        if not isinstance(parameter, torch.nn.Parameter):
            msg = "Optimizer group value must be a Parameter."
            raise TypeError(msg)
        return parameter

    def _initialize_group_state(
        self,
        *,
        group: dict[str, object],
        parameter: torch.nn.Parameter,
    ) -> None:
        state = self.state[parameter]
        if state:
            return
        group_name = str(group["name"])
        state_dtype = torch.float16 if _is_color_group(group_name) else parameter.dtype
        state["exp_avg"] = torch.zeros_like(
            parameter,
            dtype=state_dtype,
            memory_format=torch.preserve_format,
        )
        state["exp_avg_sq"] = torch.zeros_like(
            parameter,
            dtype=state_dtype,
            memory_format=torch.preserve_format,
        )
        if group_name == "positions":
            state["gaussian_steps"] = torch.zeros(
                parameter.shape[0],
                dtype=torch.int32,
                device=parameter.device,
            )

    def _restore_and_validate_state(self) -> None:
        point_count: int | None = None
        for group in self.param_groups:
            parameter = self._parameter(group)
            group_name = str(group["name"])
            if point_count is None:
                point_count = parameter.shape[0]
            elif parameter.shape[0] != point_count:
                msg = "Optimizer groups have inconsistent point counts."
                raise ValueError(msg)
            state = self.state[parameter]
            if not state and _is_color_group(group_name):
                continue
            if "exp_avg" not in state or "exp_avg_sq" not in state:
                msg = "Visibility-decayed Adam checkpoint is missing moments " f"for {group_name}."
                raise ValueError(msg)
            expected_dtype = torch.float16 if _is_color_group(group_name) else parameter.dtype
            state["exp_avg"] = state["exp_avg"].to(dtype=expected_dtype)
            state["exp_avg_sq"] = state["exp_avg_sq"].to(dtype=expected_dtype)
            if group_name != "positions":
                continue
            gaussian_steps = state.get("gaussian_steps")
            if not torch.is_tensor(gaussian_steps):
                msg = "Optimizer checkpoint is missing Gaussian steps."
                raise ValueError(msg)
            gaussian_steps = gaussian_steps.to(dtype=torch.int32)
            if gaussian_steps.shape != (parameter.shape[0],):
                msg = "Gaussian-step state has the wrong shape."
                raise ValueError(msg)
            state["gaussian_steps"] = gaussian_steps

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore a checkpoint without changing state storage dtypes."""
        super().load_state_dict(state_dict)
        self._restore_and_validate_state()

    @torch.no_grad()
    def step(self, visibility: torch.Tensor) -> None:
        """Apply the full image-visible optimizer, including color decay."""
        self._step_visible(
            visibility=visibility,
            group_names=IMAGE_GROUP_NAMES,
        )

    @torch.no_grad()
    def step_geometry(self, visibility: torch.Tensor) -> None:
        """Apply the recovered LiDAR-visible optimizer without color state."""
        self._step_visible(
            visibility=visibility,
            group_names=GEOMETRY_GROUP_NAMES,
        )

    def _step_visible(
        self,
        *,
        visibility: torch.Tensor,
        group_names: frozenset[str],
    ) -> None:
        position_group = next(group for group in self.param_groups if str(group["name"]) == "positions")
        position_parameter = self._parameter(position_group)
        point_count = position_parameter.shape[0]
        visible = visibility.reshape(-1).to(dtype=torch.bool)
        if visible.shape != (point_count,):
            msg = "Optimizer visibility must have one value per Gaussian."
            raise ValueError(msg)

        selected_groups = [group for group in self.param_groups if str(group["name"]) in group_names]
        for group in selected_groups:
            parameter = self._parameter(group)
            if parameter.shape[0] != point_count:
                msg = "Optimizer groups have inconsistent point counts."
                raise ValueError(msg)
            self._initialize_group_state(group=group, parameter=parameter)

        visible_indices = torch.nonzero(
            visible,
            as_tuple=False,
        ).squeeze(1)
        if visible_indices.numel() == 0:
            return

        position_state = self.state[position_parameter]
        gaussian_steps = position_state["gaussian_steps"]
        update_steps_int = gaussian_steps.index_select(0, visible_indices) + 1
        update_steps = update_steps_int.to(dtype=torch.float32)
        for group in selected_groups:
            self._step_group(
                group=group,
                visible_indices=visible_indices,
                update_steps=update_steps,
            )

        gaussian_steps.index_copy_(
            0,
            visible_indices,
            torch.remainder(update_steps_int, COUNTER_MODULUS),
        )

    def _step_group(
        self,
        *,
        group: dict[str, object],
        visible_indices: torch.Tensor,
        update_steps: torch.Tensor,
    ) -> None:
        parameter = self._parameter(group)
        group_name = str(group["name"])
        state = self.state[parameter]
        selected_parameter = parameter.index_select(
            0,
            visible_indices,
        ).to(dtype=torch.float32)
        if _is_color_group(group_name):
            selected_parameter = selected_parameter.to(dtype=torch.float16).to(dtype=torch.float32)

        if parameter.grad is None:
            selected_gradient = torch.zeros_like(selected_parameter)
        else:
            selected_gradient = parameter.grad.index_select(
                0,
                visible_indices,
            ).to(dtype=torch.float32)
        if _is_color_group(group_name):
            selected_gradient = selected_gradient.to(dtype=torch.float16).to(dtype=torch.float32)

        first_moment = state["exp_avg"]
        second_moment = state["exp_avg_sq"]
        selected_first = first_moment.index_select(
            0,
            visible_indices,
        ).to(dtype=torch.float32)
        selected_second = second_moment.index_select(
            0,
            visible_indices,
        ).to(dtype=torch.float32)
        beta1, beta2 = (float(value) for value in group["betas"])
        selected_first.mul_(beta1).add_(
            selected_gradient,
            alpha=1.0 - beta1,
        )
        selected_second.mul_(beta2).addcmul_(
            selected_gradient,
            selected_gradient,
            value=1.0 - beta2,
        )
        bias_correction = torch.sqrt(1.0 - torch.pow(beta2, update_steps)) / (1.0 - torch.pow(beta1, update_steps))
        normalized_update = selected_first / (torch.sqrt(selected_second) + float(group["eps"]))
        normalized_update.mul_(bias_correction[:, None])

        if group_name == "positions":
            decay = torch.clamp(
                torch.pow(POSITION_DECAY, update_steps),
                min=POSITION_DECAY_FLOOR,
            )
            normalized_update.mul_(decay[:, None])
        elif _is_color_group(group_name):
            decay = torch.clamp(
                torch.pow(COLOR_DECAY, update_steps),
                min=COLOR_DECAY_FLOOR,
            )
            normalized_update.mul_(decay[:, None])
        elif group_name == "density":
            normalized_update.add_(OPACITY_REGULARIZER)
        elif group_name == "scale":
            normalized_update.add_(ordered_log_scale_regularizer(selected_parameter))

        updated_parameter = selected_parameter - (float(group["lr"]) * normalized_update)
        if _is_color_group(group_name):
            updated_parameter = updated_parameter.to(dtype=torch.float16).to(dtype=parameter.dtype)
        else:
            updated_parameter = updated_parameter.to(dtype=parameter.dtype)
        parameter.index_copy_(0, visible_indices, updated_parameter)
        first_moment.index_copy_(
            0,
            visible_indices,
            selected_first.to(dtype=first_moment.dtype),
        )
        second_moment.index_copy_(
            0,
            visible_indices,
            selected_second.to(dtype=second_moment.dtype),
        )
