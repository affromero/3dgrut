"""Per-Gaussian optimizer semantics recovered from visibility-adaptive CUDA kernels."""

from collections.abc import Iterable

import torch

VISIBILITY_SELECTIVE_REQUIRED_GROUP_NAMES = frozenset(
    ("positions", "density", "features_albedo", "rotation", "scale")
)
VISIBILITY_SELECTIVE_OPTIONAL_GROUP_NAMES = frozenset(
    ("features_specular",)
)
VISIBILITY_SELECTIVE_GROUP_NAMES = (
    VISIBILITY_SELECTIVE_REQUIRED_GROUP_NAMES
    | VISIBILITY_SELECTIVE_OPTIONAL_GROUP_NAMES
)
VISIBILITY_SELECTIVE_COUNTER_MAX = 65_535
VISIBILITY_SELECTIVE_DEFAULT_EPSILON = 1.0e-10
POSITION_UPDATE_DECAY = 0.9998
POSITION_UPDATE_DECAY_FLOOR = 0.01
RADIANCE_UPDATE_DECAY = 0.95
RADIANCE_UPDATE_DECAY_FLOOR = 0.025
ENCODER_REGULARIZER = 0.01
DENSITY_REGULARIZER = 0.001
SCALE_REGULARIZER = 0.01
SCALE_RANGE_THRESHOLD = 8.0


def _is_radiance_group(group_name: str) -> bool:
    return group_name in {"features_albedo", "features_specular"}


def scale_regularizer(log_scales: torch.Tensor) -> torch.Tensor:
    """Return the scale regularizer in normalized-Adam update space."""
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


class VisibilitySelectiveAdam(torch.optim.Optimizer):
    """Visibility-selective Adam for per-Gaussian state."""

    def __init__(
        self,
        params: Iterable[dict[str, object]],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = VISIBILITY_SELECTIVE_DEFAULT_EPSILON,
    ) -> None:
        """Initialize visibility-selective per-Gaussian parameter groups."""
        parameter_groups = list(params)
        group_names = frozenset(
            str(group["name"]) for group in parameter_groups
        )
        if not VISIBILITY_SELECTIVE_REQUIRED_GROUP_NAMES.issubset(
            group_names
        ) or not group_names.issubset(VISIBILITY_SELECTIVE_GROUP_NAMES):
            msg = (
                "Visibility-adaptive Adam requires base groups "
                f"{sorted(VISIBILITY_SELECTIVE_REQUIRED_GROUP_NAMES)} and "
                f"supports {sorted(VISIBILITY_SELECTIVE_OPTIONAL_GROUP_NAMES)}; "
                "got "
                f"{sorted(group_names)}."
            )
            raise ValueError(msg)
        defaults = {"lr": lr, "betas": betas, "eps": eps}
        super().__init__(parameter_groups, defaults)

    @staticmethod
    def _parameter(group: dict[str, object]) -> torch.nn.Parameter:
        parameters = group["params"]
        if not isinstance(parameters, list) or len(parameters) != 1:
            msg = "Visibility-adaptive Adam requires one tensor per parameter group."
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
        state_dtype = (
            torch.float16
            if _is_radiance_group(group_name)
            else parameter.dtype
        )
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
            state = self.state[parameter]
            if "exp_avg" not in state or "exp_avg_sq" not in state:
                msg = (
                    "Visibility-selective checkpoint is missing Adam moments for "
                    f"{group_name}."
                )
                raise ValueError(msg)
            expected_dtype = (
                torch.float16
                if _is_radiance_group(group_name)
                else parameter.dtype
            )
            state["exp_avg"] = state["exp_avg"].to(dtype=expected_dtype)
            state["exp_avg_sq"] = state["exp_avg_sq"].to(dtype=expected_dtype)
            if point_count is None:
                point_count = parameter.shape[0]
            elif parameter.shape[0] != point_count:
                msg = (
                    "Visibility-selective parameter groups have inconsistent "
                    "Gaussian counts."
                )
                raise ValueError(msg)
            if group_name != "positions":
                continue
            if "gaussian_steps" not in state:
                msg = "Visibility-selective checkpoint is missing Gaussian steps."
                raise ValueError(msg)
            gaussian_steps = state["gaussian_steps"].to(dtype=torch.int32)
            if gaussian_steps.shape != (parameter.shape[0],):
                msg = "Gaussian-step state has the wrong shape."
                raise ValueError(msg)
            state["gaussian_steps"] = gaussian_steps

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Load state while preserving non-parameter state dtypes."""
        super().load_state_dict(state_dict)
        self._restore_and_validate_state()

    @torch.no_grad()
    def step(
        self,
        visibility: torch.Tensor,
        *,
        advance_age: bool = True,
        age_increment: int = 1,
    ) -> None:
        """Update visible rows and optionally advance their image-age counter."""
        if isinstance(age_increment, bool) or not isinstance(
            age_increment, int
        ):
            raise TypeError("Visibility-selective age increment must be an int.")
        if age_increment <= 0:
            raise ValueError(
                "Visibility-selective age increment must be positive."
            )
        position_group = next(
            group
            for group in self.param_groups
            if str(group["name"]) == "positions"
        )
        position_parameter = self._parameter(position_group)
        point_count = position_parameter.shape[0]
        visible = visibility.reshape(-1).to(dtype=torch.bool)
        if visible.shape != (point_count,):
            msg = (
                "Visibility-selective visibility must have one value per Gaussian."
            )
            raise ValueError(msg)

        for group in self.param_groups:
            parameter = self._parameter(group)
            if parameter.shape[0] != point_count:
                msg = (
                    "Visibility-selective parameter groups have inconsistent "
                    "Gaussian counts."
                )
                raise ValueError(msg)
            self._initialize_group_state(group=group, parameter=parameter)

        visible_indices = torch.nonzero(visible, as_tuple=False).squeeze(1)
        if visible_indices.numel() == 0:
            return

        position_state = self.state[position_parameter]
        gaussian_steps = position_state["gaussian_steps"]
        update_steps = gaussian_steps.index_select(0, visible_indices) + 1
        update_steps_float = update_steps.to(dtype=torch.float32)

        for group in self.param_groups:
            self._step_group(
                group=group,
                visible_indices=visible_indices,
                update_steps=update_steps_float,
            )

        if advance_age:
            saturated_steps = torch.clamp(
                gaussian_steps.index_select(0, visible_indices)
                + age_increment,
                max=VISIBILITY_SELECTIVE_COUNTER_MAX,
            )
            gaussian_steps.index_copy_(0, visible_indices, saturated_steps)

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
        if _is_radiance_group(group_name):
            selected_parameter = selected_parameter.to(dtype=torch.float16).to(
                dtype=torch.float32
            )

        if parameter.grad is None:
            selected_gradient = torch.zeros_like(selected_parameter)
        else:
            selected_gradient = parameter.grad.index_select(
                0,
                visible_indices,
            ).to(dtype=torch.float32)
        if _is_radiance_group(group_name):
            selected_gradient = selected_gradient.to(dtype=torch.float16).to(
                dtype=torch.float32
            )
        first_moment = state["exp_avg"]
        second_moment = state["exp_avg_sq"]
        selected_first_moment = first_moment.index_select(
            0,
            visible_indices,
        ).to(dtype=torch.float32)
        selected_second_moment = second_moment.index_select(
            0,
            visible_indices,
        ).to(dtype=torch.float32)
        beta1, beta2 = group["betas"]
        beta1 = float(beta1)
        beta2 = float(beta2)
        epsilon = float(group["eps"])
        learning_rate = float(group["lr"])

        selected_first_moment.mul_(beta1).add_(
            selected_gradient,
            alpha=1.0 - beta1,
        )
        selected_second_moment.mul_(beta2).addcmul_(
            selected_gradient,
            selected_gradient,
            value=1.0 - beta2,
        )
        normalized_update = selected_first_moment / (
            torch.sqrt(selected_second_moment) + epsilon
        )
        bias_scale = torch.sqrt(1.0 - torch.pow(beta2, update_steps))
        bias_scale.div_(1.0 - torch.pow(beta1, update_steps))
        normalized_update.mul_(bias_scale[:, None])

        if group_name == "positions":
            decay = torch.clamp(
                torch.pow(POSITION_UPDATE_DECAY, update_steps),
                min=POSITION_UPDATE_DECAY_FLOOR,
            )
            normalized_update.mul_(decay[:, None])
        elif _is_radiance_group(group_name):
            decay = torch.clamp(
                torch.pow(RADIANCE_UPDATE_DECAY, update_steps),
                min=RADIANCE_UPDATE_DECAY_FLOOR,
            )
            normalized_update.mul_(decay[:, None])
        elif group_name == "density":
            normalized_update.add_(DENSITY_REGULARIZER)
        elif group_name == "scale":
            normalized_update.add_(scale_regularizer(selected_parameter))

        updated_parameter = selected_parameter - (
            learning_rate * normalized_update
        )
        if _is_radiance_group(group_name):
            updated_parameter = updated_parameter.to(dtype=torch.float16).to(
                dtype=parameter.dtype
            )
        else:
            updated_parameter = updated_parameter.to(dtype=parameter.dtype)
        parameter.index_copy_(0, visible_indices, updated_parameter)
        first_moment.index_copy_(
            0,
            visible_indices,
            selected_first_moment.to(dtype=first_moment.dtype),
        )
        second_moment.index_copy_(
            0,
            visible_indices,
            selected_second_moment.to(dtype=second_moment.dtype),
        )


class FP16GlobalAdam(torch.optim.Optimizer):
    """Global FP16 Adam for a compact color-encoder parameter group."""

    def __init__(
        self,
        params: Iterable[dict[str, object]],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1.0e-10,
    ) -> None:
        """Initialize the single shared color-encoder parameter group."""
        parameter_groups = list(params)
        group_names = [str(group["name"]) for group in parameter_groups]
        if group_names != ["color_encoder"]:
            msg = (
                "Global FP16 Adam requires one color_encoder group; got "
                f"{group_names}."
            )
            raise ValueError(msg)
        defaults = {"lr": lr, "betas": betas, "eps": eps}
        super().__init__(parameter_groups, defaults)

    @staticmethod
    def _parameter(group: dict[str, object]) -> torch.nn.Parameter:
        parameters = group["params"]
        if not isinstance(parameters, list) or len(parameters) != 1:
            msg = "Global FP16 Adam requires one encoder tensor."
            raise ValueError(msg)
        parameter = parameters[0]
        if not isinstance(parameter, torch.nn.Parameter):
            msg = "Encoder optimizer value must be a Parameter."
            raise TypeError(msg)
        return parameter

    def _initialize_state(self, parameter: torch.nn.Parameter) -> None:
        state = self.state[parameter]
        if state:
            return
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(
            parameter,
            dtype=torch.float16,
            memory_format=torch.preserve_format,
        )
        state["exp_avg_sq"] = torch.zeros_like(
            parameter,
            dtype=torch.float16,
            memory_format=torch.preserve_format,
        )

    def _restore_and_validate_state(self) -> None:
        group = self.param_groups[0]
        parameter = self._parameter(group)
        state = self.state[parameter]
        if "step" not in state:
            msg = "Encoder checkpoint is missing its Adam step."
            raise ValueError(msg)
        if "exp_avg" not in state or "exp_avg_sq" not in state:
            msg = "Encoder checkpoint is missing Adam moments."
            raise ValueError(msg)
        step = int(state["step"])
        if step < 0:
            msg = "Encoder Adam step cannot be negative."
            raise ValueError(msg)
        state["step"] = step
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key]
            if value.shape != parameter.shape:
                msg = "Encoder Adam state has the wrong shape."
                raise ValueError(msg)
            state[key] = value.to(dtype=torch.float16)

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore the global counter and FP16 moments."""
        super().load_state_dict(state_dict)
        self._restore_and_validate_state()

    @torch.no_grad()
    def step(self) -> None:
        """Update the complete shared encoder with FP16 storage."""
        group = self.param_groups[0]
        parameter = self._parameter(group)
        self._initialize_state(parameter)
        state = self.state[parameter]
        state["step"] = int(state["step"]) + 1
        step = int(state["step"])

        selected_parameter = parameter.to(dtype=torch.float16).to(
            dtype=torch.float32
        )
        if parameter.grad is None:
            gradient = torch.zeros_like(selected_parameter)
        else:
            gradient = parameter.grad.to(dtype=torch.float16).to(
                dtype=torch.float32
            )
        first_moment = state["exp_avg"]
        second_moment = state["exp_avg_sq"]
        updated_first_moment = first_moment.to(dtype=torch.float32)
        updated_second_moment = second_moment.to(dtype=torch.float32)
        beta1, beta2 = group["betas"]
        beta1 = float(beta1)
        beta2 = float(beta2)
        updated_first_moment.mul_(beta1).add_(
            gradient,
            alpha=1.0 - beta1,
        )
        updated_second_moment.mul_(beta2).addcmul_(
            gradient,
            gradient,
            value=1.0 - beta2,
        )
        bias_correction = (1.0 - beta2**step) ** 0.5 / (1.0 - beta1**step)
        normalized_update = updated_first_moment / (
            torch.sqrt(updated_second_moment) + float(group["eps"])
        )
        normalized_update.mul_(bias_correction)
        normalized_update.add_(
            selected_parameter,
            alpha=ENCODER_REGULARIZER,
        )
        updated_parameter = selected_parameter - (
            float(group["lr"]) * normalized_update
        )
        parameter.copy_(
            updated_parameter.to(dtype=torch.float16).to(dtype=parameter.dtype)
        )
        state["exp_avg"] = updated_first_moment.to(dtype=torch.float16)
        state["exp_avg_sq"] = updated_second_moment.to(dtype=torch.float16)
