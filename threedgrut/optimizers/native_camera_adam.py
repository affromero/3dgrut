"""Per-image camera Adam recovered from the visibility-adaptive CUDA kernel."""

from collections.abc import Iterable

import torch

NATIVE_CAMERA_BETAS = (0.8, 0.95)
NATIVE_CAMERA_EPSILON = 1.0e-7
NATIVE_CAMERA_QVEC_LR = 1.0e-5
NATIVE_CAMERA_TVEC_LR = 1.0e-3
_CAMERA_GROUP_NAMES = ("qvecs", "tvecs")


class NativeCameraAdam(torch.optim.Optimizer):
    """Adam with one shared qvec/tvec counter for each source image."""

    def __init__(
        self,
        params: Iterable[dict[str, object]],
        *,
        qvec_lr: float = NATIVE_CAMERA_QVEC_LR,
        tvec_lr: float = NATIVE_CAMERA_TVEC_LR,
        betas: tuple[float, float] = NATIVE_CAMERA_BETAS,
        eps: float = NATIVE_CAMERA_EPSILON,
    ) -> None:
        """Initialize the two recovered absolute-extrinsic groups."""
        groups = list(params)
        group_names = tuple(str(group.get("name")) for group in groups)
        if group_names != _CAMERA_GROUP_NAMES:
            msg = (
                "Native camera Adam requires ordered qvecs and tvecs groups; "
                f"got {group_names}."
            )
            raise ValueError(msg)
        groups[0]["lr"] = qvec_lr
        groups[1]["lr"] = tvec_lr
        defaults = {"betas": betas, "eps": eps}
        super().__init__(groups, defaults)
        self._validate_parameters()

    @staticmethod
    def _parameter(group: dict[str, object]) -> torch.nn.Parameter:
        parameters = group["params"]
        if not isinstance(parameters, list) or len(parameters) != 1:
            msg = "Native camera groups require exactly one parameter tensor."
            raise ValueError(msg)
        parameter = parameters[0]
        if not isinstance(parameter, torch.nn.Parameter):
            msg = "Native camera optimizer values must be Parameters."
            raise TypeError(msg)
        return parameter

    def _validate_parameters(self) -> None:
        qvecs = self._parameter(self.param_groups[0])
        tvecs = self._parameter(self.param_groups[1])
        if qvecs.ndim != 2 or qvecs.shape[1] != 4:
            msg = "Native camera qvecs must have shape (N, 4)."
            raise ValueError(msg)
        if tvecs.shape != (qvecs.shape[0], 3):
            msg = "Native camera tvecs must have shape (N, 3)."
            raise ValueError(msg)
        if (
            qvecs.dtype is not torch.float32
            or tvecs.dtype is not torch.float32
        ):
            msg = "Native camera parameters must use float32."
            raise ValueError(msg)

    def _initialize_state(self) -> None:
        qvecs = self._parameter(self.param_groups[0])
        tvecs = self._parameter(self.param_groups[1])
        qvec_state = self.state[qvecs]
        if not qvec_state:
            qvec_state["image_steps"] = torch.zeros(
                qvecs.shape[0],
                dtype=torch.int32,
                device=qvecs.device,
            )
            qvec_state["exp_avg"] = torch.zeros_like(qvecs)
            qvec_state["exp_avg_sq"] = torch.zeros_like(qvecs)
        tvec_state = self.state[tvecs]
        if not tvec_state:
            tvec_state["exp_avg"] = torch.zeros_like(tvecs)
            tvec_state["exp_avg_sq"] = torch.zeros_like(tvecs)

    def _restore_and_validate_state(self) -> None:
        self._validate_parameters()
        qvecs = self._parameter(self.param_groups[0])
        tvecs = self._parameter(self.param_groups[1])
        expected = (
            (qvecs, ("image_steps", "exp_avg", "exp_avg_sq")),
            (tvecs, ("exp_avg", "exp_avg_sq")),
        )
        for parameter, keys in expected:
            state = self.state[parameter]
            if any(key not in state for key in keys):
                msg = "Native camera checkpoint is missing optimizer state."
                raise ValueError(msg)
            for key in keys:
                value = state[key]
                expected_shape = (
                    (parameter.shape[0],)
                    if key == "image_steps"
                    else parameter.shape
                )
                if value.shape != expected_shape:
                    msg = "Native camera optimizer state has the wrong shape."
                    raise ValueError(msg)
                expected_dtype = (
                    torch.int32 if key == "image_steps" else torch.float32
                )
                state[key] = value.to(dtype=expected_dtype)

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore per-image counters and FP32 Adam moments."""
        super().load_state_dict(state_dict)
        self._restore_and_validate_state()

    @torch.no_grad()
    def step(self, source_frame_idx: int) -> None:
        """Update one image's absolute qvec and tvec."""
        self._initialize_state()
        qvecs = self._parameter(self.param_groups[0])
        if not 0 <= source_frame_idx < qvecs.shape[0]:
            msg = f"Invalid native camera source frame {source_frame_idx}."
            raise ValueError(msg)

        qvec_state = self.state[qvecs]
        image_steps = qvec_state["image_steps"]
        update_step = int(image_steps[source_frame_idx].item()) + 1
        for group in self.param_groups:
            self._step_group(
                group=group,
                source_frame_idx=source_frame_idx,
                update_step=update_step,
            )
        image_steps[source_frame_idx] = update_step

        updated_qvec = qvecs[source_frame_idx]
        qvec_norm = torch.linalg.vector_norm(updated_qvec)
        if not bool(torch.isfinite(qvec_norm)) or qvec_norm <= 0.0:
            msg = "Native camera Adam produced an invalid quaternion."
            raise RuntimeError(msg)
        updated_qvec.div_(qvec_norm)

    def _step_group(
        self,
        *,
        group: dict[str, object],
        source_frame_idx: int,
        update_step: int,
    ) -> None:
        parameter = self._parameter(group)
        state = self.state[parameter]
        selected_parameter = parameter[source_frame_idx]
        if parameter.grad is None:
            gradient = torch.zeros_like(selected_parameter)
        else:
            gradient = parameter.grad[source_frame_idx]
        first_moment = state["exp_avg"][source_frame_idx]
        second_moment = state["exp_avg_sq"][source_frame_idx]
        beta1, beta2 = group["betas"]
        beta1 = float(beta1)
        beta2 = float(beta2)
        first_moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        second_moment.mul_(beta2).addcmul_(
            gradient,
            gradient,
            value=1.0 - beta2,
        )
        bias_scale = (1.0 - beta2**update_step) ** 0.5
        bias_scale /= 1.0 - beta1**update_step
        normalized_update = first_moment / (
            torch.sqrt(second_moment) + float(group["eps"])
        )
        selected_parameter.add_(
            normalized_update,
            alpha=-float(group["lr"]) * bias_scale,
        )
