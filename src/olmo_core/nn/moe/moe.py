import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement, Replicate, Shard
from torch.distributed.tensor.parallel import (
    PrepareModuleInput,
    PrepareModuleOutput,
    parallelize_module,
)

from olmo_core.config import Config, DType, StrEnum
from olmo_core.exceptions import OLMoConfigurationError

from ..buffer_cache import BufferCache
from ..feed_forward import FeedForwardConfig
from .loss import MoELoadBalancingLoss, MoELoss, MoERouterZLoss
from .metric import MoELoadImbalanceMetric, MoEMetric
from .mlp import DroplessMoEMLP, MoEMLP
from .parallel_mlp import ParallelDroplessMLP, ParallelMLP, ParallelMLPBase
from .router import MoERouterConfig

if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType

__all__ = ["MoEBase", "MoE", "DroplessMoE", "MoEConfig", "MoEType"]


log = logging.getLogger(__name__)


class MoEType(StrEnum):
    """
    An enumeration of the different MoE implementations.
    """

    default = "default"
    """
    ➡️ :class:`MoE`
    """

    dropless = "dropless"
    """
    ➡️ :class:`DroplessMoE`
    """


@dataclass
class MoEConfig(Config):
    name: MoEType = MoEType.default
    """
    The name of the implementation.
    """
    num_experts: int = 1
    hidden_size: int = 256
    capacity_factor: Optional[float] = None
    router: MoERouterConfig = field(default_factory=MoERouterConfig)
    shared_mlp: Optional[FeedForwardConfig] = None
    lb_loss_weight: Optional[float] = 1.0
    z_loss_weight: Optional[float] = None
    dtype: DType = DType.float32

    def num_params(self, d_model: int) -> int:
        num_params = 0
        num_params += self.router.num_params(d_model, self.num_experts)
        num_params += 3 * d_model * self.hidden_size * self.num_experts
        if self.shared_mlp is not None:
            num_params += self.shared_mlp.num_params(d_model)
        return num_params

    def num_active_params(self, d_model: int) -> int:
        return (
            self.num_params(d_model)
            - (3 * d_model * self.hidden_size * self.num_experts)
            + (3 * d_model * self.hidden_size * self.router.top_k)
        )

    def build(
        self,
        d_model: int,
        *,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> "MoEBase":
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")
        kwargs.update(
            d_model=d_model,
            init_device=init_device,
            dtype=kwargs.pop("dtype").as_pt(),
            cache=cache,
        )

        try:
            if self.name == MoEType.default:
                return MoE(**kwargs)
            elif self.name == MoEType.dropless:
                return DroplessMoE(**kwargs)
            else:
                raise NotImplementedError(self.name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e


class MoEBase(nn.Module):
    """
    Base class for MoE implementations.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_experts: int,
        hidden_size: int,
        router: MoERouterConfig,
        shared_mlp: Optional[FeedForwardConfig] = None,
        init_device: str = "cpu",
        lb_loss_weight: Optional[float] = None,
        z_loss_weight: Optional[float] = None,
        dtype: torch.dtype = torch.float32,
        cache: Optional[BufferCache] = None,
        **kwargs,
    ):
        super().__init__()
        self.router = router.build(d_model, num_experts, dtype=dtype, init_device=init_device)
        self.experts = self._init_parallel_mlp(
            d_model=d_model,
            num_experts=num_experts,
            hidden_size=hidden_size,
            dtype=dtype,
            init_device=init_device,
            cache=cache,
            **kwargs,
        )
        self.shared_mlp = (
            None
            if shared_mlp is None
            else shared_mlp.build(d_model, dtype=dtype, init_device=init_device)
        )
        self.losses: List[MoELoss] = []
        self.metrics: List[MoEMetric] = [
            MoELoadImbalanceMetric(num_experts=num_experts, top_k=self.router.top_k)
        ]
        if lb_loss_weight is not None:
            self.losses.append(
                MoELoadBalancingLoss(
                    loss_weight=lb_loss_weight,
                    num_experts=num_experts,
                    top_k=self.router.top_k,
                )
            )
        if z_loss_weight is not None:
            self.losses.append(MoERouterZLoss(loss_weight=z_loss_weight, num_experts=num_experts))

        self._ep_enabled = False

        # Initialize self.losses as an empty list if no losses are provided
        self.losses = [] if lb_loss_weight is None and z_loss_weight is None else self.losses

    @property
    def num_experts(self) -> int:
        return self.router.num_experts

    @property
    def top_k(self) -> int:
        return self.router.top_k

    @property
    def ep_enabled(self) -> bool:
        return self._ep_enabled

    def warmup_cache(self, max_local_microbatch_size: int):
        self.experts.warmup_cache(max_local_microbatch_size)

    def update_losses_and_metrics(
        self,
        *,
        expert_logits: torch.Tensor,
        expert_scores: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
    ):
        if not self.losses and not self.metrics:
            return

        expert_logits = expert_logits.float()

        for loss_fn in self.losses:
            loss_fn.update(
                expert_logits=expert_logits,
                expert_scores=expert_scores,
                expert_weights=expert_weights,
                expert_indices=expert_indices,
                batch_size_per_expert=batch_size_per_expert,
            )

        for metric in self.metrics:
            metric.update(
                expert_logits=expert_logits,
                expert_scores=expert_scores,
                expert_weights=expert_weights,
                expert_indices=expert_indices,
                batch_size_per_expert=batch_size_per_expert,
            )

    def compute_losses(
        self, total_bz: Union[int, float, torch.Tensor], reset: bool = True, step: int = 0
    ) -> Dict[str, torch.Tensor]:
        out = {}
        if not self.losses:  # Check if self.losses is empty or None
            return out
        for loss_fn in self.losses:
            out.update(loss_fn.compute(total_bz, reset=reset, step=step))
        return out

    def reset_losses(self):
        for loss_fn in self.losses:
            loss_fn.reset()

    def compute_metrics(
        self, total_bz: Union[int, float, torch.Tensor], reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        out: Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]] = {}
        for metric in self.metrics:
            out.update(metric.compute(total_bz, reset=reset))
        return out

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset()

    def post_batch(self, dry_run: bool = False):
        """
        Should be called right after the final backward of a complete batch but before the optimizer step.
        """
        self.router.post_batch(dry_run=dry_run)

    @abstractmethod
    def _init_parallel_mlp(
        self,
        *,
        d_model: int,
        num_experts: int,
        hidden_size: int,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        **kwargs,
    ) -> ParallelMLPBase:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the MoE on the input ``x`` of shape ``(*, d_model)``.

        :param x: The input of shape ``(*, d_model)``.

        :returns: The output of the MoE layer, the optional load-balancing loss, and the optional
            router Z-loss.
        """
        (
            expert_logits,
            expert_scores,
            expert_weights,
            expert_indices,
            batch_size_per_expert,
        ) = self.router(x)

        shared_out: Optional[torch.Tensor] = None
        if self.shared_mlp is not None:
            shared_out = self.shared_mlp(x)

        out = self.experts(x, expert_weights, expert_indices, batch_size_per_expert)

        if shared_out is not None:
            shared_out = shared_out / (self.top_k + 1)
            out = shared_out.add(out, alpha=self.top_k / (self.top_k + 1))

        if self.training:
            self.update_losses_and_metrics(
                expert_logits=expert_logits,
                expert_scores=expert_scores,
                expert_weights=expert_weights,
                expert_indices=expert_indices,
                batch_size_per_expert=batch_size_per_expert,
            )

        return out

    def apply_pp(self, pp_mesh: DeviceMesh):
        self.router.pp_group = pp_mesh.get_group()

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        """
        Apply expert parallelism.
        """
        self.experts.apply_ep(ep_mesh, **kwargs)
        self._ep_enabled = True

    def prepare_experts_for_fsdp(self, **kwargs):
        """
        Should be called before wrapping this module with FSDP2.
        """
        self.experts.prepare_experts_for_fsdp(**kwargs)

    def prepare_experts_for_ddp(self, **kwargs):
        """
        Should be called before wrapping this module with DDP2.
        """
        self.experts.prepare_experts_for_ddp(**kwargs)

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        # Sequence parallel for the most part.
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleInput(
                input_layouts=None if input_layout is None else (input_layout,),
                desired_input_layouts=(Shard(1),),
                use_local_output=False,
            ),
        )

        # Sequence parallel.
        self.router.apply_tp(tp_mesh, float8_enabled=float8_enabled)

        # Expert parallel.
        self.experts.apply_tp(tp_mesh, float8_enabled=float8_enabled)

        # Model parallel.
        if self.shared_mlp is not None:
            self.shared_mlp.apply_tp(
                tp_mesh,
                input_layout=Shard(1),
                output_layout=Shard(1),
                use_local_output=True,
                float8_enabled=float8_enabled,
            )

        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleOutput(
                output_layouts=(Shard(1),),
                desired_output_layouts=(output_layout or Replicate(),),
                use_local_output=use_local_output,
            ),
        )


class MoE(MoEBase):
    """
    A basic MoE implementation.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_experts: int,
        hidden_size: int,
        router: MoERouterConfig,
        shared_mlp: Optional[FeedForwardConfig] = None,
        capacity_factor: float = 1.2,
        init_device: str = "cpu",
        lb_loss_weight: Optional[float] = None,
        z_loss_weight: Optional[float] = None,
        dtype: torch.dtype = torch.float32,
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(
            d_model=d_model,
            num_experts=num_experts,
            hidden_size=hidden_size,
            router=router,
            shared_mlp=shared_mlp,
            init_device=init_device,
            lb_loss_weight=lb_loss_weight,
            z_loss_weight=z_loss_weight,
            dtype=dtype,
            capacity_factor=capacity_factor,
            cache=cache,
        )

        # Initialize self.losses as an empty list if no losses are provided
        self.losses = self.losses if self.losses is not None else []

    def _init_parallel_mlp(  # type: ignore[override]
        self,
        *,
        d_model: int,
        num_experts: int,
        hidden_size: int,
        capacity_factor: float,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> ParallelMLP:
        return ParallelMLP(
            mlp=MoEMLP(
                d_model=d_model,
                hidden_size=hidden_size,
                num_experts=num_experts,
                dtype=dtype,
                init_device=init_device,
            ),
            top_k=self.router.top_k,
            capacity_factor=capacity_factor,
            cache=cache,
        )


class DroplessMoE(MoEBase):
    """
    A dropless MoE implementation.
    """

    def _init_parallel_mlp(  # type: ignore[override]
        self,
        *,
        d_model: int,
        num_experts: int,
        hidden_size: int,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> ParallelDroplessMLP:
        return ParallelDroplessMLP(
            mlp=DroplessMoEMLP(
                d_model=d_model,
                num_experts=num_experts,
                hidden_size=hidden_size,
                dtype=dtype,
                init_device=init_device,
            ),
            top_k=self.router.top_k,
            cache=cache,
        )
