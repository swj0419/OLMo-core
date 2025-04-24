from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Union
from torch.nn import functional as F
import torch
import math

__all__ = ["MoELoss", "MoELoadBalancingLoss", "MoERouterZLoss"]


class MoELoss(metaclass=ABCMeta):
    @abstractmethod
    def update(
        self,
        *,
        expert_logits: torch.Tensor,
        expert_scores: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
        **kwargs,
    ):
        raise NotImplementedError

    @abstractmethod
    def compute(
        self, total_bz: Union[int, float, torch.Tensor], reset: bool = True, **kwargs
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError


class MoELoadBalancingLoss(MoELoss):
    """
    Implements the load balancing loss from Switch Transformers.
    """

    def __init__(self, *, loss_weight: float, num_experts: int, top_k: int):
        self.loss_weight = loss_weight
        self.num_experts = num_experts
        self.top_k = top_k
        self.loss: Optional[torch.Tensor] = None
        self.expert_scores: Optional[torch.Tensor] = None

        # swj
        self.decay_steps: int = 12000
        self.decay_style: str = "linear"
        self.min_scale: float = 0.0

    def update(
        self,
        *,
        expert_scores: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
        **kwargs,
    ):
        del kwargs
        # shape: (batch_size, num_experts) -> (num_experts,)
        expert_scores = expert_scores.mean(dim=0)
        self.expert_scores = expert_scores

        # original:
        # loss = torch.dot(batch_size_per_expert.type_as(expert_scores), expert_scores)
        # swj change:
        probs = F.softmax(expert_scores, dim=0)
        log_probs = torch.log(probs + 1e-10)  # Add small epsilon to avoid log(0)
        neg_entropy = torch.sum(probs * log_probs)
        loss = neg_entropy
        # from ipdb import set_trace as bp; bp()
        
        if self.loss is None:
            self.loss = loss
        else:
            self.loss += loss

    # swj change
    def get_scale(self, step: int, loss_weight: float) -> float:
        """
        Calculate the current scale factor based on training step.
        
        Args:
            step: Current training step
            
        Returns:
            Current scale for the load balancing loss
        """
        if step >= self.decay_steps:
            return self.min_scale
        
        progress = step / self.decay_steps
        
        if self.decay_style == "linear":
            # Linear decay from initial_scale to min_scale
            scale = loss_weight - (loss_weight - self.min_scale) * progress
        
        elif self.decay_style == "cosine":
            # Cosine decay from initial_scale to min_scale
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            scale = self.min_scale + (loss_weight - self.min_scale) * cosine_decay
        
        elif self.decay_style == "exponential":
            # Exponential decay from initial_scale to min_scale
            decay_rate = -math.log(self.min_scale / loss_weight) if self.min_scale > 0 else 5.0
            scale = loss_weight * math.exp(-decay_rate * progress)
            scale = max(scale, self.min_scale)
        
        else:
            raise ValueError(f"Unknown decay style: {self.decay_style}")
        
        return scale


    def compute(
        self, total_bz: Union[int, float, torch.Tensor], reset: bool = True, step: Optional[int] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        del kwargs

        if self.loss is None:
            # from ipdb import set_trace as bp; bp()
            raise RuntimeError(
                f"'{self.__class__.__name__}.update()' needs to be called before '.compute()'"
            )
        # swj change:
        # scale = (self.num_experts * self.loss_weight) / (total_bz * self.top_k)
        # from ipdb import set_trace as bp; bp()
        current_scale = self.get_scale(step, self.loss_weight)
        # make current_scale a tensor
        current_scale = torch.tensor(current_scale)
        # print("step: ", step, "current_scale: ", current_scale)
        lb_loss = current_scale * self.loss
        if reset:
            self.reset()

        expert_scores_dict = {f"expert_{i}": self.expert_scores[i] for i in range(self.num_experts)}
        # from ipdb import set_trace as bp
        # bp()
        # expert_scale = {f"expert_scale": current_scale}
        # print("expert_scores_dict: ", expert_scores_dict)

        # from ipdb import set_trace as bp; bp()
        final_dict = {"load balancing loss": lb_loss}
        # final_dict.update(expert_scores_dict)
        # final_dict.update(expert_scale)
        # print("final_dict: ", final_dict)
        # from ipdb import set_trace as bp; bp()
        return final_dict

    def reset(self):
        self.loss = None


class MoERouterZLoss(MoELoss):
    def __init__(self, *, loss_weight: float, num_experts: int):
        self.loss_weight = loss_weight
        self.num_experts = num_experts
        self.loss: Optional[torch.Tensor] = None

    def update(self, *, expert_logits: torch.Tensor, **kwargs):
        del kwargs
        loss = torch.logsumexp(expert_logits, dim=-1).square().sum()
        if self.loss is None:
            self.loss = loss
        else:
            self.loss += loss

    def compute(
        self, total_bz: Union[int, float, torch.Tensor], reset: bool = True, **kwargs
    ) -> Dict[str, torch.Tensor]:
        del kwargs
        if self.loss is None:
            raise RuntimeError(
                f"'{self.__class__.__name__}.update()' needs to be called before '.compute()'"
            )
        scale = self.loss_weight / total_bz
        lb_loss = scale * self.loss
        if reset:
            self.reset()
        return {"router Z loss": lb_loss, }

    def reset(self):
        self.loss = None
