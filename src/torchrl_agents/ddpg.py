from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from tensordict import TensorDictBase
from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators
from torchrl.data import TensorDictReplayBuffer
from tensordict.nn import TensorDictModule
from torch import Tensor, nn, optim
import torch
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorSpec
from torchrl_agents import Agent, serializable, unserializable, weights
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T", bound="DDPGAgent")


@dataclass
class DDPGAgent(Agent, ABC):
    """Deep Deterministic Policy Gradient (DDPG) agent."""

    action_spec: TensorSpec = unserializable()

    _device: torch.device = unserializable(default_factory=lambda: torch.device("cpu"))

    # DDPG parameters
    gamma: float = serializable(default=0.99)
    loss_function: str = serializable(default="l2")
    delay_actor: bool = serializable(default=False)
    delay_value: bool = serializable(default=True)
    separate_losses: bool = serializable(default=False)

    # target network update rate
    update_tau: float = serializable(default=0.005)

    # Optimizer parameters
    lr: float = serializable(default=1e-3)
    max_grad_norm: float = serializable(default=1)

    # Replay buffer parameters
    replay_buffer_size: int = serializable(default=1000)
    sub_batch_size: int = serializable(default=100)
    num_samples: int = serializable(default=10)
    replay_buffer_alpha: float = serializable(default=0.6)
    replay_buffer_beta_init: float = serializable(default=0.4)
    replay_buffer_beta_end: float = serializable(default=1.0)
    replay_buffer_beta_annealing_num_batches: int = serializable(default=10000)
    init_random_frames: int = serializable(default=0)
    replay_buffer_device: torch.device = unserializable(
        default_factory=lambda: torch.device("cpu")
    )

    # Set in constructor
    policy_module: TensorDictModule = weights(init=False)
    state_action_value_module: TensorDictModule = weights(init=False)
    loss_module: DDPGLoss = field(init=False)
    target_net_updater: SoftUpdate = field(init=False)
    loss_keys: list[str] = field(init=False)
    optimizer: optim.Adam = field(init=False)
    replay_buffer: ReplayBuffer = field(init=False)

    # Getters to be defined in subclasses

    @abstractmethod
    def get_policy_module(self) -> TensorDictModule:
        """Get the action value module."""
        pass

    @abstractmethod
    def get_state_action_value_module(self) -> TensorDictModule:
        """Get the action value module."""
        pass

    def __post_init__(self) -> None:
        self.policy_module = self.get_policy_module().to(self._device)
        self.state_action_value_module = self.get_state_action_value_module().to(
            self._device
        )

        self.loss_module = DDPGLoss(
            actor_network=self.policy_module,
            value_network=self.state_action_value_module,
        )
        self.loss_module.make_value_estimator(ValueEstimators.TD0, gamma=self.gamma)
        self.target_net_updater = SoftUpdate(self.loss_module, eps=1 - self.update_tau)
        self.loss_keys = ["loss_actor", "loss_value"]
        self.optimizer = optim.Adam(self.loss_module.parameters(), lr=self.lr)
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                max_size=self.replay_buffer_size, device=self.replay_buffer_device
            ),
            sampler=PrioritizedSampler(
                max_capacity=self.replay_buffer_size,
                alpha=self.replay_buffer_alpha,
                beta=self.replay_buffer_beta_init,
            ),
            priority_key="td_error",
            batch_size=self.sub_batch_size,
        )

    @property
    def policy(self) -> TensorDictModule:
        return self.policy_module

    def _anneal_replay_buffer_beta(self) -> None:
        """Anneal the beta parameter for prioritized sampling."""
        if self.replay_buffer.sampler.beta < self.replay_buffer_beta_end:
            self.replay_buffer.sampler.beta += (
                self.replay_buffer_beta_end - self.replay_buffer_beta_init
            ) / self.replay_buffer_beta_annealing_num_batches

    def process_batch(self, td: TensorDictBase) -> dict[str, float]:
        """Process a batch of training data, returning the average loss for each loss key."""
        # Add to replay buffer
        self.replay_buffer.extend(td)  # type: ignore

        # Only continue if we have enough samples in the replay buffer, in which case we return 0.0 for all losses
        if len(self.replay_buffer) < self.init_random_frames:
            return {k: 0.0 for k in self.loss_keys}

        # Initialize total loss dictionary
        total_loss_td = {k: 0.0 for k in self.loss_keys}

        for _ in range(self.num_samples):
            loss_td = self._sample_and_train()

            # Accumulate losses
            for k in self.loss_keys:
                total_loss_td[k] += loss_td[k].item()

        # Update target network
        self.target_net_updater.step()

        # Anneal beta for prioritized sampling
        self._anneal_replay_buffer_beta()

        # Compute average loss
        avg_loss_td = {k: v / self.num_samples for k, v in total_loss_td.items()}

        return avg_loss_td

    def _sample_and_train(self) -> TensorDictBase:
        """Sample from the replay buffer and train the policy, returning the loss td."""
        td = self.replay_buffer.sample()
        td = td.to(self.device)

        self.optimizer.zero_grad()
        loss_td: TensorDictBase = self.loss_module(td)
        loss_tensor: Tensor = sum(
            (loss_td[k] for k in self.loss_keys), torch.tensor(0.0, device=td.device)
        )
        loss_tensor.backward()
        nn.utils.clip_grad_norm_(
            self.loss_module.parameters(), max_norm=self.max_grad_norm
        )
        self.optimizer.step()

        # Update priorities in the replay buffer
        self.replay_buffer.update_tensordict_priority(td)  # type: ignore

        return loss_td

    def get_train_info(self) -> dict[str, Any]:
        """Get training information."""
        return {
            "replay_buffer_beta": self.replay_buffer.sampler.beta,
            "replay_buffer_size": len(self.replay_buffer),
        }

    @property
    def device(self) -> torch.device:
        """Get the device of the agent."""
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self._device = device
        self.policy_module = self.policy_module.to(self._device)
        self.action_value_module = self.action_value_module.to(self._device)
