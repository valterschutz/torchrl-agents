from tensordict import TensorDictBase
from torchrl.data import TensorDictReplayBuffer
from tensordict.nn import ProbabilisticTensorDictSequential, TensorDictModule
from torch import Tensor, nn, optim
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
import torch
from torchrl.data import LazyTensorStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl_agents import Agent, unserializable, weights
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T", bound="PPOAgent")


@dataclass
class PPOAgent(Agent, ABC):
    """Proximal Policy Optimization (PPO) agent."""

    # Device. All modules created by subclasses will be moved to this device.
    _device: torch.device = unserializable(default_factory=lambda: torch.device("cpu"))

    # PPO parameters
    gamma: float = unserializable(default=1)
    lmbda: float = unserializable(default=0.95)
    clip_epsilon: float = unserializable(default=0.2)  # weight clipping threshold
    entropy_bonus: bool = unserializable(
        default=True
    )  # whether to encourage exploration
    entropy_coef: float = unserializable(
        default=1e-4
    )  # how much to weight the entropy loss term
    critic_coef: float = unserializable(
        default=1.0
    )  # how much to weight the critic loss term
    loss_critic_type: str = unserializable(
        default="smooth_l1"
    )  # what type of loss to use for the critic

    # Optimizer parameters
    lr: float = unserializable(default=1e-3)
    max_grad_norm: float = unserializable(default=1)

    # Replay buffer parameters
    batch_size: int = unserializable(default=1000)
    sub_batch_size: int = unserializable(
        default=100
    )  # size of batch when sampling from replay buffer
    num_epochs: int = unserializable(default=10)
    replay_buffer_device: torch.device = unserializable(
        default_factory=lambda: torch.device("cpu")
    )

    # Set in constructor
    policy_module: ProbabilisticTensorDictSequential = weights(init=False)
    state_value_module: TensorDictModule = weights(init=False)
    advantage_module: GAE = field(init=False)
    loss_module: ClipPPOLoss = field(init=False)
    loss_keys: list[str] = field(init=False)
    optimizer: optim.Adam = field(init=False)
    replay_buffer: ReplayBuffer = field(init=False)

    @abstractmethod
    def get_policy_module(self) -> ProbabilisticTensorDictSequential:
        """Get the policy module."""
        pass

    @abstractmethod
    def get_state_value_module(self) -> TensorDictModule:
        """Get the state value module."""
        pass

    def __post_init__(self) -> None:
        # Ensure batch_size is divisible by sub_batch_size
        if self.batch_size % self.sub_batch_size != 0:
            raise ValueError("batch_size must be divisible by sub_batch_size.")

        self.policy_module = self.get_policy_module().to(self._device)
        self.state_value_module = self.get_state_value_module().to(self._device)

        self.advantage_module = GAE(
            gamma=self.gamma,
            lmbda=self.lmbda,
            value_network=self.state_value_module,
            average_gae=True,
        )
        self.loss_module = ClipPPOLoss(
            actor_network=self.policy_module,
            critic_network=self.state_value_module,
            clip_epsilon=self.clip_epsilon,
            entropy_bonus=self.entropy_bonus,
            entropy_coef=self.entropy_coef,
            critic_coef=self.critic_coef,
            loss_critic_type=self.loss_critic_type,
        )
        self.loss_keys = ["loss_objective", "loss_critic"] + (
            ["loss_entropy"] if self.entropy_bonus else []
        )
        self.optimizer = optim.Adam(self.loss_module.parameters(), lr=self.lr)
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                max_size=self.batch_size, device=self.replay_buffer_device
            ),
            sampler=SamplerWithoutReplacement(),
            batch_size=self.sub_batch_size,
        )

    @property
    def policy(self) -> ProbabilisticTensorDictSequential:
        return self.policy_module

    def process_batch(self, td: TensorDictBase) -> dict[str, float]:
        """Process a batch of training data, returning the average loss for each loss key."""
        assert td.batch_size == torch.Size((self.batch_size,)), "Batch size mismatch"

        # Initialize total loss dictionary
        total_loss_td = {k: 0.0 for k in self.loss_keys}

        # Perform multiple epochs of training
        for _ in range(self.num_epochs):
            # Compute advantages each epoch
            self.advantage_module(td)

            # Reset replay buffer each epoch
            self.replay_buffer.extend(td)  # type: ignore

            for _ in range(self.batch_size // self.sub_batch_size):
                loss_td = self._sample_and_train()

                # Accumulate losses
                for k in self.loss_keys:
                    total_loss_td[k] += loss_td[k].item()

        # Compute average loss
        num_updates = self.num_epochs * (self.batch_size // self.sub_batch_size)
        avg_loss_td = {k: v / num_updates for k, v in total_loss_td.items()}

        return avg_loss_td

    def _sample_and_train(self) -> TensorDictBase:
        """Sample from the replay buffer and train the policy, returning the loss td."""
        td = self.replay_buffer.sample()
        td = td.to(self._device)

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

        return loss_td

    @property
    def device(self) -> torch.device:
        """Get the device of the agent."""
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self._device = device
        self.policy_module = self.policy_module.to(self._device)
        self.state_value_module = self.state_value_module.to(self._device)
