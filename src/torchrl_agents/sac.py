from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from torchrl.objectives import SACLoss
from tensordict import TensorDictBase
from torchrl.objectives import SoftUpdate, ValueEstimators
from torchrl.data import TensorDictReplayBuffer
from tensordict.nn import TensorDictModule, TensorDictModuleBase
from torch import Tensor, nn, optim
import torch
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorSpec
from torchrl_agents import Agent, serializable, unserializable, weights
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T", bound="SACAgent")


@dataclass(kw_only=True, eq=False, order=False)
class SACAgent(Agent, ABC):
    """Soft Actor-Critic (SAC) agent."""

    action_spec: TensorSpec = unserializable()

    _device: torch.device = unserializable(default_factory=lambda: torch.device("cpu"))

    gamma: float = serializable()

    # SAC parameters
    num_qvalue_nets: int = serializable(default=2)
    loss_function: str = serializable(default="smooth_l1")
    alpha_init: bool = serializable(default=False)
    min_alpha: bool = serializable(default=True)
    max_alpha: bool = serializable(default=False)
    fixed_alpha: bool = serializable(default=False)
    target_entropy: float | str = serializable(default="auto")
    delay_actor: bool = serializable(default=False)
    delay_qvalue: bool = serializable(default=True)
    delay_value: bool = serializable(default=True)
    separate_losses: bool = serializable(default=False)
    reduction: str = serializable(default="mean")
    skip_done_states: bool = serializable(default=False)

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
    state_value_module: TensorDictModule = weights(init=False)
    state_action_value_module: TensorDictModule = weights(init=False)
    loss_module: SACLoss = field(init=False)
    target_net_updater: SoftUpdate = field(init=False)
    loss_keys: list[str] = field(init=False)
    optimizer: optim.Adam = field(init=False)
    replay_buffer: ReplayBuffer = field(init=False)

    def pre_init_hook(self) -> None:
        """Hook for subclasses to optionally run code before __post_init__."""
        pass

    def post_init_hook(self) -> None:
        """Hook for subclasses to optionally run code after __post_init__."""
        pass

    @abstractmethod
    def get_policy_module(self) -> TensorDictModuleBase:
        """Get the action value module."""
        pass

    @abstractmethod
    def get_state_value_module(self) -> TensorDictModule:
        """Get the action value module."""
        pass

    @abstractmethod
    def get_state_action_value_module(self) -> TensorDictModuleBase:
        """Get the action value module."""
        pass

    def __post_init__(self) -> None:
        self.pre_init_hook()

        self.policy_module = self.get_policy_module().to(self._device)
        self.state_value_module = self.get_state_value_module().to(self._device)
        self.state_action_value_module = self.get_state_action_value_module().to(
            self._device
        )

        self.loss_module = SACLoss(
            actor_network=self.policy_module,
            qvalue_network=self.state_action_value_module,
            value_network=self.state_value_module,
            num_qvalue_nets=self.num_qvalue_nets,
            loss_function=self.loss_function,
            alpha_init=self.alpha_init,
            min_alpha=self.min_alpha,
            max_alpha=self.max_alpha,
            action_spec=self.action_spec,
            fixed_alpha=self.fixed_alpha,
            target_entropy=self.target_entropy,
            delay_actor=self.delay_actor,
            delay_qvalue=self.delay_qvalue,
            delay_value=self.delay_value,
            priority_key="td_error",
            separate_losses=self.separate_losses,
            reduction=self.reduction,
            skip_done_states=self.skip_done_states,
        )
        self.loss_module.make_value_estimator(ValueEstimators.TD0, gamma=self.gamma)
        self.target_net_updater = SoftUpdate(self.loss_module, eps=1 - self.update_tau)
        self.loss_keys = ["loss_actor", "loss_alpha", "loss_qvalue", "loss_value"]
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

        self.post_init_hook()

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

    @property
    def device(self) -> torch.device:
        """Get the device of the agent."""
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self._device = device
        self.policy_module = self.policy_module.to(self._device)
        self.action_value_module = self.action_value_module.to(self._device)
