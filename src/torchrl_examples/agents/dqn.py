from pathlib import Path
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from tensordict.nn import TensorDictSequential
from tensordict import TensorDictBase
from torchrl.objectives import DQNLoss, SoftUpdate, ValueEstimators
from torchrl.modules import EGreedyModule, QValueModule
from torchrl.data import TensorDictReplayBuffer
from tensordict.nn import ProbabilisticTensorDictSequential, TensorDictModule
from torch import Tensor, nn, optim
import torch
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorSpec
import yaml
from torchrl_examples.agents import Agent
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T", bound="DQNAgent")

def field_with_metadata(field_type: str, default=None, default_factory=None, init=True):
    """Helper function to create a dataclass field with metadata."""
    metadata = {"type": field_type}
    if default is not None:
        return field(default=default, init=init, metadata=metadata)
    elif default_factory is not None:
        return field(default_factory=default_factory, init=init, metadata=metadata)
    else:
        return field(init=init, metadata=metadata)

def serializable(default=None, default_factory=None, init=True):
    """Mark a field as serializable (saved to params.yml)."""
    return field_with_metadata("serializable",default, default_factory, init)

def weights(default=None, default_factory=None, init=False):
    """Mark a field as containing model weights (saved to model.pt)."""
    return field_with_metadata("weights", default, default_factory, init)

def unserializable(default=None, default_factory=None, init=True):
    """Mark a field as unserializable (saved to extra.pt)."""
    return field_with_metadata("unserializable", default, default_factory, init)


@dataclass
class DQNAgent(Agent, ABC):
    """Deep Q-Network (DQN) agent."""

    action_spec: TensorSpec = unserializable()

    _device: torch.device = unserializable(default_factory=lambda: torch.device("cpu"))

    # DQN parameters
    gamma: float = serializable(default=1)
    loss_function: str = serializable(default="l2")
    delay_value: bool = serializable(default=True)
    double_dqn: bool = serializable(default=False)

    # epsilon greedy parameters
    eps_annealing_num_batches: int = serializable(default=10000)
    eps_init: float = serializable(default=1.0)
    eps_end: float = serializable(default=0.1)
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
    replay_buffer_device: torch.device = unserializable(default_factory=lambda: torch.device("cpu"))

    # Set in constructor
    action_value_module: TensorDictModule = weights(init=False)
    greedy_module: QValueModule = field(init=False)
    greedy_policy_module: TensorDictModule = field(init=False)
    egreedy_module: EGreedyModule = field(init=False)
    egreedy_policy_module: TensorDictModule = field(init=False)
    loss_module: DQNLoss = field(init=False)
    target_net_updater: SoftUpdate = field(init=False)
    loss_keys: list[str] = field(init=False)
    optimizer: optim.Adam = field(init=False)
    replay_buffer: ReplayBuffer = field(init=False)

    # Getters to be defined in subclasses

    @abstractmethod
    def get_action_value_module(self) -> TensorDictModule:
        """Get the action value module."""
        pass

    def __post_init__(self) -> None:
        self.action_value_module = self.get_action_value_module().to(
            self._device
        )

        self.greedy_module = QValueModule(
            spec=self.action_spec,
            action_value_key="action_value",
            out_keys=["action", "action_value", "chosen_action_value"],
        ).to(self._device)

        self.greedy_policy_module = TensorDictSequential(
            [
                self.action_value_module,
                self.greedy_module
            ]
        )

        self.egreedy_module = EGreedyModule(
            spec=self.action_spec,
            annealing_num_steps=self.eps_annealing_num_batches,
            eps_init=self.eps_init,
            eps_end=self.eps_end,
        ).to(self._device)

        self.egreedy_policy_module = TensorDictSequential(
            [
                self.greedy_policy_module,
                self.egreedy_module
            ]
        )

        self.loss_module = DQNLoss(
            value_network=self.greedy_policy_module,
            loss_function=self.loss_function,
            delay_value=self.delay_value,
            double_dqn=self.double_dqn,
            action_space=self.action_spec,
        )
        self.loss_module.make_value_estimator(ValueEstimators.TD0, gamma=self.gamma)
        self.target_net_updater = SoftUpdate(self.loss_module, eps=1 - self.update_tau)
        self.loss_keys = ["loss"]
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
        return self.egreedy_policy_module

    def _anneal_replay_buffer_beta(self) -> None:
        # TODO
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

        # Anneal epsilon for epsilon greedy exploration
        self.egreedy_module.step()

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
            "eps": self.egreedy_module.eps.item(),
        }

    def get_eval_info(self) -> dict[str, Any]:
        """Get evaluation information."""
        return {}

    @property
    def device(self) -> torch.device:
        """Get the device of the agent."""
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self._device = device
        self.state_value_module = self.state_value_module.to(self._device)


    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        # Save weights
        weights_dict = {
            field.name: getattr(self, field.name).state_dict()
            for field in self.__dataclass_fields__.values()
            if field.metadata.get("type") == "weights"
        }
        torch.save(weights_dict, path / "model.pt")

        # Save unserializable fields
        unserializable_dict = {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
            if field.metadata.get("type") == "unserializable"
        }
        torch.save(unserializable_dict, path / "extra.pt")

        # Save serializable fields
        serializable_dict = {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
            if field.metadata.get("type") == "serializable"
        }
        with open(path / "params.yml", "w") as f:
            yaml.dump(serializable_dict, f)

    @classmethod
    def load(cls: type[T], path: Path) -> T:
        # Load serializable fields
        with open(path / "params.yml", "r") as f:
            serializable_dict = yaml.safe_load(f)

        # Load unserializable fields
        unserializable_dict = torch.load(path / "extra.pt", weights_only=False)

        # Merge all fields
        full_dict = {**serializable_dict, **unserializable_dict}

        # Construct agent
        agent = cls(**full_dict)

        # Load weights
        weights_dict = torch.load(path / "model.pt")
        for _field in agent.__dataclass_fields__.values():
            if _field.metadata.get("type") == "weights":
                getattr(agent, _field.name).load_state_dict(weights_dict[_field.name])

        return agent
