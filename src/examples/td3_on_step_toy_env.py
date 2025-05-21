from typing import Any
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from torchrl.modules import (
    MLP,
    NoisyLinear,
)
from torchrl.objectives import TD3Loss
from tensordict import TensorDictBase
from torchrl.objectives import SoftUpdate, ValueEstimators
from torchrl.data import TensorDictReplayBuffer
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import Tensor, nn, optim
import torch
from torchrl.data import LazyTensorStorage, TensorSpec
import wandb
from envs import StepToyEnv
from examples.training import train
from torchrl_agents import Agent, serializable, unserializable
from abc import ABC
from dataclasses import dataclass


class StepEmbeddingNet(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.step_embeddings = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=self.embedding_dim
        )

    def forward(self, step: Tensor) -> Tensor:
        # step: (batch_size, 1)
        step = step.squeeze(-1)  # (batch_size,)
        return self.step_embeddings(step)  # (batch_size, embedding_dim)

    def normalize_weights(self) -> None:
        self.step_embeddings.weight.data = F.normalize(
            self.step_embeddings.weight.data, dim=-1
        )


class StateValueNet(nn.Module):
    def __init__(self, step_embedding_dim: int):
        super().__init__()
        self.step_embedding_dim = step_embedding_dim
        self.net = MLP(self.step_embedding_dim, 1, num_cells=(32, 32))

    def forward(self, step_embedding: Tensor) -> Tensor:
        # step_embedding: (batch_size, embedding_dim)
        return self.net(step_embedding)  # (batch_size, 1)


class StateActionValueNet(nn.Module):
    def __init__(self, step_embedding_dim: int):
        super().__init__()
        self.step_embedding_dim = step_embedding_dim
        # self.net = MLP(self.step_embedding_dim + 1, 1, num_cells=(32, 32), norm_class=nn.LayerNorm, dropout=0.1)
        self.net = nn.Sequential(
            weight_norm(nn.Linear(self.step_embedding_dim + 1, 32), dim=None),
            # nn.LayerNorm(32),
            nn.Tanh(),
            nn.Dropout(0.1),
            weight_norm(nn.Linear(32, 32), dim=None),
            # nn.LayerNorm(32),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, step_embedding: Tensor, action: Tensor) -> Tensor:
        # step_embedding: (batch_size, embedding_dim)
        # action: (batch_size, 1)
        # step_embedding = F.normalize(step_embedding, dim=-1)
        x = torch.cat(
            (step_embedding, action), dim=-1
        )  # (batch_size, step_embedding_dim + 1)
        return self.net(x)  # (batch_size, 1)


class TempPolicyNet(nn.Module):
    def __init__(self, n_steps: int, init_exploration_std: float):
        super().__init__()
        self.n_steps = n_steps
        self.init_exploration_std = init_exploration_std
        # self.net = nn.Sequential(
        #     MLP(self.n_steps+1, 32, num_cells=(32), activate_last_layer=True),
        #     NoisyLinear(32, 1, std_init=self.init_exploration_std),
        # )
        self.net = MLP(
            self.n_steps + 1,
            1,
            num_cells=(32, 32),
            layer_class=NoisyLinear,
            layer_kwargs={"std_init": self.init_exploration_std},
        )
        # Initialize policy to have mean 0.5
        # self.net[-1].weight_mu.data.normal_(0.0, 1e-3)
        # self.net[-1].bias_mu.data.fill_(0.5)
        pass

    def forward(self, step: Tensor) -> Tensor:
        # step: (batch_size, 1)
        # New noise each time we take an action
        self.reset_noise()
        x = F.one_hot(
            step.squeeze(-1), num_classes=self.n_steps + 1
        ).float()  # (batch_size, n_steps+1)
        x = self.net(x)  # (batch_size, 1)
        # Actions are in the range [0, 1]
        x = F.sigmoid(x)  # (batch_size, 1)
        return x  # (batch_size, 1)

    def reset_noise(self) -> None:
        """Reset the noise in the policy network."""
        for module in self.net.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class PolicyNet(nn.Module):
    def __init__(self, step_embedding_dim: int, init_exploration_std):
        super().__init__()
        self.step_embedding_dim = step_embedding_dim
        self.init_exploration_std = init_exploration_std
        self.net = MLP(
            self.step_embedding_dim,
            1,
            num_cells=(32, 32),
            layer_class=NoisyLinear,
            layer_kwargs={"std_init": self.init_exploration_std},
        )

    def forward(self, step_embedding: Tensor) -> Tensor:
        # step_embedding: (batch_size, embedding_dim)
        # New noise each time we take an action
        self.reset_noise()
        x = self.net(step_embedding)  # (batch_size, 1)
        # Actions are in the range [0, 1]
        x = F.sigmoid(x)  # (batch_size, 1)
        return x

    def reset_noise(self) -> None:
        """Reset the noise in the policy network."""
        for module in self.net.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


@dataclass(kw_only=True, eq=False, order=False)
class StepToyEnvTD3Agent(Agent, ABC):
    action_spec: TensorSpec = unserializable()
    n_steps: int = serializable()
    step_embedding_dim: int = serializable(default=1)

    _device: torch.device = unserializable(default_factory=lambda: torch.device("cpu"))

    gamma: float = serializable()

    # Exploration parameters
    init_exploration_std: float = serializable(default=0.1)

    # TD3 parameters
    num_qvalue_nets: int = serializable(default=2)
    policy_noise: float = serializable(default=0.1)
    noise_clip: float = serializable(default=0.5)
    loss_function: str = serializable(default="l2")
    separate_losses: bool = serializable(default=False)

    # target network update rate
    update_tau: float = serializable(default=0.005)

    # Optimizer parameters
    backbone_net_lr: float = serializable(default=1e-3)
    state_action_value_net_lr: float = serializable(default=1e-3)
    policy_net_lr: float = serializable(default=1e-3)
    max_grad_norm: float = serializable(default=1)

    # Replay buffer parameters
    init_frames: int = serializable(default=0)
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
    # policy_module: TensorDictModuleBase = weights(init=False)
    # state_value_module: TensorDictModuleBase | None = weights(init=False)
    # loss_module: TD3Loss = field(init=False)
    # target_net_updater: SoftUpdate = field(init=False)
    # loss_keys: list[str] = field(init=False)
    # optimizer: optim.Adam = field(init=False)
    # replay_buffer: ReplayBuffer = field(init=False)

    def __post_init__(self) -> None:
        self.backbone_net = StepEmbeddingNet(
            num_embeddings=self.n_steps + 1, embedding_dim=self.step_embedding_dim
        )
        self.backbone = TensorDictModule(
            self.backbone_net,
            in_keys=["step"],
            out_keys=["step_embedding"],
        )

        # self.policy_net = PolicyNet(
        #     step_embedding_dim=self.step_embedding_dim,
        #     init_exploration_std=self.init_exploration_std,
        # )
        # self.policy_head = TensorDictModule(
        #     self.policy_net,
        #     in_keys=["step_embedding"],
        #     out_keys=["action"],
        # )
        # self.policy_module = TensorDictSequential(
        #     [self.backbone, self.policy_head],
        # )
        self.policy_net = TempPolicyNet(self.n_steps, init_exploration_std=self.init_exploration_std)
        self.policy_module = TensorDictModule( self.policy_net, in_keys=["step"], out_keys=["action"])

        self.state_action_value_net = StateActionValueNet(
            step_embedding_dim=self.step_embedding_dim
        )
        self.state_action_value_head = TensorDictModule(
            self.state_action_value_net,
            in_keys=["step_embedding", "action"],
            out_keys=["state_action_value"],
        )
        self.state_action_value_module = TensorDictSequential(
            [self.backbone, self.state_action_value_head],
        )
        # self.actor_critic = ActorCriticOperator(
        #     common_operator=self.backbone,
        #     policy_operator=self.policy_head,
        #     value_operator=self.state_action_value_head,
        # )

        self.loss_module = TD3Loss(
            # actor_network=self.actor_critic.get_policy_operator(),
            # qvalue_network=self.actor_critic.get_value_head(),
            actor_network=self.policy_module,
            qvalue_network=self.state_action_value_module,
            action_spec=self.action_spec,
            num_qvalue_nets=self.num_qvalue_nets,
            policy_noise=self.policy_noise,
            noise_clip=self.noise_clip,
            loss_function=self.loss_function,
            separate_losses=self.separate_losses,
        )
        self.loss_module.make_value_estimator(ValueEstimators.TD0, gamma=self.gamma)
        self.target_net_updater = SoftUpdate(self.loss_module, tau=self.update_tau)
        self.loss_keys = ["loss_qvalue"]
        if self.policy_net_lr > 0:
            self.loss_keys.append("loss_actor")

        # Optimizers for each network
        self.backbone_net_optimizer = optim.Adam(
            self.backbone_net_params.values(True, True), lr=self.backbone_net_lr
        )
        # def backbone_normalize_hook(optimizer, *args, **kwargs):
        #     self.loss_module.qvalue_network_params["module"]["0"]["module"]["step_embeddings"]["weight"].data = F.normalize(self.loss_module.qvalue_network_params["module"]["0"]["module"]["step_embeddings"]["weight"].data, dim=-1)
        # self.backbone_net_optimizer.register_step_post_hook(backbone_normalize_hook)
        self.state_action_value_net_optimizer = optim.Adam(
            self.state_action_value_net_params.values(True, True),
            lr=self.state_action_value_net_lr,
        )
        self.policy_net_optimizer = optim.Adam(
            self.policy_net_params.values(True, True), lr=self.policy_net_lr
        )
        # self.total_optimizer = optim.Adam(
        #     self.loss_module.parameters(), lr=1e-4
        # )

        self.optimizers = [
            self.backbone_net_optimizer,
            self.state_action_value_net_optimizer,
            self.policy_net_optimizer,
            # self.total_optimizer
        ]

        # Replay buffer
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
    def backbone_net_params(self):
        # return self.loss_module.actor_network_params["module"]["0"]
        return self.loss_module.qvalue_network_params["module"]["0"]

    @property
    def state_action_value_net_params(self):
        return self.loss_module.qvalue_network_params["module"]["1"]

    @property
    def target_state_action_value_net_params(self):
        return self.loss_module.target_qvalue_network_params["module"]["1"]

    @property
    def policy_net_params(self):
        # return self.loss_module.actor_network_params["module"]["1"]
        return self.loss_module.actor_network_params["module"]

    @property
    def policy(self) -> TensorDictModule:
        # return self.actor_critic.get_policy_operator()
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

        if len(self.replay_buffer) < self.init_frames:
            # If we don't have enough samples in the replay buffer, we just return 0.0 for all losses
            return {k: 0.0 for k in self.loss_keys}

        # Only continue if we have enough samples in the replay buffer, in which case we return 0.0 for all losses
        if len(self.replay_buffer) < self.init_random_frames:
            return {k: 0.0 for k in self.loss_keys}

        process_info_dict: dict[str, Any] = {k: 0.0 for k in self.loss_keys}

        for _ in range(self.num_samples):
            loss_td = self._sample_and_train()

            # Accumulate losses
            for k in self.loss_keys:
                process_info_dict[k] += loss_td[k].item()

            process_info_dict["td_error"] = loss_td["td_error"].mean().item()

        # Anneal beta for prioritized sampling
        self._anneal_replay_buffer_beta()

        # Compute average loss
        process_info_dict = {
            k: v / self.num_samples for k, v in process_info_dict.items()
        }

        with torch.no_grad():
            self.loss_module.qvalue_network(td)
            process_info_dict["Advantage?"] = td["state_action_value"].detach().cpu()

        process_info_dict["action"] = td["action"].detach().cpu()
        process_info_dict["step"] = td["step"].detach().cpu()

        return process_info_dict

    def _sample_and_train(self) -> TensorDictBase:
        """Sample from the replay buffer and train the policy, returning the loss td."""
        td = self.replay_buffer.sample()
        td = td.to(self.device)

        for optimizer in self.optimizers:
            optimizer.zero_grad()
        loss_td: TensorDictBase = self.loss_module(td)
        loss_tensor: Tensor = sum(
            (loss_td[k] for k in self.loss_keys), torch.tensor(0.0, device=td.device)
        )
        loss_tensor.backward()
        nn.utils.clip_grad_norm_(
            self.loss_module.parameters(), max_norm=self.max_grad_norm
        )
        for optimizer in self.optimizers:
            optimizer.step()

        # Update target network
        self.target_net_updater.step()

        # Update priorities in the replay buffer
        self.replay_buffer.update_tensordict_priority(td)  # type: ignore

        # Loss module also writes a td_error key which is useful for logging
        loss_td["td_error"] = td["td_error"]

        return loss_td

    def get_train_info(self) -> dict[str, Any]:
        return {
            "backbone_net_norm": sum(
                p.norm() for p in self.backbone_net_params.values(True, True)
            ),
            "state_action_value_net_norm": sum(
                p.norm() for p in self.state_action_value_net_params.values(True, True)
            ),
            "target_state_action_value_net_norm": sum(
                p.norm()
                for p in self.target_state_action_value_net_params.values(True, True)
            ),
            "policy_net_norm": sum(
                p.norm() for p in self.policy_net_params.values(True, True)
            ),
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

    def eval(self) -> None:
        """Set the agent to evaluation mode."""
        # self.actor_critic.eval()
        self.policy_module.eval()

    def train(self) -> None:
        """Set the agent to evaluation mode."""
        # self.actor_critic.train()
        self.policy_module.train()


def get_eval_metrics(td_evals: list[TensorDictBase]) -> dict[str, Any]:
    return {
        "reward sum": sum(td["next", "reward"].sum().item() for td in td_evals)
        / len(td_evals),
    }


def main() -> None:
    device = torch.device("cpu")
    batch_size = 64
    total_frames = 5000 * 64
    # total_frames = 10*64
    n_batches = total_frames // batch_size

    env = StepToyEnv(n_steps=5, batch_dim=1)
    env = env.to(device)

    agent: Agent = StepToyEnvTD3Agent(
        action_spec=env.action_spec,
        n_steps=5,
        step_embedding_dim=5,
        gamma=0.99,
        update_tau=0.01,
        state_action_value_net_lr=1e-2,
        backbone_net_lr=1e-3,
        policy_net_lr=0,
        replay_buffer_size=64,
        replay_buffer_beta_annealing_num_batches=total_frames,
        sub_batch_size=64,
        num_samples=1,
        num_qvalue_nets=2,  # TODO: 2 is the default
        loss_function="l2",
        policy_noise=0.05,
        noise_clip=0.1,
        init_exploration_std=1,
        separate_losses=False,
        init_frames=0,
    )

    # Manual debugging
    # td = env.reset()
    # td["action"] = torch.tensor([[0.5]])
    # td = env.step(td)

    collector = SyncDataCollector(
        env,  # type: ignore
        policy=agent.policy,
        frames_per_batch=batch_size,
        total_frames=total_frames,
    )

    run = wandb.init()

    eval_max_steps = 1000
    n_eval_episodes = 10

    # wandb.watch(agent.backbone_net_params, log="all", log_freq=1000, idx=0)
    # wandb.watch(agent.state_value_net_params, log="all", log_freq=1000, idx=1)
    # wandb.watch(agent.state_action_value_net_params, log="all", log_freq=1000, idx=2)
    # wandb.watch(agent.policy_net_params, log="all", log_freq=1000, idx=3)

    train(
        collector,
        env,
        agent,
        run,
        eval_every_n_batches=100,
        eval_max_steps=eval_max_steps,
        n_eval_episodes=n_eval_episodes,
        get_eval_metrics=get_eval_metrics,
        pixel_env=None,
    )

    # print("Saving agent...")
    # agent.save(Path("saved_models/temp"))

    # # Load the agent from the saved model and see if it still performs well
    # del agent
    # print("Loading agent...")
    # agent = Agent.load(Path("saved_models/temp"))

    # with (
    #     torch.no_grad(),
    #     set_exploration_type(ExplorationType.DETERMINISTIC),
    # ):
    #     td_evals = [
    #         env.rollout(eval_max_steps, agent.policy)
    #         for _ in tqdm(range(n_eval_episodes), desc="Evaluating")
    #     ]
    # metrics_eval = get_eval_metrics(td_evals)

    # run.log(
    #     {f"final/{k}": v for k, v in (metrics_eval | agent.get_eval_info()).items()},
    # )

    run.finish()


if __name__ == "__main__":
    main()
