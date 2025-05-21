from typing import Any
from torch.nn import functional as F
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from torchrl.modules import (
    MLP,
    NormalParamExtractor,
    ProbabilisticActor,
    TruncatedNormal,
)
from torchrl.objectives import SACLoss
from tensordict import TensorDictBase
from torchrl.objectives import SoftUpdate, ValueEstimators
from torchrl.data import TensorDictReplayBuffer
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from torch import Tensor, nn, optim
import torch
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorSpec
import wandb
from envs import StepToyEnv
from examples.training import train
from torchrl_agents import Agent, serializable, unserializable, weights
from abc import ABC
from dataclasses import dataclass, field


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

    # def normalize_weights(self) -> None:
    #     self.step_embeddings.weight.data = F.normalize(self.step_embeddings.weight.data, dim=-1)


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
        self.net = MLP(self.step_embedding_dim + 1, 1, num_cells=(32, 32))

    def forward(self, step_embedding: Tensor, action: Tensor) -> Tensor:
        # step_embedding: (batch_size, embedding_dim)
        # action: (batch_size, 1)
        x = torch.cat(
            (step_embedding, action), dim=-1
        )  # (batch_size, step_embedding_dim + 1)
        return self.net(x)  # (batch_size, 1)


class TempPolicyNet(nn.Module):
    def __init__(self, n_steps: int):
        super().__init__()
        self.n_steps = n_steps
        mlp = MLP(self.n_steps + 1, 2, num_cells=(32, 32))
        mlp[-1].weight.data.normal_(0.0, 0.001)
        mlp[-1].bias[0].data.fill_(0.4)
        mlp[-1].bias[1].data.fill_(-2.0)
        normal_param_extractor = NormalParamExtractor(
            scale_mapping="softplus", scale_lb=1e-6
        )
        self.net = nn.Sequential(
            mlp,
            normal_param_extractor,
        )
        pass

    def forward(self, step: Tensor) -> tuple[Tensor, Tensor]:
        # step: (batch_size, 1)
        x = F.one_hot(
            step.squeeze(-1), num_classes=self.n_steps + 1
        ).float()  # (batch_size, n_steps+1)
        return self.net(x)  # (batch_size, 1), (batch_size, 1)


class PolicyNet(nn.Module):
    def __init__(self, step_embedding_dim: int):
        super().__init__()
        self.step_embedding_dim = step_embedding_dim
        mlp = MLP(self.step_embedding_dim, 2, num_cells=(32, 32))
        mlp[-1].weight.data.normal_(0.0, 0.001)
        mlp[-1].bias[0].data.fill_(0.4)
        mlp[-1].bias[1].data.fill_(-4.0)
        normal_param_extractor = NormalParamExtractor(
            scale_mapping="softplus", scale_lb=1e-6
        )
        self.net = nn.Sequential(
            mlp,
            normal_param_extractor,
        )
        pass

    def forward(self, step_embedding: Tensor) -> tuple[Tensor, Tensor]:
        # step_embedding: (batch_size, embedding_dim)
        return self.net(step_embedding)  # (batch_size, 1), (batch_size, 1)


@dataclass(kw_only=True, eq=False, order=False)
class StepToyEnvSACAgent(Agent, ABC):
    action_spec: TensorSpec = unserializable()
    n_steps: int = serializable()
    step_embedding_dim: int = serializable(default=1)

    _device: torch.device = unserializable(default_factory=lambda: torch.device("cpu"))

    gamma: float = serializable()

    # SAC parameters
    use_entropy: bool = serializable(default=True)
    num_qvalue_nets: int = serializable(default=2)
    loss_function: str = serializable(default="smooth_l1")
    alpha_init: float = serializable(default=1.0)
    min_alpha: float | None = serializable(default=None)
    max_alpha: float | None = serializable(default=None)
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
    backbone_net_lr: float = serializable(default=1e-3)
    state_value_net_lr: float = serializable(default=1e-3)
    state_action_value_net_lr: float = serializable(default=1e-3)
    policy_net_lr: float = serializable(default=1e-3)
    alpha_lr: float = serializable(default=1e-3)
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
    policy_module: TensorDictModuleBase = weights(init=False)
    state_value_module: TensorDictModuleBase | None = weights(init=False)
    state_action_value_module: TensorDictModuleBase = weights(init=False)
    loss_module: SACLoss = field(init=False)
    target_net_updater: SoftUpdate = field(init=False)
    loss_keys: list[str] = field(init=False)
    optimizer: optim.Adam = field(init=False)
    replay_buffer: ReplayBuffer = field(init=False)

    def __post_init__(self) -> None:
        self.backbone_net = StepEmbeddingNet(
            num_embeddings=self.n_steps + 1, embedding_dim=self.step_embedding_dim
        )
        self.backbone = TensorDictModule(
            self.backbone_net,
            in_keys=["step"],
            out_keys=["step_embedding"],
        )

        # self.policy_net = PolicyNet(step_embedding_dim=self.step_embedding_dim)
        # self.policy_head = TensorDictModule(
        #     self.policy_net,
        #     in_keys=["step_embedding"],
        #     out_keys=["loc", "scale"],
        # )
        self.policy_net = TempPolicyNet(self.n_steps)
        self.policy_module = ProbabilisticActor(
            # TensorDictSequential([self.backbone, self.policy_head]),
            TensorDictModule(
                self.policy_net,
                in_keys=["step"],
                out_keys=["loc", "scale"],
            ),
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TruncatedNormal,
            distribution_kwargs={
                "low": 0.0,
                "high": 1.0,
                "tanh_loc": False,
                "upscale": 1.0,
            },
            return_log_prob=True,  # TODO ?
        )

        self.state_value_net = StateValueNet(step_embedding_dim=self.step_embedding_dim)
        self.state_value_head = TensorDictModule(
            self.state_value_net,
            in_keys=["step_embedding"],
            out_keys=["state_value"],
        )
        self.state_value_module = TensorDictSequential(
            [self.backbone, self.state_value_head],
        )

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
            separate_losses=self.separate_losses,
            reduction=self.reduction,
            skip_done_states=self.skip_done_states,
        )
        self.loss_module.make_value_estimator(ValueEstimators.TD0, gamma=self.gamma)
        self.target_net_updater = SoftUpdate(self.loss_module, tau=self.update_tau)
        self.loss_keys = ["loss_actor", "loss_qvalue", "loss_value"]
        if self.use_entropy:
            self.loss_keys.append("loss_alpha")

        # Optimizers for each network
        self.backbone_net_optimizer = optim.Adam(
            self.backbone_net_params.values(True, True), lr=self.backbone_net_lr
        )
        self.state_value_net_optimizer = optim.Adam(
            self.state_value_net_params.values(True, True), lr=self.state_value_net_lr
        )
        self.state_action_value_net_optimizer = optim.Adam(
            self.state_action_value_net_params.values(True, True),
            lr=self.state_action_value_net_lr,
        )
        self.policy_net_optimizer = optim.Adam(
            self.policy_net_params.values(True, True), lr=self.policy_net_lr
        )
        self.alpha_optimizer = optim.Adam(
            iter([self.loss_module.log_alpha]), lr=self.alpha_lr
        )

        self.optimizers = [
            self.backbone_net_optimizer,
            self.state_value_net_optimizer,
            self.state_action_value_net_optimizer,
            self.policy_net_optimizer,
            self.alpha_optimizer,
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
        return self.loss_module.value_network_params["module"]["0"]

    @property
    def target_backbone_net_params(self):
        return self.loss_module.target_value_network_params["module"]["0"]

    @property
    def state_value_net_params(self):
        return self.loss_module.value_network_params["module"]["1"]

    @property
    def target_state_value_net_params(self):
        return self.loss_module.target_value_network_params["module"]["1"]

    @property
    def state_action_value_net_params(self):
        return self.loss_module.qvalue_network_params["module"]["1"]

    @property
    def target_state_action_value_net_params(self):
        return self.loss_module.target_qvalue_network_params["module"]["1"]

    # @property
    # def policy_net_params(self):
    #     return self.loss_module.actor_network_params["module"]["0"]["module"][
    #         "1"
    #     ]
    @property
    def policy_net_params(self):
        return self.loss_module.actor_network_params

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

        process_info_dict: dict[str, Any] = {k: 0.0 for k in self.loss_keys}

        for _ in range(self.num_samples):
            loss_td = self._sample_and_train()

            # Accumulate losses
            for k in self.loss_keys:
                process_info_dict[k] += loss_td[k].item()

            process_info_dict["td_error"] = loss_td["td_error"].mean().item()
            process_info_dict["alpha"] = loss_td["alpha"].mean().item()
            process_info_dict["entropy"] = loss_td["entropy"].mean().item()

        # Anneal beta for prioritized sampling
        self._anneal_replay_buffer_beta()

        # Compute average loss
        process_info_dict = {
            k: v / self.num_samples for k, v in process_info_dict.items()
        }

        with torch.no_grad():
            self.loss_module.value_network(td)
            process_info_dict["State values"] = td["state_value"].detach().cpu()
            # Also return Q-values, useful for debugging
            self.loss_module.qvalue_network(td)
            process_info_dict["Advantage?"] = td["state_action_value"].detach().cpu()

        process_info_dict["loc"] = td["loc"].detach().cpu()
        process_info_dict["scale"] = td["scale"].detach().cpu()
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

        # SAC loss module also writes a td_error key which is usefull for logging
        loss_td["td_error"] = td["td_error"]

        return loss_td

    def get_train_info(self) -> dict[str, Any]:
        return {
            "backbone_net_norm": sum(
                p.norm() for p in self.backbone_net_params.values(True, True)
            ),
            "target_backbone_net_norm": sum(
                p.norm() for p in self.target_backbone_net_params.values(True, True)
            ),
            "state_value_net_norm": sum(
                p.norm() for p in self.state_value_net_params.values(True, True)
            ),
            "target_state_value_net_norm": sum(
                p.norm() for p in self.target_state_value_net_params.values(True, True)
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


def get_eval_metrics(td_evals: list[TensorDictBase]) -> dict[str, Any]:
    return {}


def main() -> None:
    device = torch.device("cpu")
    batch_size = 64
    total_frames = 10000 * 64
    # total_frames = 10*64
    n_batches = total_frames // batch_size

    env = StepToyEnv(n_steps=5, batch_dim=1)
    env = env.to(device)

    agent: Agent = StepToyEnvSACAgent(
        action_spec=env.action_spec,
        n_steps=5,
        step_embedding_dim=5,
        gamma=0.99,
        update_tau=0.005,
        state_action_value_net_lr=1e-3,
        state_value_net_lr=1e-3,
        backbone_net_lr=1e-3,
        policy_net_lr=1e-4,
        alpha_lr=1e-4,
        replay_buffer_size=10000,
        sub_batch_size=64,
        num_samples=5,
        num_qvalue_nets=1,  # TODO: 2 is the default
        use_entropy=True,
        alpha_init=1.0,
        loss_function="l2",
        target_entropy=0.0,
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
        eval_every_n_batches=9001,
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
