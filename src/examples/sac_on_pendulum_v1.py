"""Train a SAC agent on the Cartpole v1 environment."""

from pathlib import Path
from typing import Any

import torch
from torchrl.modules import ProbabilisticActor, TanhNormal
from tqdm import tqdm
import wandb
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictModuleBase
from torchrl.collectors import SyncDataCollector

from torchrl_agents import Agent
from torchrl_agents.sac import SACAgent

from tensordict.nn import NormalParamExtractor
from torch import nn
from torchrl.envs import (
    Compose,
    ExplorationType,
    GymEnv,
    TransformedEnv,
    set_exploration_type,
)
from examples.training import train


class PendulumV1StateActionValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        num_cells = 256
        self.net = nn.Sequential(
            nn.Linear(3 + 1, num_cells),
            nn.ReLU(),
            nn.Linear(num_cells, num_cells),
            nn.ReLU(),
            nn.Linear(num_cells, 1),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute the state-action value."""
        # Concatenate the observation and action
        x = torch.cat([observation, action], dim=-1)
        # Pass through the network
        return self.net(x)


class PendulumV1StateValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        num_cells = 256
        self.net = nn.Sequential(
            nn.Linear(3, num_cells),
            nn.ReLU(),
            nn.Linear(num_cells, num_cells),
            nn.ReLU(),
            nn.Linear(num_cells, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Compute the state value."""
        return self.net(observation)


class PendulumV1PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        num_cells = 256
        self.net = nn.Sequential(
            nn.Linear(3, num_cells),
            nn.ReLU(),
            nn.Linear(num_cells, num_cells),
            nn.ReLU(),
            nn.Linear(num_cells, 2),
            NormalParamExtractor(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # Pass through the network
        return self.net(observation)


class PendulumV1SACAgent(SACAgent):
    def get_state_action_value_module(self) -> TensorDictModuleBase:
        self.state_action_value_net = PendulumV1StateActionValueNet()
        return TensorDictModule(
            module=self.state_action_value_net,
            in_keys=["observation", "action"],
            out_keys=["state_action_value"],
        )

    def get_policy_module(self) -> TensorDictModuleBase:
        self.policy_net = PendulumV1PolicyNet()
        policy_module = TensorDictModule(
            self.policy_net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        policy_module = ProbabilisticActor(
            module=policy_module,
            spec=self.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={"low": -2.0, "high": 2.0},
            return_log_prob=True,
        )

        return policy_module

    def get_state_value_module(self) -> TensorDictModule:
        self.state_value_net = PendulumV1StateValueNet()
        return TensorDictModule(
            self.state_value_net,
            in_keys=["observation"],
            out_keys=["state_value"],
        )

    def get_eval_info(self) -> dict[str, Any]:
        """Get evaluation info."""
        return {
            "replay buffer stored elems": len(self.replay_buffer),
        }


def get_eval_metrics(td_evals: list[TensorDictBase]) -> dict[str, Any]:
    """Get evaluation metrics from a list of TensorDicts."""
    metrics = {}
    metrics["reward_sum"] = 0.0
    for td in td_evals:
        metrics["reward_sum"] += td["next", "reward"].sum().item()
    metrics["reward_sum"] /= len(td_evals)
    return metrics


def main() -> None:
    device = torch.device("cuda:0")
    batch_size = 64
    total_frames = 1000000
    n_batches = total_frames // batch_size

    env = TransformedEnv(
        GymEnv("Pendulum-v1"),
        Compose(),
    )
    env = env.to(device)

    pixel_env = GymEnv("Pendulum-v1", from_pixels=True, pixels_only=False)
    pixel_env = pixel_env.to(device)

    agent: Agent = PendulumV1SACAgent(
        action_spec=env.action_spec,
        _device=device,
        gamma=0.99,
        # TODO: SAC kwargs
        update_tau=0.005,
        lr=1e-3,
        max_grad_norm=1,
        replay_buffer_size=10000,
        sub_batch_size=100,
        num_samples=10,
        replay_buffer_device=device,
        replay_buffer_alpha=0.6,
        replay_buffer_beta_init=0.4,
        replay_buffer_beta_end=1,
        replay_buffer_beta_annealing_num_batches=n_batches,
        init_random_frames=1000,
    )

    collector = SyncDataCollector(
        env,  # type: ignore
        policy=agent.policy,
        frames_per_batch=batch_size,
        total_frames=total_frames,
    )

    run = wandb.init()

    eval_max_steps = 1000
    n_eval_episodes = 100

    wandb.watch(agent.policy_net, log="all", log_freq=1000)
    wandb.watch(agent.state_value_net, log="all", log_freq=1000)
    wandb.watch(agent.state_action_value_net, log="all", log_freq=1000)

    train(
        collector,
        env,
        agent,
        run,
        eval_every_n_batches=200,
        eval_max_steps=eval_max_steps,
        n_eval_episodes=n_eval_episodes,
        get_eval_metrics=get_eval_metrics,
        pixel_env=pixel_env,
    )

    print("Saving agent...")
    agent.save(Path("saved_models/temp"))

    # Load the agent from the saved model and see if it still performs well
    del agent
    print("Loading agent...")
    agent = Agent.load(Path("saved_models/temp"))

    with (
        torch.no_grad(),
        set_exploration_type(ExplorationType.DETERMINISTIC),
    ):
        td_evals = [
            env.rollout(eval_max_steps, agent.policy)
            for _ in tqdm(range(n_eval_episodes), desc="Evaluating")
        ]
    metrics_eval = get_eval_metrics(td_evals)

    run.log(
        {f"final/{k}": v for k, v in (metrics_eval | agent.get_eval_info()).items()},
    )

    run.finish()


if __name__ == "__main__":
    main()
