"""Train a DDPG agent on the pendulum 4 environment."""

from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
import wandb
from tensordict import TensorDictBase
from torchrl.collectors import SyncDataCollector

from torchrl_examples.agents.ddpg import DDPGAgent

from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.envs import (
    Compose,
    ExplorationType,
    GymEnv,
    TransformedEnv,
    set_exploration_type,
)
from torchrl_examples.training import train


class PendulumV1PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        num_cells = 256
        self.net = nn.Sequential(
            nn.Linear(3, num_cells),
            nn.Tanh(),
            nn.Linear(num_cells, num_cells),
            nn.Tanh(),
            nn.Linear(num_cells, 1),
            nn.Tanh(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # Pass through the network and scale
        return 2 * self.net(observation)


class PendulumV1ActionValueNet(nn.Module):
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
        # Pass through the network and scale
        return self.net(torch.cat([observation, action], dim=-1))


class PendulumV1DDPGAgent(DDPGAgent):
    def get_policy_module(self) -> TensorDictModule:
        return TensorDictModule(
            # CartpoleV1QNet(),
            module=PendulumV1PolicyNet(),
            in_keys=["observation"],
            out_keys=["action"],
        )

    def get_state_action_value_module(self) -> TensorDictModule:
        return TensorDictModule(
            module=PendulumV1ActionValueNet(),
            in_keys=["observation", "action"],
            out_keys=["state_action_value"],
        )


def get_eval_metrics(td_evals: list[TensorDictBase]) -> dict[str, Any]:
    """Get evaluation metrics from a list of TensorDicts."""
    metrics = {}
    metrics["reward_sum"] = 0.0
    for td in td_evals:
        metrics["reward_sum"] += td["next", "reward"].sum().item()
    metrics["reward_sum"] /= len(td_evals)
    return metrics


def main() -> None:
    batch_size = 100
    # Define the total number of frames for training
    total_frames = int(1e6)  # 1 million frames
    n_batches = total_frames // batch_size

    env = TransformedEnv(
        GymEnv("Pendulum-v1"),
        Compose(),
    )
    env = env.to(torch.device("cuda:1"))

    pixel_env = None

    agent = PendulumV1DDPGAgent(
        action_spec=env.action_spec,
        _device=torch.device("cuda:1"),
        gamma=0.99,
        update_tau=0.005,
        lr=1e-3,
        max_grad_norm=1,
        replay_buffer_size=1000000,
        sub_batch_size=100,
        num_samples=10,
        replay_buffer_device=torch.device("cuda:1"),
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

    run = wandb.init(
        entity="valterschutz-chalmers-university-of-technology",
        project="torchrl-examples-ddpg-pendulum-v1",
    )

    eval_max_steps = 1000
    n_eval_episodes = 100
    train(
        collector,
        env,
        agent,
        run,
        eval_every_n_batches=100,
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
    agent = PendulumV1DDPGAgent.load(Path("saved_models/temp"))

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
