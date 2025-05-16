"""Train a PPO agent on the Inverted Double Pendulum v4 environment."""

from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
import wandb
from tensordict import TensorDictBase
from torchrl.collectors import SyncDataCollector

from torchrl_agents import Agent
from torchrl_agents.ppo import PPOAgent

from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ExplorationType,
    GymEnv,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
    set_exploration_type,
)
from torchrl.modules import NormalParamExtractor, ProbabilisticActor, TanhNormal
from torchrl_examples.training import train


class InvertedDoublePendulumV4PPOAgent(PPOAgent):
    def get_policy_module(self) -> ProbabilisticActor:
        num_cells = 256

        return ProbabilisticActor(
            TensorDictModule(
                nn.Sequential(
                    nn.Linear(11, num_cells),
                    nn.Tanh(),
                    nn.Linear(num_cells, num_cells),
                    nn.Tanh(),
                    nn.Linear(num_cells, num_cells),
                    nn.Tanh(),
                    nn.Linear(num_cells, 2),
                    NormalParamExtractor(),
                ),
                in_keys=["observation"],
                out_keys=["loc", "scale"],
            ),
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TanhNormal,
            distribution_kwargs={"low": -1.0, "high": 1.0},
            return_log_prob=True,
        )

    def get_state_value_module(self) -> TensorDictModule:
        num_cells = 256

        return TensorDictModule(
            nn.Sequential(
                nn.Linear(11, num_cells),
                nn.Tanh(),
                nn.Linear(num_cells, num_cells),
                nn.Tanh(),
                nn.Linear(num_cells, num_cells),
                nn.Tanh(),
                nn.Linear(num_cells, 1),
            ),
            in_keys=["observation"],
            out_keys=["state_value"],
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
    device = torch.device("cuda:0")
    batch_size = 1000

    env = TransformedEnv(
        GymEnv("InvertedDoublePendulum-v4"),
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )
    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)  # type: ignore
    env = env.to(device)

    pixel_env = TransformedEnv(
        GymEnv("InvertedDoublePendulum-v4", from_pixels=True, pixels_only=False),
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )
    pixel_env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)  # type: ignore
    pixel_env = pixel_env.to(device)

    agent: Agent = InvertedDoublePendulumV4PPOAgent(
        _device=device,
        batch_size=batch_size,
        sub_batch_size=100,
        num_epochs=10,
        gamma=0.99,
        lmbda=0.95,
        clip_epsilon=0.2,
        entropy_bonus=True,
        entropy_coef=1e-4,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
        lr=3e-4,
        max_grad_norm=1.0,
        replay_buffer_device=device,
    )

    collector = SyncDataCollector(
        env,  # type: ignore
        policy=agent.policy,
        frames_per_batch=batch_size,
        total_frames=1000000,
    )

    run = wandb.init()

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
