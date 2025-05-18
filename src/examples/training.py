from typing import Any
from collections.abc import Callable
from tensordict import TensorDictBase
import torch
from torchrl.collectors import SyncDataCollector
from torchrl.envs import EnvBase, ExplorationType, set_exploration_type
from tqdm import tqdm
import wandb
from wandb.wandb_run import Run

from torchrl_agents import Agent


def train(
    train_collector: SyncDataCollector,
    # train_env: EnvBase,
    eval_env: EnvBase,
    agent: Agent,
    run: Run,
    eval_every_n_batches: int,
    eval_max_steps: int,
    n_eval_episodes: int,
    get_eval_metrics: Callable[[list[TensorDictBase]], dict[str, Any]],
    pixel_env: EnvBase | None = None,
) -> None:
    try:
        for batch_idx, td_train in enumerate(tqdm(train_collector)):
            train_collector.update_policy_weights_()
            loss_info = agent.process_batch(td_train)

            # Log training info
            run.log(
                {
                    f"train/{k}": v
                    for k, v in (loss_info | agent.get_train_info()).items()
                },
            )

            # Evaluation every now and then
            if batch_idx > 0 and batch_idx % eval_every_n_batches == 0:
                with (
                    torch.no_grad(),
                    set_exploration_type(ExplorationType.DETERMINISTIC),
                ):
                    td_evals = [
                        eval_env.rollout(eval_max_steps, agent.policy)
                        for _ in tqdm(range(n_eval_episodes), desc="Evaluating")
                    ]
                    # A single episode in the visual env
                    if pixel_env is not None:
                        td_pixel = pixel_env.rollout(
                            eval_max_steps,
                            agent.policy,
                        )
                        pixel_dict = {
                            "eval/video": wandb.Video(
                                td_pixel["pixels"].permute(0, 3, 1, 2).cpu().numpy()
                            )
                        }
                    else:
                        pixel_dict = {}
                metrics_eval = get_eval_metrics(td_evals)

                run.log(
                    {
                        f"eval/{k}": v
                        for k, v in (metrics_eval | agent.get_eval_info()).items()
                    }
                    | pixel_dict
                )

    except KeyboardInterrupt:
        print("Training interrupted.")
