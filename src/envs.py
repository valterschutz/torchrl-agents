from tensordict import TensorDict, TensorDictBase
import torch
from torchrl.data import Bounded, Composite
from torchrl.envs import EnvBase


class StepToyEnv(EnvBase):
    def __init__(self, n_steps: int, batch_dim: int = 1):
        super().__init__(batch_size=torch.Size((batch_dim,)))
        self.n_steps = n_steps

        self._make_spec()

    def _make_spec(self):
        self.observation_spec = Composite(
            step=Bounded(
                low=0,
                high=self.n_steps,
                shape=self.batch_size + torch.Size((1,)),
                dtype=torch.int64,
            ),
            batch_size=self.batch_size,
        )
        self.action_spec = Bounded(
            low=0.0,
            high=1.0,
            shape=self.batch_size + torch.Size((1,)),
            dtype=torch.float32,
        )
        self.reward_spec = Bounded(
            low=0.0,
            high=1.0,
            shape=self.batch_size + torch.Size((1,)),
            dtype=torch.float32,
        )

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        if tensordict is None:
            tensordict = TensorDict({}, batch_size=self.batch_size, device=self.device)
        return TensorDict(
            {
                "step": torch.zeros(
                    tensordict.batch_size + torch.Size((1,)),
                    device=tensordict.device,
                    dtype=torch.int64,
                ),
            },
            batch_size=tensordict.batch_size,
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Always increment the step
        next_step = tensordict["step"] + 1

        # Reward is action deviation from 0.(next_step)
        reward = 1 - torch.abs(tensordict["action"] - (next_step / 10))

        return TensorDict(
            {
                "step": next_step,
                "reward": reward,
                "done": next_step >= self.n_steps,
            },
            batch_size=tensordict.batch_size,
            device=tensordict.device,
        )

    def _set_seed(self, seed: int | None) -> None:
        rng = torch.manual_seed(seed)
        self.rng = rng
