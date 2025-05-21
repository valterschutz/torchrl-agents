import torch
from torchrl.envs import check_env_specs
from envs import StepToyEnv


def test_step_toy_env():
    """Test the StepToyEnv class."""
    env = StepToyEnv(n_steps=2)
    check_env_specs(env)

    td = env.reset()
    assert td["step"].item() == 0

    td["action"] = torch.tensor([[0.5]])
    td = env.step(td)
    assert td["next", "step"].item() == 1
    assert torch.allclose(td["next", "reward"], torch.tensor([1.0 - 0.4]))
    td = td["next"]

    td["action"] = torch.tensor([[0.3]])
    td = env.step(td)
    assert td["next", "step"].item() == 2
    assert torch.allclose(td["next", "reward"], torch.tensor([1.0 - 0.1]))
    assert td["next", "done"].item() is True
