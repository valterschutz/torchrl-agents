
from pathlib import Path
import torch
from tensordict.nn import TensorDictModuleBase
from typing import Any, Protocol, TypeVar

from tensordict import TensorDictBase

T = TypeVar("T", bound="Agent")

class Agent(Protocol):
    def process_batch(self, td: TensorDictBase) -> dict[str, float]: ...

    def get_train_info(self) -> dict[str, Any]: ...

    def get_eval_info(self) -> dict[str, Any]: ...

    @property
    def policy(self) -> TensorDictModuleBase: ...

    @property
    def device(self) -> torch.device: ...

    @device.setter
    def device(self, device: torch.device) -> None:
        """Move agent networks to a specific device."""
        ...

    def save(self, path: Path) -> None:
        """Save the agent to a file."""
        ...

    @classmethod
    def load(cls: type[T], path: Path) -> T:
        """Load the agent from a file."""
        ...
