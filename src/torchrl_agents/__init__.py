from abc import ABC, abstractmethod
from dataclasses import MISSING, field
from pathlib import Path
import torch
from tensordict.nn import TensorDictModuleBase
from typing import Any, ClassVar, TypeVar

from tensordict import TensorDictBase
import yaml

T = TypeVar("T", bound="Agent")


def field_with_metadata(
    field_type: str, default=MISSING, default_factory=MISSING, init: bool = True
):
    """Create a dataclass field with metadata."""
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("Cannot specify both default and default_factory")
    metadata = {"type": field_type}
    # If default is provided, prefer it over default_factory
    if default is not MISSING:
        return field(default=default, init=init, metadata=metadata)
    elif default_factory is not MISSING:
        return field(default_factory=default_factory, init=init, metadata=metadata)
    else:
        return field(init=init, metadata=metadata)


def serializable(default=MISSING, default_factory=MISSING, init: bool = True):
    """Mark a field as serializable (saved to params.yml)."""
    return field_with_metadata("serializable", default, default_factory, init)


def weights(default=MISSING, default_factory=MISSING, init: bool = False):
    """Mark a field as containing model weights (saved to model.pt)."""
    return field_with_metadata("weights", default, default_factory, init)


def unserializable(default=MISSING, default_factory=MISSING, init: bool = True):
    """Mark a field as unserializable (saved to extra.pt)."""
    return field_with_metadata("unserializable", default, default_factory, init)


class Agent(ABC):
    # Subclasses will be dataclasses
    __dataclass_fields__: ClassVar[dict[str, Any]]

    @abstractmethod
    def process_batch(self, td: TensorDictBase) -> dict[str, float]: ...

    def get_train_info(self) -> dict[str, Any]:
        return {}

    def get_eval_info(self) -> dict[str, Any]:
        return {}

    @property
    @abstractmethod
    def policy(self) -> TensorDictModuleBase: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @device.setter
    @abstractmethod
    def device(self, device: torch.device) -> None:
        """Move agent networks to a specific device."""
        ...

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        # Save the class name
        with open(path / "class_name.txt", "w") as f:
            f.write(self.__class__.__name__)

        # Save weights to model.pt
        weights_dict = {
            field.name: getattr(self, field.name).state_dict()
            if getattr(self, field.name) is not None
            else None
            for field in self.__dataclass_fields__.values()
            if field.metadata.get("type") == "weights"
        }
        torch.save(weights_dict, path / "model.pt")

        # Save unserializable fields to extra.pt
        unserializable_dict = {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
            if field.metadata.get("type") == "unserializable"
        }
        torch.save(unserializable_dict, path / "extra.pt")

        # Save serializable fields to params.yml
        serializable_dict = {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
            if field.metadata.get("type") == "serializable"
        }
        with open(path / "params.yml", "w") as f:
            yaml.dump(serializable_dict, f)

    @classmethod
    def load(cls: type[T], path: Path) -> T:
        # Load the class name
        with open(path / "class_name.txt") as f:
            class_name = f.read().strip()

        # Find the subclass, even if it's several layers down
        def find_subclass(base_class: type, name: str) -> type | None:
            for subclass in base_class.__subclasses__():
                if subclass.__name__ == name:
                    return subclass
                found = find_subclass(subclass, name)
                if found:
                    return found
            return None

        subclass = find_subclass(cls, class_name)
        if subclass is None:
            raise ValueError(f"Unknown subclass: {class_name}")

        # Load serializable fields
        with open(path / "params.yml") as f:
            serializable_dict = yaml.safe_load(f)

        # Load unserializable fields
        unserializable_dict = torch.load(path / "extra.pt", weights_only=False)

        # Merge all fields
        full_dict = {**serializable_dict, **unserializable_dict}

        # Construct agent
        agent = subclass(**full_dict)

        # Load weights
        weights_dict = torch.load(path / "model.pt")
        for _field in agent.__dataclass_fields__.values():
            if _field.metadata.get("type") == "weights":
                weights = weights_dict.get(_field.name)
                model = getattr(agent, _field.name)
                if weights is not None:
                    model.load_state_dict(weights)

        return agent
