# TorchRL Agents

TorchRL Agents is a Python package that provides object-oriented reinforcement learning (RL) agents built on top of PyTorch and TorchRL.

## Object-Oriented Design

At the core of this package is the `Agent` abstract base class, which defines a structured interface for RL agents. The `Agent` class provides a blueprint that other classes can subclass to implement specific RL algorithms. This design ensures consistency and reusability across different agents.

### Key Features of the `Agent` Class:
- **Abstract Methods**: Enforces the implementation of essential functionalities, such as processing batches and defining policies.
- **Serialization**: Supports saving and loading agent configurations and weights, enabling easy training and deployment.
- **Modularity**: Allows for easy extension and customization by subclassing.

By subclassing the `Agent` class, you can implement various RL algorithms while adhering to a consistent structure.

## Examples

The `examples` folder contains practical demonstrations of how to use the agents implemented in this package. These examples showcase:
- Training RL agents on benchmark environments such as CartPole and Pendulum.
- Using the modular components of the `Agent` class to customize behavior.
- Saving and loading trained agents for evaluation or deployment.


## Getting Started

1. Install [uv](https://docs.astral.sh/uv/)

2. Clone the repository and create a virtual environment:
   ```bash
   git clone https://github.com/valterschutz/torchrl-examples.git
   uv sync
   ```

3. Explore the `examples` folder to see how to train and evaluate RL agents. The scripts can be run without any arguments.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
