# How This Project Works

This project is a PyTorch implementation of the **DreamerV3 algorithm**, a reinforcement learning agent designed to learn effectively in a wide variety of complex environments. It achieves this by learning a "world model" – an internal simulation of the environment – which allows it to learn from imagined experiences.

## Core Components

The DreamerV3 agent is primarily composed of three interconnected parts:

1.  **The World Model**:
    *   **Purpose**: To understand the environment's dynamics. It learns from the agent's past experiences.
    *   **Functionality**:
        *   **Encoder**: Takes raw sensory inputs (like game screen images or robot joint angles) and compresses them into a compact numerical representation called a "latent state."
        *   **Dynamics Model (RSSM - Recurrent State-Space Model)**: Given the current latent state and an action taken by the agent, it predicts the next latent state. This forms the basis of the agent's ability to "see" into the future.
        *   **Prediction Heads**: Attached to the latent state, these networks predict:
            *   **Observations**: Reconstructs what the agent might see in the predicted state (e.g., the next frame in a game).
            *   **Rewards**: Predicts the immediate reward the agent would receive in that state.
    *   **Benefit**: By learning to predict these outcomes, the world model allows the agent to generate hypothetical trajectories ("dreams") without needing to constantly interact with the real, often slower, environment.

2.  **The Imagined Behavior (Actor-Critic)**:
    *   **Purpose**: To learn the optimal way to act to maximize cumulative rewards.
    *   **Functionality**: This component learns almost entirely from trajectories imagined by the world model.
        *   **Actor**: This is the agent's policy. Given a latent state (from the world model's imagination), it decides which action to take. It's trained to pick actions that lead to high long-term rewards.
        *   **Critic**: This network estimates the expected total future reward (the "value") that the agent can achieve from a given latent state if it follows the actor's policy. It helps the actor by providing a signal of how good its actions are.
    *   **Benefit**: Training on imagined trajectories is highly efficient. The agent can experience and learn from millions of steps of "dreamed" interaction, far more than it could typically achieve in the real world in the same amount of time.

3.  **The Exploration Behavior**:
    *   **Purpose**: To guide the agent in efficiently gathering new and informative experiences from the *real* environment. This is crucial for improving the world model and discovering new, potentially rewarding, parts of the environment.
    *   **Functionality**: While the actor-critic learns to exploit its current knowledge, the exploration behavior encourages novelty. This can be:
        *   **Simple randomness**: Adding noise to the actor's actions.
        *   **Advanced strategies (e.g., Plan2Explore)**: The agent is intrinsically motivated to explore areas where its world model is uncertain or has high prediction error. This means it actively tries to learn more about parts of the world it doesn't understand well.
    *   **Benefit**: Effective exploration ensures that the world model becomes more accurate and comprehensive, leading to better overall performance.

## The Training Loop

The agent learns through an iterative process:

1.  **Data Collection (Real Interaction)**:
    *   The agent (using its current actor policy, possibly augmented by the exploration behavior) interacts with the actual environment.
    *   It collects a sequence of observations, actions taken, and rewards received. These experiences are stored in a replay buffer (a dataset of past experiences).

2.  **World Model Training (Learning from Reality)**:
    *   The agent samples batches of real experiences from the replay buffer.
    *   It trains the world model (encoder, dynamics model, and prediction heads) to accurately predict the observed sequences of states, rewards, and next observations.

3.  **Behavior Learning (Learning in Imagination/Dreaming)**:
    *   The agent uses its updated world model to generate long sequences of imagined trajectories.
    *   It starts from a state observed in the real world (or a latent state from the replay buffer).
    *   The actor proposes actions, and the world model predicts the next state and reward. This process is repeated for many steps, creating "dreams."
    *   The actor and critic are then trained on these imagined trajectories to maximize the expected cumulative rewards within these dreams.

4.  **Repeat**:
    *   The agent, now with an improved world model and a refined actor-critic policy, returns to step 1 to collect more data from the real environment.
    *   This cycle of real interaction, world model refinement, and behavior learning through imagination continues, allowing the agent to progressively improve its understanding of the world and its ability to achieve its goals.

This interplay between real experience, learned simulation, and imagined learning is what makes DreamerV3 a powerful and sample-efficient reinforcement learning algorithm.

## How to Run This Project

This section provides a step-by-step guide to get the DreamerV3 agent running. For the most authoritative and potentially updated instructions, always consult the main `README.md` file.

**Prerequisites**:
*   Python (the `README.md` mentions Python 3.11).
*   `pip` for installing packages.
*   Git for cloning the repository.

**Step 1: Get the Project Code**
If you're reading this file, you likely already have the project code.
This file (`PROJECT_EXPLANATION.md`) should be in the root directory of the project.

If you were starting from scratch, the first step would be to clone the repository:
```bash
# Example: git clone https://github.com/your-username/dreamerv3-torch.git
# cd dreamerv3-torch
```
Ensure you are in the project's root directory before proceeding to the next steps.

**Step 2: Install Dependencies**
The project uses a set of Python libraries listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```
This command will download and install all necessary packages.

**Step 3: Environment-Specific Setup (If Applicable)**
Some environments require additional setup. For example:
*   **Atari**: You might need to run a setup script as indicated in `envs/setup_scripts/atari.sh`. This often involves installing ROMs or specific environment dependencies.
    ```bash
    # Example for Atari, check envs/setup_scripts/ for others
    # sh envs/setup_scripts/atari.sh
    ```
*   **Minecraft**: Similarly, Minecraft has its own setup script in `envs/setup_scripts/minecraft.sh`.
*   **DeepMind Control Suite (DMC)**: Generally, DMC environments work after `pip install` if the dependencies (like MuJoCo) are correctly installed. The `MUJOCO_GL=osmesa` environment variable is often set in `dreamer.py`, which helps run it on headless servers.

Always check the `README.md` or the `envs/setup_scripts/` directory for specific instructions related to the environment you want to use.

**Step 4: Run Training**
The main script for training is `dreamer.py`. You'll need to provide configurations to tell it which environment to use and where to save results.

An example command (as seen in `README.md` for DMC Vision):
```bash
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk
```

Let's break down these arguments:
*   `--configs`: Specifies the base configuration to use. These are defined in `configs.yaml`. Examples:
    *   `dmc_vision`: For DeepMind Control Suite tasks using image observations.
    *   `dmc_proprio`: For DMC tasks using state (proprioceptive) observations.
    *   `atari`: For Atari game environments.
    *   `crafter`: For the Crafter environment.
*   `--task`: Specifies the particular task within the chosen configuration. Examples:
    *   `dmc_walker_walk`: The "walker_walk" task in DMC.
    *   `atari_pong`: The game Pong in Atari.
    The available tasks are usually implicitly defined by the environment suite and how it's set up (e.g., what sub-environments DMC or Atari provide).
*   `--logdir`: The directory where training logs, model checkpoints, and potentially videos of agent performance will be saved. It's good practice to make this specific to your experiment.

You can find other configurable parameters (like `batch_size`, `learning_rate`, etc.) in `configs.yaml` or by inspecting the argument parser in `dreamer.py` (towards the end of the file). You can often override these defaults from the command line:
```bash
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/test_run --batch_size 32 --learning_rate 1e-4
```

**Step 5: Monitor Training**
The project uses TensorBoard to log metrics during training. To view them:
1.  Open a new terminal.
2.  Navigate to the directory containing your `logdir`.
3.  Run TensorBoard:
    ```bash
    tensorboard --logdir ./logdir
    ```
    (If your `logdir` is `./logdir/dmc_walker_walk`, you can point directly to it or its parent directory).
4.  Open your web browser and go to the URL TensorBoard provides (usually `http://localhost:6006`). You'll see plots of rewards, loss functions, and other useful metrics.

**Alternative: Using Docker**
The `README.md` also mentions a `Dockerfile`. If you are familiar with Docker, this can be a convenient way to manage dependencies and ensure a consistent runtime environment. Refer to the instructions within the `Dockerfile` itself or in the `README.md` for guidance on building and running the project with Docker.

This covers the basic workflow. Remember to consult `configs.yaml` for available configurations and `dreamer.py` for the full list of command-line arguments to customize your training runs.
