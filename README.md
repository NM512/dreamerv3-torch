# dreamerv3-torch
Pytorch implementation of [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104v1). DreamerV3 is a scalable algorithm that outperforms previous approaches across various domains with fixed hyperparameters.

## Instructions

Get dependencies:
```
pip install -r requirements.txt
```
Run training on DMC Vision:
```
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk
```
Monitor results:
```
tensorboard --logdir ./logdir
```

## Benchmarks
So far, this repository allows testing the following benchmarks.
| Environment        | Observation | Action | Description |
|-------------------|---|---|-----------------------|
| [DMC Proprio](https://github.com/deepmind/dm_control) | State | Continuous | This benchmark contains 18 continuous control tasks with low-dimensional inputs and a budget of 500K environment steps. |
| [DMC Vision](https://github.com/deepmind/dm_control) | Image | Continuous | This benchmark consists of 20 continuous control tasks where the agent receives only high-dimensional images as inputs and a budget of 1M environment steps. |
| [Atari 100k](https://github.com/openai/atari-py) | Image | Discrete | This benchmark includes 26 Atari games and a budget of only 400K environment steps, amounting to 100K steps after action repeat or 2 hours of real time. |
| [Crafter](https://github.com/danijar/crafter) | Image | Discrete | This survival environment evaluates diverse agent abilities, including exploration, reasoning, credit assignment, and generalization.|
| [Memory Maze](https://github.com/jurgisp/memory-maze) | Image | Discrete | Memory Maze is a 3D benchmark with randomized mazes to evaluate RL agents' long-term memory.|

## Results
#### DMC Vision
![dmcvision](https://github.com/NM512/dreamerv3-torch/assets/70328564/b710d217-2428-4fa0-8471-55e15ec5aa43)

#### Atari 100k
![atari100k](https://github.com/NM512/dreamerv3-torch/assets/70328564/0da6d899-d91d-44b4-a8c4-d5b37413aa11)

#### DMC Proprio
![dmcproprio](https://github.com/NM512/dreamerv3-torch/assets/70328564/7f6e47a5-3235-4bc4-bef9-15ff96782d5e)


## Acknowledgments
This code is heavily inspired by the following works:
- danijar's Dreamer-v3 jax implementation: https://github.com/danijar/dreamerv3
- danijar's Dreamer-v2 tensorflow implementation: https://github.com/danijar/dreamerv2
- jsikyoon's Dreamer-v2 pytorch implementation: https://github.com/jsikyoon/dreamer-torch
- RajGhugare19's Dreamer-v2 pytorch implementation: https://github.com/RajGhugare19/dreamerv2
- denisyarats's DrQ-v2 original implementation: https://github.com/facebookresearch/drqv2
