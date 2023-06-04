# dreamerv3-torch
Pytorch implementation of [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104v1). DreamerV3 is a scalable algorithm that outperforms previous approaches across various domains with fixed hyperparameters.

## Instructions

Get dependencies:
```
pip install -r requirements.txt
```
Train the agent on Walker Walk in DMC Vision:
```
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk
```
Train the agent on Walker Walk in DMC Proprio:
```
python3 dreamer.py --configs dmc_proprio --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk
```
Train the agent on Alien in Atari 100K:
```
python3 dreamer.py --configs atari100k --task atari_alien --logdir ./logdir/atari_alien
```
Monitor results:
```
tensorboard --logdir ./logdir
```

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
