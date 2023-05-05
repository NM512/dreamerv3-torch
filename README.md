# dreamerv3-torch
Pytorch implementation of [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104v1). DreamerV3 is a scalable algorithm that outperforms previous approaches across various domains with fixed hyperparameters.

## Instructions

Get dependencies:
```
pip install -r requirements.txt
```
Train the agent on Walker Walk in Vision DMC:
```
python3 dreamer.py --configs defaults --task dmc_walker_walk --logdir ~/dreamerv3-torch/logdir/dmc_walker_walk
```
Train the agent on Alien in Atari 100K:
```
python3 dreamer.py --configs defaults atari100k --task atari_alien --logdir ~/dreamerv3-torch/logdir/atari_alien
```
Monitor results:
```
tensorboard --logdir ~/dreamerv3-torch/logdir
```

## Evaluation Results
More results will be added in the future.

![dmc_vision](https://user-images.githubusercontent.com/70328564/236276650-ae706f29-4c14-4ed3-9b61-1829a1fdedae.png)
![atari100k](https://user-images.githubusercontent.com/70328564/236276669-16a56be3-40d6-49fd-befa-97c72b7d2460.png)
## ToDo
- [x] Prototyping
- [x] Modify implementation details based on the author's implementation
- [x] Evaluate on DMC vision
- [x] Evaluate on Atari 100K
- [ ] Add state input capability
- [ ] Evaluate on DMC Proprio
- [ ] etc.

## Acknowledgments
This code is heavily inspired by the following works:
- danijar's Dreamer-v3 jax implementation: https://github.com/danijar/dreamerv3
- danijar's Dreamer-v2 tensorflow implementation: https://github.com/danijar/dreamerv2
- jsikyoon's Dreamer-v2 pytorch implementation: https://github.com/jsikyoon/dreamer-torch
- RajGhugare19's Dreamer-v2 pytorch implementation: https://github.com/RajGhugare19/dreamerv2
- denisyarats's DrQ-v2 original implementation: https://github.com/facebookresearch/drqv2
