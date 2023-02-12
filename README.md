# Dreamer-v3 Pytorch
Pytorch implementation of [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104v1)

![image_walker_walk](https://user-images.githubusercontent.com/70328564/218313056-c1158a7d-10f3-4052-b19d-6d642ee4850b.gif)

## Instructions
Get dependencies:
```
pip install -r requirements.txt
```
Train the agent:
```
python3 dreamer.py --configs defaults --logdir $ABSOLUTEPATH_TO_SAVE_LOG
```
Monitor results:
```
tensorboard --logdir $ABSOLUTEPATH_TO_SAVE_LOG
```
## Evaluation Results
work-in-progress

![Fig](https://user-images.githubusercontent.com/70328564/218313252-3d42193a-a7c4-4fd1-bd0a-df4f4f5787d5.png)

## Awesome Environments used for testing:
- Deepmind control suite: https://github.com/deepmind/dm_control
- will be added soon

## Acknowledgments
This code is heavily inspired by the following works:
- danijar's Dreamer-v2 tensorflow implementation: https://github.com/danijar/dreamerv2
- jsikyoon's Dreamer-v2 pytorch implementation: https://github.com/jsikyoon/dreamer-torch
- RajGhugare19's Dreamer-v2 pytorch implementation: https://github.com/RajGhugare19/dreamerv2
- denisyarats's DrQ-v2 original implementation: https://github.com/facebookresearch/drqv2
