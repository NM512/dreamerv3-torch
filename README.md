# dreamerv3-torch
Pytorch implementation of [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104v1).\
Validation is in progress.
![Figure_1](https://user-images.githubusercontent.com/70328564/226098745-4f6f33f2-2cb8-495a-a5c7-4a9196859238.png)

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

## ToDo
- [x] Prototyping
- [x] Modify implementation details based on the author's implementation
- [ ] Evaluate on visual DMC suite
- [ ] Add state input capability and evaluate on Proprio Control Suite environment
- [ ] Add model size options and evaluate on environments which requires that (like Minecraft)
- [ ] etc.


## Acknowledgments
This code is heavily inspired by the following works:
- danijar's Dreamer-v3 jax implementation: https://github.com/danijar/dreamerv3
- danijar's Dreamer-v2 tensorflow implementation: https://github.com/danijar/dreamerv2
- jsikyoon's Dreamer-v2 pytorch implementation: https://github.com/jsikyoon/dreamer-torch
- RajGhugare19's Dreamer-v2 pytorch implementation: https://github.com/RajGhugare19/dreamerv2
- denisyarats's DrQ-v2 original implementation: https://github.com/facebookresearch/drqv2
