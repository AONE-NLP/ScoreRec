## ScoreRec

This is the official source code for ScoreRec.

Code and implementation details of our paper `ScoreRec: Reallocating Probability Mass in Sequential Recommendation via Score-based Diffusion Modeling`, accepted by ACM Transactions on Information Systems (TOIS) 2026.

## Motivation

![](https://github.com/AONE-NLP/ScoreRec/tree/main/ScoreRec/figure/Motivation.png)

## Model Architecture

![](https://github.com/AONE-NLP/ScoreRec/tree/main/ScoreRec/figure/model.jpg)

## Reproduce the results

### Zhihu Data

```python
python ScoreRec.py --Model ScoreRec --data zhihu --batch_size 128 --lr 0.001 --sigma 0.2 --sampler ode_sampler  --rtol 0.00001 --atol 0.00001 --diffuser_type mlp1 --num_head 1 --InfoNCE False --alpha 0.9 --temperature 0.1 --z 4
```

### YooChoose Data

```python
python ScoreRec.py --Model ScoreRec --data yc --batch_size 256 --lr 0.001 --sigma 0.3 --sampler ode_sampler  --rtol 0.001 --atol 0.0001 --diffuser_type mlp1 --num_head 1 --InfoNCE False --alpha 0.9 --temperature 0.1 --z 3
```

### Sports and Outdoors Data

```python
python ScoreRec.py --Model ScoreRec --data sports_and_outdoors --batch_size 256 --lr 0.00005 --sigma 0.18 --sampler ode_sampler  --rtol 0.0001 --atol 0.0001 --diffuser_type mlp1 --num_head 1 --InfoNCE False --alpha 0.9 --temperature 0.1 --z 3
```

## Acknowledgements

DreamRec: https://github.com/YangZhengyi98/DreamRec

FreqRec: https://github.com/AONE-NLP/FreqRec

PerferDiff: https://github.com/lswhim/PreferDiff

