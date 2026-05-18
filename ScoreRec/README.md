
## ScoreRec

## Reproduce the results

### Zhihu Data

python ScoreRec.py --Model ScoreRec --data zhihu --batch_size 128 --lr 0.001 --sigma 0.2 --sampler ode_sampler  --rtol 0.00001 --atol 0.00001 --diffuser_type mlp1 --num_head 1 --InfoNCE False --alpha 0.9 --temperature 0.1 --z 4

### YooChoose Data

python ScoreRec.py --Model ScoreRec --data yc --batch_size 256 --lr 0.001 --sigma 0.3 --sampler ode_sampler  --rtol 0.001 --atol 0.0001 --diffuser_type mlp1 --num_head 1 --InfoNCE False --alpha 0.9 --temperature 0.1 --z 3

### Sports and Outdoors Data

python ScoreRec.py --Model ScoreRec --data sports_and_outdoors --batch_size 256 --lr 0.00005 --sigma 0.18 --sampler ode_sampler  --rtol 0.0001 --atol 0.0001 --diffuser_type mlp1 --num_head 1 --InfoNCE False --alpha 0.9 --temperature 0.1 --z 3

### Acknowledgements

DreamRec: https://github.com/YangZhengyi98/DreamRec

BSARec: https://github.com/yehjin-shin/BSARec

PerferDiff: https://github.com/lswhim/PreferDiff
 

