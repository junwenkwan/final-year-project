# Final Year Project
## Few-Shot Learning 
Few-shot learning experiments include:
* Siamese network [[paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)]
* Graph Neural Network (GNN) [[paper](https://openreview.net/pdf?id=BJj6qGbRW)]
* Model-Agnostic Meta-Learning (MAML) [[paper](https://arxiv.org/pdf/1703.03400.pdf)]

## Results
The overall results are summarized below.
|              | 1 shot 5 ways | 1 shot 20 ways|
| ------------ | ------------- | ------------- |
| Siamese Net  | 95.3% | 85.7% |
| MAML  | 99.2% | 96.6% |
| GNN   | 99.3% | 97.8% |


|              | 5 shots 5 ways | 5 shots 20 ways|
| ------------ | ------------- | ------------- |
| MAML  | 99.6% | 98.0% |
| GNN   | 99.7% | 98.5% |

## Credits
* [fangpin/siamese-pytorch](https://github.com/fangpin/siamese-pytorch)
* [vgsatorras/few-shot-gnn](https://github.com/vgsatorras/few-shot-gnn)
* [dragen1860/MAML-Pytorch](https://github.com/dragen1860/MAML-Pytorch)
