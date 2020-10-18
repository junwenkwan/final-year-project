# Graph Neural Network (GNN)

```bash
python3 main.py --exp_name omniglot_N20_S1 --dataset omniglot \
		--test_N_way 20 --train_N_way 20 \
		--train_N_shots 1 --test_N_shots 1 \
		--batch_size 128  --dec_lr=10000  \
		--iterations 50000 \
		--dataset_root ../omniglot/python/ 
```
