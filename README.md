<h3 align="center">
    Decision Transformer in Tinygrad
</h3>

This codebase is an implementation of the 
["Decision Transformer: Reinforcement Learning via Sequence Modeling"](https://arxiv.org/abs/2106.01345) 
paper in [`tinygrad`](https://github.com/tinygrad/tinygrad/).

The `flake.nix` uses [poetry2nix](https://github.com/nix-community/poetry2nix) 
for building and running the code. The data fecthing **must** be done manually.

## Running

To run the code, you will have to clone it:
```
git clone https://github.com/ethanthoma/decision-transformer.git
cd decision-transformer
```

Next, you will need to download the data. The code only takes a subset from one
split (of five) per each game from the Batch RL [DQN Replay dataset](https://github.com/google-research/batch_rl).
There is a simple script that will download all the splits for `Pong`:
```
./download_pong_data.sh
```

Finally, you can train the model via the `train` command:
```
nix run . -- train
```

And test your model via
```
nix run . -- test
```

Both the train and test commands have a lot of arguments you can set. Use `-h` 
with the command to see all available flags.

## Performance

### Data Loading

I did not do any timings or benchmarks for the code. However, I did originally 
use the data loader code from Google Research's Batch RL codebase. This used
`TensorFlow` and `dopamine-rl`. Based on the logs after running it for 2 days, it 
would have taken a total of 7 days to generate the training set that was used in
the original paper.

I remade the data loader using only `numpy`. It also uses multiple threads to load
the 50 checkpoints. Data loading is now done in one day instead of 7.

### Model

The model code is very similar to the `PyTorch` implementation from the [original
codebase](https://github.com/kzl/decision-transformer). Everything is reimplemented
in `tinygrad`. 

The model uses about 31GBs of RAM during training and 0.5GBs during testing.

## Modifications

TBD

## Results

TBD
