mkdir -p ./data/Pong

gsutil -m cp -R gs://atari-replay-datasets/dqn/Pong/5 ./data/Pong
