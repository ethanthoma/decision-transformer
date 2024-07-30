from tinygrad import Device

def train():
    print("Training backend: ", Device.DEFAULT)
    from .train import train_rl

    train_rl();
