import argparse
from tinygrad import Device

def train():
    print("Training backend: ", Device.DEFAULT)
    from .train import train

    train();

def test():
    print("Testing backend: ", Device.DEFAULT)
    from .test import test
    test();

def main():
    parser = argparse.ArgumentParser(description="RL training and testing")
    parser.add_argument('command', choices=['train', 'test'], help="Command to execute")

    args = parser.parse_args()

    if args.command == 'train':
        train()
    elif args.command == 'test':
        test()
    else:
        print("Please specify --train or --test")

if __name__ == "__main__":
    main()
