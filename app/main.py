from tinygrad import Device
from .config import get_config


def main():
    print("Backend: ", Device.DEFAULT)
    config = get_config()

    match config["command"]:
        case "train":
            from .train import train

            assert config["dataset_size"] >= config["batch_size"]

            train(config)
        case "test":
            from .test import test

            test(config)
        case _:
            print("Please specify --train or --test")


if __name__ == "__main__":
    main()
