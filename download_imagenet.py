import torchvision


def test():
    data_train = torchvision.datasets.ImageNet("./imagenet", download=True)
    data_test = torchvision.datasets.ImageNet("./imagenet", download=True, split="val")


if __name__ == "__main__":
    test()
