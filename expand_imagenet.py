import torchvision


def test():
    data_train = torchvision.datasets.ImageNet("./imagenet")
    data_test = torchvision.datasets.ImageNet("./imagenet", split="val")


if __name__ == "__main__":
    test()
