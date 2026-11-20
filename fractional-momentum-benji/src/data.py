import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_data_loaders(dataset_name='MNIST', batch_size=64, data_dir='./data'):
    if dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError("Dataset not supported")

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader