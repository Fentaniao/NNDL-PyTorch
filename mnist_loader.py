# -*- coding: utf-8 -*-
"""
Data loader for the MNIST dataset.
MNIST数据集的数据加载器。

Initial Author: Jing Li
Last modified by: Ruiyang Zhou
"""

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms

batch_size = 64


def load_data():
    """
    Download the training set and test set in the MNIST dataset through the `datasets` module in the torchvision.
    Load the training set and test set as train_loader, validation_loader, and test_loader through DataLoader.

    通过torchvision中datasets模块导入MNIST数据集中的训练集和测试集。
    通过DataLoader加载训练集。
    """

    # Import the training set and test set in the MNIST dataset through the `datasets` module in the torchvision.
    # 通过torchvision中datasets模块导入MNIST数据集中的训练集和测试集。
    train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    # Load train_dataset as train_loader through DataLoader.
    # 通过DataLoader加载训练集。
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Shuffle the samples in the test set and divide them into validation set and test set.
    # 打乱测试集样本，并分为验证集和测试集。
    indices = range(len(test_dataset))
    indices_val = indices[:5000]
    indices_test = indices[5000:]

    sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)
    sampler_test = torch.utils.data.sampler.SubsetRandomSampler(indices_test)

    # Loaded the validation set and test set as validation_loader and test_loader through DataLoader.
    # 通过DataLoader加载验证集和测试集。
    validation_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler_val)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler_test)

    return train_loader, validation_loader, test_loader
