# -*- coding: utf-8 -*-
"""
Solution for the MNIST dataset classification problem using pytorch.
MNIST数据集分类问题的pytorch解决方案。

By using the modules and functions of the neural network in pytorch,
construct the network of the MNIST dataset, training, validation, and testing.
and build a network with three layers of neurons;
Finally, the correct rate in the test set is about 94.36%.

通过pytorch中神经网络的模块和函数
构建对MNIST数据集网络的构建、 训练、验证、和测试。
整个过程使用了三层的神经元的网络来建立网络；
最后，测试集中的正确率有94.36%左右。

By increasing the number of network layers, adjusting parameters, number of iterations, loss functions, etc.,
can improve the correct rate to a certain extent.

通过增加网络层数，调整参数，迭代次数， 损失函数等等
都能对提高正确率起一定效果。

Initial Author: Jing Li
Last modified by: Ruiyang Zhou
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import mnist_loader


class Network(nn.Module):
    """
    Construct a three-layer neural network with
    784 input neurons,
    30 hidden neurons with Linear activation function,
    and 10 output neurons with LogSoftmax activation function.

    构建一个三层神经网络，
    输入层有784个神经元，
    隐藏层有30个神经元，采用Linear激活函数，
    输出层有10个神经元，采用LogSoftmax激活函数。
    """

    def __init__(self, sizes):
        super(Network, self).__init__()
        self.sizes = sizes
        self.layer1 = nn.Linear(sizes[0], sizes[1])
        self.layer2 = nn.Linear(sizes[1], sizes[2])

    def forward(self, a):
        # view function will convert the input Tensor to (64, 784).
        # view函数将输入Tensor转换成（64, 784）。
        a = a.view(-1, self.sizes[0])

        a = self.layer1(a)
        a = self.layer2(a)

        a = torch.log_softmax(a, dim=1)

        return a


def rightness(output, target):
    """
    Input the output Tensor of the network and the target Tensor,
    compare the output Tensor of the network and the target Tensor,
    and return the number of correct matches in the comparison results and the length of target Tensor.

    输入神经网络的输出张量和目标张量，
    比较神经网络的输出张量和目标张量中对应相等的结果，
    返回比较结果中匹配正确的个数和目标张量的长度。
    """

    rights = 0
    for index in range(len(target.data)):
        if torch.argmax(output[index]) == target.data[index]:
            rights += 1
    return rights, len(target.data)


def train_model(train_loader, epochs, eta):
    """
    Train the model.

    本函数的功能是训练模型。

    Use the cross-entropy loss function and the stochastic gradient descent optimization algorithm,
    the learning rate is 0.001, and the momentum is 0.9.

    使用交叉熵损失函数和随机梯度下降优化算法，
    学习率为0.001，动量为0.9。
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=eta, momentum=0.9)

    for epoch in range(epochs):
        # Record the correct results and total samples of each iteration.
        # 记录每次迭代正确的结果和总样本。
        train_rights = []

        for batch_idx, (data, target) in enumerate(train_loader):
            # Set the network to training mode.
            # 将网络设置为训练模式。
            net.train()

            # Wrap the data and target into Variable.
            # 将数据和目标包装进Variable。
            data, target = Variable(data), Variable(target)

            # Forward propagation.
            # 前向传播。
            output = net(data)

            # Calculate the loss.
            # 计算损失。
            loss = criterion(output, target)

            # Clear the gradient.
            # 清空梯度。
            optimizer.zero_grad()

            # Back propagation.
            # 反向传播。
            loss.backward()

            # One step stochastic gradient descent algorithm.
            # 一步随机梯度下降算法。
            optimizer.step()

            # Calculate correct results in a batch of accuracy (number of correct samples / total number of samples).
            # 计算一批次的准确率（正确样例数 / 总样本数）。
            right = rightness(output, target)
            train_rights.append(right)

            if batch_idx % 100 == 0:
                validation_model(validation_loader)

        # Calculate the total number of correct samples and the total number of samples in the entire training sample,
        # and compare the two to get the correct rate of training.
        # 求得整个训练样本中正确的样例总数和总样本数，
        # 通过比较两者得到训练的正确率。
        train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
        print("Epoch {0}: {1}/{2}".format(epoch, train_r[0], train_r[1]))


def validation_model(validation_loader):
    """
    Validate the model.

    验证模型。
    """

    net.eval()
    val_rights = []

    for data, target in validation_loader:
        data, target = Variable(data), Variable(target)
        output = net(data)
        right = rightness(output, target)
        val_rights.append(right)

    val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
    print("Accuracy of validation set: {:.2f}%".format(100.0 * val_r[0] / val_r[1]))


def test_model(test_loader):
    """
    Test the model.

    测试模型。
    """

    net.eval()
    vals = []
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        output = net(data)
        val = rightness(output, target)
        vals.append(val)

    rights = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
    print("Accuracy of test set: {:.2f}%".format(100.0 * rights[0] / rights[1]))


if __name__ == '__main__':
    train_loader, validation_loader, test_loader = mnist_loader.load_data()
    net = Network([784, 30, 10])
    train_model(train_loader, 20, 0.001)
    test_model(test_loader)
