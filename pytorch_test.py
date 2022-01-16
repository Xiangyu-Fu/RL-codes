import torch
import numpy as np
import torch.nn.functional as F
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# x = torch.linspace(-1, 1, 100)
y = x.pow(2) + 0.2*torch.rand(x.size())
print('x, y: {}, {},\n'.format(x.size(), y.size()))

# x, y = Variable(x), Variable(y)
print('new x, y: {}, {},\n'.format(x.size(), y.size()))

plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


# 建立神经网络模型
class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


plt.ion()
plt.show()

net = Net(1, 10, 1)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)   # 设置优化器
loss_func = torch.nn.MSELoss()                          # 设置损失函数
print(net)
for t in range(100):
    prediction = net(x)

    loss = loss_func(prediction, y)

    optimizer.zero_grad()  # 将梯度设为零
    loss.backward()
    optimizer.step()
    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})
        print(loss.data)
        plt.pause(0.1)

plt.ioff()
plt.show()

print(torch.version.cuda)
print(torch.cuda.is_available())