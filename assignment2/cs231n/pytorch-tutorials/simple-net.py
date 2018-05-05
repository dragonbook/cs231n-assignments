import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        # after cov5-maxpool2-conv5-maxpool2: (32, 32) -> (5, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size: num_features *= s
        return num_features


net = Net()
print(net)



for p in net.parameters():
    print(p.data.size())

params = list(net.parameters())
print(len(params))
print(params[0].size())
print(params[1].size())
print(params[2].size())
print(params[3].size())
print(params[4].size())
print(params[5].size())
print(params[6].size())
print(params[7].size())
print(params[8].size())
print(params[9].size())

input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
print(out)

#net.zero_grad()
#out.backward(torch.randn(1, 10))

target = Variable(torch.arange(1, 11))
criterion = nn.MSELoss()
loss = criterion(out, target)
print(loss)

#print(loss.grad_fn)

net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.01)
# in training loop
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()


for name, param in net.named_parameters():
    if param.requires_grad:
        print(name, param.data)


#print('net.conv1: ', net.conv1)
print('net.conv1.weights: ', net.conv1.weight.data.cpu().numpy().shape)
print('net.conv1.bias: ', net.conv1.bias.data.cpu().numpy().shape)