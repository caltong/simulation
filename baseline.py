from load_dataset import load_dataset, load_test_dataset
from torch.distributions.one_hot_categorical import OneHotCategorical
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

use_gpu = torch.cuda.is_available()
num_epochs = 500
batch_size = 40
learning_rate = 0.05


class SimulationDataset(Dataset):
    def __init__(self, transform=None):
        self.all_inputs, self.all_outputs = load_dataset()
        self.transform = transform

    def __len__(self):
        return len(self.all_inputs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        inputs = self.all_inputs[idx]
        outputs = self.all_outputs[idx]
        sample = {'input': inputs, 'output': outputs}

        if self.transform:
            sample = self.transform(sample)

        return sample


class TestSimulationDataset(SimulationDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.all_inputs, self.all_outputs = load_test_dataset()
        self.transform = transform


class ToTensor(object):
    def __call__(self, sample):
        inputs, outputs = sample['input'], sample['output']
        return {'input': torch.from_numpy(inputs).float(), 'output': torch.tensor(outputs, dtype=torch.long)}


simulation_dataset = SimulationDataset(transform=transforms.Compose([ToTensor()]))
test_dataset = TestSimulationDataset(transform=transforms.Compose([ToTensor()]))
# for i in range(5):
#     sample = test_dataset[i]
#     print(sample['input'])
#     print(sample['output'])

data_loader = DataLoader(simulation_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)


class BaselineModel(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, out_dim):
        super(BaselineModel, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, out_dim), nn.ReLU(True))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


model = BaselineModel(1280, 1024, 256, 256, 10)
# print(model)
if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    print('*' * 10)
    print(f'epoch {epoch + 1}')
    running_loss = 0.0
    running_acc = 0.0

    for i, data in enumerate(data_loader, 1):
        x_train, y_train = data['input'], data['output']
        if use_gpu:
            x_train = x_train.cuda()
            y_train = y_train.cuda()
        out = model(x_train)
        loss = criterion(out, y_train)
        running_loss += loss.item()
        _, pred = torch.max(out, 1)
        running_acc += (pred == y_train).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 300 == 0:
            print(f'[{epoch + 1}/{num_epochs}] Loss: {running_loss / i:.6f}, Acc: {running_acc / i:.6f}')
    print(f'Finish {epoch + 1} epoch, Loss: {running_loss / i:.6f}, Acc: {running_acc / i:.6f}')

correct = 0
total = 0
with torch.no_grad():
    for data in test_data_loader:
        x_train, y_train = data['input'], data['output']
        if use_gpu:
            x_train = x_train.cuda()
            y_train = y_train.cuda()
        output = model(x_train)
        _, pred = torch.max(output.data, 1)
        total += y_train.size(0)
        correct += (pred == y_train).sum().item()
print('Accuracy of the network on the 12 test data: %d %%' % (
        100 * correct / total))
