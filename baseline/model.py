import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class CnnModel(nn.Module):
    def __init__(self, num_classes=11, image_size=0):
        super(CnnModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 6 * 6, 128)  # After three 2x2 max-pool layers, 48 -> 24 -> 12 -> 6
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

        self.logits = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)  # Flatten the feature maps
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        #x = self.logits(x)
        return x  # No softmax, as it's handled in CrossEntropyLoss

    def get_representation(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)  # Flatten the feature maps
        x = self.fc1(x)
        return x


class TemperatureScalingCalibrationModule(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

        # the single temperature scaling parameter, the initialization value doesn't
        # seem to matter that much based on some ad-hoc experimentation
        self.temperature = nn.Parameter(torch.ones(1))

    def forward_unscaled(self, x):
        logits = self.model(x)
        scores = nn.functional.softmax(logits, dim=1)
        return scores

    def forward(self, x):
        scaled_logits = self.forward_logit(x)
        scores = nn.functional.softmax(scaled_logits, dim=1)
        return scores

    def forward_logit(self, x):
        logits = self.model(x)
        return logits / self.temperature

    def fit(self, data_loader, n_epochs: int = 10, lr: float = 0.01, start_value=1.):
        """fits the temperature scaling parameter."""
        assert isinstance(data_loader, DataLoader), "data_loader must be an instance of DataLoader"

        self.temperature.data.fill_(start_value)

        self.freeze_base_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=lr)

        for epoch in range(n_epochs):
            epoch_loss = 0
            for batch in data_loader:
                images, labels = batch[:2]

                self.zero_grad()
                scaled_logits = self.forward_logit(images)  # Use forward to get scaled logits
                loss = criterion(scaled_logits, labels)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach()
                # print("   ", self.temperature)
            # print(epoch_loss/len(data_loader), self.temperature.detach())

        return self

    def freeze_base_model(self):
        """remember to freeze base model's parameters when training temperature scaler"""
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        return self


## source:
## https://github.com/clcarwin/focal_loss_pytorch/blob/e11e75bad957aecf641db6998a1016204722c1bb/focalloss.py#L6
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()