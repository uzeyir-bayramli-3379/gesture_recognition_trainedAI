import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from torch import nn
from windowing import arr
from windowing import labels
from windowing import seq_ids
import torch
from sklearn.metrics import confusion_matrix
mask=np.isin(seq_ids,['4','9','10_3','11_2'])
X_train,y_train=arr[~mask],labels[~mask]
X_val,y_val=arr[mask],labels[mask]
X_train=np.permute_dims(X_train,(0,2,1))
print(X_train.shape)
X_val=np.permute_dims(X_val,(0,2,1))
print(X_val.shape)

class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()
        self.layer1 = nn.Conv1d(in_channels=63, out_channels=32, kernel_size=5, stride=2)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)
        self.act2 = nn.ReLU()
        self.m=nn.AdaptiveAvgPool1d(1)
        self.layer3=nn.Linear(32,3)
    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x= self.act2(x)
        x= self.m(x)
        x=x.squeeze(-1)
        x=self.layer3(x)
        return x

cel_loss=nn.CrossEntropyLoss()


if __name__ == "__main__":
    model=CNN_model()
    opt_adam=torch.optim.Adam(model.parameters(),lr=0.001)
    X_train=torch.from_numpy(X_train).float()
    y_train=torch.from_numpy(y_train).long()
    X_val=torch.from_numpy(X_val).float()
    y_val=torch.from_numpy(y_val).long()

    opt_adam.zero_grad()
    outputs=model(X_train)
    loss=cel_loss(outputs,y_train)
    loss.backward()
    opt_adam.step()

    for epoch in range(60):
        opt_adam.zero_grad()
        outputs=model(X_train)
        loss=cel_loss(outputs,y_train)
        loss.backward()
        opt_adam.step()
        print(str(epoch) + " " + str(loss.item()))


    model.eval()
    with torch.no_grad():
        outputs = model(X_val)
        _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_val).float().mean()
    print(f"Val Accuracy: {accuracy:.4f}")

print(confusion_matrix(y_val.numpy(), predicted.numpy()))
torch.save(model.state_dict(), 'scuba_model.pth')