import torch
import numpy as np
from dataset import CustomDataset
from model import MyModel
from loss import BinaryFocalLoss
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / (y_test.shape[0] * y_test.shape[1])
    acc = torch.round(acc * 100)
    
    return acc

def generate_model(train_data, epochs=10, batch_size=64, dataset=None, java_path=None, t=None):
    if dataset is None:
        print('proccessing data...')
        dataset = CustomDataset(train_data, java_path=java_path, t=t)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model = MyModel(len(dataset.tokens) + 1, 32, 128, [256, 128, 64])
    model.to(device)
    criterion = BinaryFocalLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    for epoch in tqdm(range(epochs), desc='Epochs', leave=True):

        train_epoch_loss = 0
        train_epoch_acc = 0

        model.train()

        for lengths, inputs, labels in tqdm(train_loader, desc='Training Batches', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(lengths, inputs).squeeze()

            loss = criterion(outputs, labels)
            acc = binary_acc(outputs, labels)

            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()
            train_epoch_acc += acc
        
        train_loss = train_epoch_loss / len(train_loader)
        train_acc = train_epoch_acc / len(train_loader)

        print("epoch %d | loss: %.5f | accuracy: %.2f%%" % (epoch+1, train_loss, train_acc))

    print('Finished Training')
    return dataset.tokens, dataset.t, model

def compute_accuracy(dataset, model):
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=64)
    model.to(device)

    acc = 0

    with torch.no_grad():

        model.eval()

        for lengths, inputs, labels in tqdm(test_loader, desc='Batches', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(lengths, inputs).squeeze()
            acc += binary_acc(outputs, labels)

    return acc / len(test_loader)