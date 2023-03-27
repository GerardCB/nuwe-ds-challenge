import os
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import pandas as pd
from sklearn.metrics import f1_score   
from matplotlib import pyplot as plt

def train(net, criterion, optimizer, trainloader, epoch, device):
    """
    Train the model for binary classification.
    Args:
        net: the model
        criterion: the loss function
        optimizer: the optimizer
        trainloader: the dataloader for the training set
        epoch: the current epoch
        device: the device to use (cpu or gpu)
    """
    running_loss = 0.0
    for data in trainloader:
        inputs, labels = data['data'], data['target']
        inputs, labels = inputs.to(device), labels.to(device)

        # Convert to float32, else we get an error
        inputs = inputs.type(torch.float32) 
        labels = labels.type(torch.float32)

        inputs.reshape(-1, 1)

        optimizer.zero_grad()

        outputs = net(inputs)  # get predictions

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    training_loss = running_loss / len(trainloader)
    print(f'Epoch {epoch} (train) - Loss: {training_loss:.3f}')
    return training_loss

def test(net, testloader, epoch, device):
    """
    Test the model on the test/validation set for binary classification.
    Compute the accuracy and the F1 score.
    Args:
        net: the model
        testloader: the dataloader for the test/validation set
        epoch: the current epoch
        device: the device to use (cpu or gpu)
    """
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data['data'], data['target']
            inputs, labels = inputs.to(device), labels.to(device)

            # Convert to float32, else we get an error
            inputs = inputs.type(torch.float32) 
            labels = labels.type(torch.float32)

            inputs.reshape(-1, 1)

            outputs = net(inputs)  # get model predictions
            predicted = torch.round(outputs)  # binary classification

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true += labels.tolist()
            y_pred += predicted.tolist()
    accuracy = correct / total
    f1 = f1_score(y_true, y_pred)
    print(f'Epoch {epoch} (test) - Accuracy: {accuracy:.3f} - F1 score: {f1:.3f}')
    return accuracy, f1

def get_nn_nparams(net: torch.nn.Module) -> int:
  """
  Function that returns all parameters regardless of the require_grad value.
  https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/6
  """
  return sum([torch.numel(p) for p in list(net.parameters())])

def plot_evaluation(losses, accuracies, f1_scores, model_name:str=None, dataset_name:str=None, save=False):
    """
    Plot the training loss, validation accuracy and validation F1 score.
    Args:
        losses: list of training losses
        accuracies: list of validation accuracies
        f1_scores: list of validation F1 scores
        model_name: name of the model
        dataset_name: name of the dataset
        save: whether to save the plot or not
    """
    assert len(losses) == len(accuracies) == len(f1_scores), "The lists must have the same length."
    assert not (save and (model_name is None or dataset_name is None)), "If save is True, model_name and dataset_name must be specified."

    fig, axarr = plt.subplots(3, sharex=True)

    axarr[0].plot(losses)
    axarr[0].set_title('Training loss')
    axarr[0].set_ylabel('Train Loss')
    axarr[0].grid()

    axarr[1].plot(accuracies)
    axarr[1].set_ylabel('Val Accuracy (%)')
    axarr[1].grid()

    axarr[2].plot(f1_scores)
    axarr[2].set_ylabel('Val F1 score')
    axarr[2].grid()
    plt.xlabel('Epoch')

    if save:
        print(f"Saving the plot as 'results_{model_name}_{dataset_name}.png'...")
        plot_relative_path = f'../model/results_{model_name}_{dataset_name}.png'
        plot_full_path = os.path.join(os.path.dirname(__file__), plot_relative_path)
        plt.savefig(plot_full_path)

    return fig

class SmoothedF1Loss(nn.Module):
    """Custom smoothed F1 loss function."""
    def __init__(self):
        super(SmoothedF1Loss, self).__init__()

    def forward(self, y_true, y_pred):
        """
        Args:
            y_true: tensor of shape (batch_size, 1)
            y_pred: tensor of shape (batch_size, 1)
        """
        eps = 1e-8
        tp = torch.sum(y_true * y_pred, axis=0)
        fp = torch.sum((1 - y_true) * y_pred, axis=0)
        fn = torch.sum(y_true * (1 - y_pred), axis=0)

        f1 = 2 * tp / (2 * tp + fp + fn + eps)
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
        return 1 - torch.mean(f1)

class SupplyChainDataset(Dataset):
    """Supply Chain dataset for NUWE challenge."""
    def __init__(self, data_path, split:str=None, frac:float=0.85):
        """
        Args:
            data_path (string): Path to the csv file with annotations.
            split (string, optional): Optional split data into train/test
            frac (float, optional): Fraction of data to use for training
        """
        self.data_frame = pd.read_csv(data_path, index_col=0)

        # Split the data
        if split is not None:
            train_data = self.data_frame.sample(frac=frac, random_state=94)
            test_data = self.data_frame.drop(train_data.index)
            if split == 'train':
                self.data_frame = train_data.reset_index(drop=True)
            else:  # split == val
                self.data_frame = test_data.reset_index(drop=True)
        
        # Preprocess the data
        self.data_frame = self._data_processing(self.data_frame)

    def _data_processing(self, data_frame):
        # One-hot encoding
        data_frame = pd.get_dummies(
            data_frame,
            columns=data_frame.columns[data_frame.dtypes == 'object'],
        )
        # Move the target column to the end
        target_column = data_frame.pop('Attrition_Flag')
        data_frame['Attrition_Flag'] = target_column

        # Normalize the data (except binary columns)
        binary_columns = data_frame.columns[data_frame.nunique() == 2]
        binary_cols_data = data_frame[binary_columns]
        data_frame = (data_frame - data_frame.mean()) / data_frame.std()
        data_frame[binary_columns] = binary_cols_data
        return data_frame
        
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data_frame.iloc[idx].to_numpy()        
        sample = {
            'data': data[:-1],
            'target': data[-1].astype(int)}
        return sample

