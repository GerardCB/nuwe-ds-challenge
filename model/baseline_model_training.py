import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import f1_score   
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

# ----------------- Hyperparameters -----------------
model_name = 'baseline_model'
dataset_name = 'balanced' # one of 'raw', 'cleaned', 'balanced'
assert dataset_name in {'raw', 'cleaned', 'balanced'}, 'Invalid dataset name'

hparams = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 30,
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'weight_decay': 0.0001
}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------- Function definitions -----------------
def train(net, criterion, optimizer, trainloader, epoch):
    """
    Train the model for binary classification.
    Args:
        net: the model
        criterion: the loss function
        optimizer: the optimizer
        trainloader: the dataloader for the training set
        epoch: the current epoch
    """
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data['data'], data['target']
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # Convert to float32, else we get an error
        inputs = inputs.type(torch.float32) 
        labels = labels.type(torch.float32)

        inputs.reshape(-1, 1)

        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    training_loss = running_loss / len(trainloader)
    print(f'Epoch {epoch} (train) - Loss: {training_loss:.3f}')
    return training_loss


def test(net, testloader, epoch):
    """
    Test the model on the test/validation set for binary classification.
    Compute the accuracy and the F1 score.
    Args:
        net: the model
        testloader: the dataloader for the test/validation set
        epoch: the current epoch
    """
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data['data'], data['target']
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Convert to float32, else we get an error
            inputs = inputs.type(torch.float32) 
            labels = labels.type(torch.float32)

            inputs.reshape(-1, 1)

            outputs = net(inputs)
            predicted = torch.round(outputs)  # binary classification

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true += labels.tolist()
            y_pred += predicted.tolist()
    accuracy = correct / total
    f1 = f1_score(y_true, y_pred)
    print(f'EPOCH {epoch} (test) - Accuracy: {accuracy:.3f} - F1 score: {f1:.3f}')
    return accuracy, f1

def get_nn_nparams(net: torch.nn.Module) -> int:
  """
  Function that returns all parameters regardless of the require_grad value.
  https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/6
  """
  return sum([torch.numel(p) for p in list(net.parameters())])

# ----------------- Data preprocessing -----------------
print('Creating dataset...')
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
    
# Load and preprocess the data: create Dataset objects
data_relative_path = f'../data/supply_chain_train_{dataset_name}.csv'
data_path = os.path.join(os.path.dirname(__file__), data_relative_path)
print(f'Loading data from {data_path}')

# create the dataset splits
train_dataset = SupplyChainDataset(data_path, split='train')
val_dataset = SupplyChainDataset(data_path, split='val')

# create the dataloaders
train_loader = DataLoader(train_dataset, hparams['batch_size'],
                        shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, hparams['batch_size'],
                        shuffle=False, num_workers=0)

# Test the dataloaders
for i_batch, sample_batched in enumerate(train_loader):
    print(f'Train batch {i_batch}:')
    print(sample_batched)
    break
for i_batch, sample_batched in enumerate(val_loader):
    print(f'Validation batch {i_batch}:')
    print(sample_batched)
    break

batch_data, batch_target = sample_batched.values()
print('batch_data.shape:', batch_data.shape)

print('Percentage of positive and negative samples in the training set:')
print(train_dataset.data_frame["Attrition_Flag"].value_counts(normalize=True))

# ----------------- Model -----------------
class BaselineNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        """
        Args:
            input_size: the size of the input
            hidden_size: the size of the hidden layers
            num_layers: the number of hidden layers
            dropout: the dropout rate
        There are as many hidden layers as num_layers.
        """
        super(BaselineNet, self).__init__()
        self.fc = nn.Sequential()  # Fully connected layers

        # First layer
        self.fc.add_module('fc0', nn.Linear(input_size, hidden_size))
        self.fc.add_module('relu0', nn.ReLU())
        self.fc.add_module('dropout0', nn.Dropout(dropout))

        # Hidden layers
        for i in range(1, num_layers):
            self.fc.add_module(f'fc{i}', nn.Linear(hidden_size, hidden_size))
            self.fc.add_module(f'relu{i}', nn.ReLU())
            self.fc.add_module(f'dropout{i}', nn.Dropout(dropout))
            
        # Last layer
        self.fc.add_module(f'fc{num_layers}', nn.Linear(hidden_size, 1))
        self.fc.add_module(f'sigmoid', nn.Sigmoid())

    def forward(self, x):
        x = self.fc(x)
        x = x.squeeze(1)
        return x

# Get the input size from the dataset, to be used in the model as the first layer's input size
input_size = train_dataset.data_frame.shape[1] - 1
model = BaselineNet(input_size, hparams['hidden_size'], hparams['num_layers'], hparams['dropout'])

# Print the model parameters
print(model)
print('Num params: ', get_nn_nparams(model))

model.to(DEVICE)

# ----------------- Training -----------------
print('Training the model...')
# Loss function and the optimizer for binary classification
criterion = nn.BCELoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=hparams['learning_rate'],
    weight_decay=hparams['weight_decay']
)

# Train the model
losses = []
accuracies = []
f1_scores = []
for epoch in tqdm(range(hparams['num_epochs'])):
    train_loss = train(model, criterion, optimizer, train_loader, epoch)
    losses.append(train_loss)

    val_acc, val_f1 = test(model, val_loader, epoch)
    accuracies.append(val_acc)
    f1_scores.append(val_f1)

# ------------------- EVALUATE RESULTS -------------------
# Plot the training loss, validation accuracy and validation F1 score
f, axarr = plt.subplots(3, sharex=True)

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

# Save plot
print(f"Saving the plot as 'results_{model_name}_{dataset_name}.png'...")
plot_relative_path = f'./results_{model_name}_{dataset_name}.png'
plot_full_path = os.path.join(os.path.dirname(__file__), plot_relative_path)
plt.savefig(plot_full_path)

# ------------------- SAVE MODEL -------------------
print(f"Saving the model as '{model_name}_{dataset_name}'...")
model_scripted = torch.jit.script(model) # Export to TorchScript

model_relative_path = f'./{model_name}_{dataset_name}.pt'
model_full_path = os.path.join(os.path.dirname(__file__), model_relative_path)
model_scripted.save(model_full_path)

# ------------------- GENERATE TEST JSON -------------------
print(f"Generating the test.json file...")

script_relative_path = f'../scripts/generate_test_json.py'
script_full_path = os.path.join(os.path.dirname(__file__), script_relative_path)
os.system(f'python3 {script_full_path} {model_name} {dataset_name}')

print('Done!')