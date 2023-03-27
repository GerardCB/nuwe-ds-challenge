import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from model_utils import train, test, get_nn_nparams, plot_evaluation, SupplyChainDataset, SmoothedF1Loss
from torch_models import BaselineNet

# ----------------- Hyperparameters -----------------
model_name = 'smoothed_baseline_model_finetuned_2'
dataset_name = 'raw' # one of 'raw', 'cleaned', 'balanced'
assert dataset_name in {'raw', 'cleaned', 'balanced'}, 'Invalid dataset name'

hparams = {
    'batch_size': 32,
    'learning_rate': 0.003727,
    'num_epochs': 50,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.1,
    'weight_decay': 0.0001
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------- Data -----------------
# Load and preprocess the data
print('Creating dataset...')
data_relative_path = f'../data/supply_chain_train_{dataset_name}.csv'
data_path = os.path.join(os.path.dirname(__file__), data_relative_path)
print(f'Loading data from {data_path}')

# create the dataset splits
train_dataset = SupplyChainDataset(data_path, split='train', frac=0.95)
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
criterion = SmoothedF1Loss()  # nn.BCELoss()
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
    train_loss = train(model, criterion, optimizer, train_loader, epoch, DEVICE)
    losses.append(train_loss)

    val_acc, val_f1 = test(model, val_loader, epoch, DEVICE)
    accuracies.append(val_acc)
    f1_scores.append(val_f1)

# ------------------- PLOT RESULTS -------------------
print('Plotting the results...')
_ = plot_evaluation(losses, accuracies, f1_scores, model_name, dataset_name, save=True)

# ------------------- SAVE MODEL -------------------
print(f"Saving the model as '{model_name}_{dataset_name}'...")
model_scripted = torch.jit.script(model)  # export to TorchScript
model_relative_path = f'./{model_name}_{dataset_name}.pt'
model_full_path = os.path.join(os.path.dirname(__file__), model_relative_path)
model_scripted.save(model_full_path)

# ------------------- GENERATE TEST JSON -------------------
print(f"Generating the test.json file...")

script_relative_path = f'../scripts/generate_test_json.py'
script_full_path = os.path.join(os.path.dirname(__file__), script_relative_path)
os.system(f'python3 {script_full_path} {model_name} {dataset_name}')

print('Done!')