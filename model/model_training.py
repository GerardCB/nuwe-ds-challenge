import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wandb

# Iniciar Weights & Biases
wandb.init(project='nuwe-ds-challenge')

# Definir las funciones de entrenamiento y prueba
def train(net, criterion, optimizer, trainloader, epoch):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    wandb.log({'train_loss': running_loss / len(trainloader)})
    print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))

def test(net, criterion, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = net(inputs)
            predicted = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    wandb.log({'test_accuracy': accuracy})
    print('Accuracy: %.2f %%' % accuracy)

# Cargar los datos
data = pd.read_csv('train.csv')

# Preprocesar los datos
X = data.drop('Attrition_Flag', axis=1)
y = data['Attrition_Flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear el modelo
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Definir los hiperparámetros
config = wandb.config
config.learning_rate = 0.001
config.epochs = 100
config.batch_size = 32

# Inicializar el modelo y el optimizador
net = Net()
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)

# Crear los dataloaders
trainset = torch.utils.data.TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
testset = torch.utils.data.TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

# Entrenar y probar el modelo
for epoch in range(config.epochs):
    train(net, criterion, optimizer, trainloader, epoch)
    test(net, criterion, testloader)

# Guardar el modelo entrenado
torch.save(net.state_dict(), 'model.pt')