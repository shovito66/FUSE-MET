#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from matplotlib import pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam
from torch.utils.data import DataLoader, TensorDataset

from .utils import rearrange_and_concat_data


def print_metrics(Y_plain, Y_pred_plain, modelname=None):
    """
    Prints classification results in different metrics. For now, it only prints accuracy. Eventually, it will print f-1 scores, recall, precision, etc.
    :param Y_plain:
    :param Y_pred_plain:
    :return:
    """
    #print('True:', Y_plain)
    #print('Pred:', Y_pred_plain)

    acc = np.count_nonzero(Y_pred_plain == Y_plain) / len(Y_plain)
    pr, rec, f1, sup = precision_recall_fscore_support(Y_plain, Y_pred_plain, average='macro')
    if modelname is not None:
        print(f'Model:{modelname}, Accuracy: {acc}, Precision: {pr}, Recall: {rec}, F1: {f1}')
    else:
        print(f'Accuracy: {acc}, Precision: {pr}, Recall: {rec}, F1: {f1}')
    return acc, pr, rec, f1


def train_and_test_common(clf, Xtrain, Ytrain, Xtest, Ytest, modelname=None):
    """
    This is a common function to train and test different sorts of classifiers,
    such as random forest, nearest neighbor, SVM, etc.
    :param clf:
    :param Xtrain:
    :param Ytrain:
    :param Xtest:
    :param Ytest:
    :return:
    """
    clf.fit(Xtrain, Ytrain)
    Y_pred = clf.predict(Xtest)
    # print(Y_pred.shape, Ytest.shape)

    Y_pred_plain = np.argmax(Y_pred, axis=1)
    Y_plain = np.argmax(Ytest, axis=1)
    # print(Y_pred[:5])
    # print(Ytest[:5])
    # print(Y_pred_plain[:5])
    acc, pr, rec, f1 = print_metrics(Y_plain, Y_pred_plain, modelname)
    # acc, pr, rec, f1 = print_metrics(Ytest, Y_pred)
    return acc, pr, rec, f1, Y_pred_plain


def train_and_test_3NN(Xtrain, Ytrain, Xtest, Ytest):
    print('Training and testing 3NN')
    clf = KNeighborsClassifier(n_neighbors=3)
    acc, pr, rec, f1, Y_pred = train_and_test_common(clf, Xtrain, Ytrain, Xtest, Ytest, '3NN')
    return acc, pr, rec, f1, Y_pred


def train_and_test_1NN(Xtrain, Ytrain, Xtest, Ytest):
    print('Training and testing 1NN')
    clf = KNeighborsClassifier(n_neighbors=1)
    acc, pr, rec, f1, Y_pred = train_and_test_common(clf, Xtrain, Ytrain, Xtest, Ytest, '1NN')
    return acc, pr, rec, f1, Y_pred


def train_and_test_random_forest(Xtrain, Ytrain, Xtest, Ytest):
    print('Training and testing random forest')
    clf = RandomForestClassifier(max_depth=10, random_state=0)
    acc, pr, rec, f1, Y_pred = train_and_test_common(clf, Xtrain, Ytrain, Xtest, Ytest, 'Random Forest')
    return acc, pr, rec, f1, Y_pred


def train_and_test_svm(Xtrain, Ytrain, Xtest, Ytest):
    print('Training and testing SVM')
    clf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1)
    clf.fit(Xtrain, np.argmax(Ytrain, axis=1))
    Y_plain = np.argmax(Ytest, axis=1)
    Y_pred_plain = clf.predict(Xtest)
    acc, pr, rec, f1 = print_metrics(Y_plain, Y_pred_plain, 'SVM')
    # clf.fit(Xtrain, Ytrain)
    # Y_pred_plain = clf.predict(Xtest)
    # acc, pr, rec, f1 = print_metrics(Ytest, Y_pred_plain)
    return acc, pr, rec, f1, Y_pred_plain


def train_and_test_neuralnet(Xtrain, Ytrain, Xtest, Ytest, outputpath, K, llambda, acc_filename=None,
                             loss_filename=None):
    """
    This function will train and test a neural network.
    :param Xtrain:
    :param Ytrain:
    :param Xtest:
    :param Ytest:
    :param method:
    :return:
    """
    print(Ytrain.shape)
    print('Training and testing neuralnet')
    model = Sequential()
    model.add(Input(shape=(Xtrain.shape[1],)))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(Ytrain.shape[1], activation='softmax'))

    adam_optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=adam_optimizer)  #do not write val_accuracy here. That creates frustating error: 'str' object not callable
    history = model.fit(Xtrain, Ytrain, epochs=40, batch_size=32, validation_split=0.2, verbose=1)

    Y_pred = model.predict(Xtest)
    Y_pred_plain = np.argmax(Y_pred, axis=1)
    Y_plain = np.argmax(Ytest, axis=1)
    acc, pr, rec, f1 = print_metrics(Y_plain, Y_pred_plain, 'Neural Net')
    # acc, pr, rec, f1 = print_metrics(Ytest, Y_pred)
    plot_training_curve(history, outputpath, K, llambda, acc_filename, loss_filename, modelname='NN')
    return acc, pr, rec, f1, Y_pred_plain


def train_and_test_xgboost(Xtrain, Ytrain, Xtest, Ytest):
    # Convert the labels to the correct format for XGBoost
    Ytrain_plain = np.argmax(Ytrain, axis=1)
    Ytest_plain = np.argmax(Ytest, axis=1)
    dtrain = xgb.DMatrix(Xtrain, label=Ytrain_plain)
    dtest = xgb.DMatrix(Xtest, label=Ytest_plain)

    # Set XGBoost parameters
    params = {
        'objective': 'multi:softmax',  # Multi-class classification
        'num_class': Ytrain.shape[1],  # Number of classes
        'max_depth': 6,  # Maximum tree depth
        'eta': 0.3,  # Learning rate
        'verbosity': 1
    }

    # Train the model
    num_round = 100  # Number of boosting rounds
    bst = xgb.train(params, dtrain, num_round)

    # Predict
    Y_pred_plain = bst.predict(dtest)

    acc, pr, rec, f1 = print_metrics(Ytest_plain, Y_pred_plain, 'XGBoost')
    return acc, pr, rec, f1, Y_pred_plain


def train_and_test_tabnet(Xtrain, Ytrain, Xtest, Ytest, epochs=100):
    clf = TabNetClassifier(verbose=1, seed=42)

    Ytrain_plain = np.argmax(Ytrain, axis=1)
    Ytest_plain = np.argmax(Ytest, axis=1)
    # Train the model
    clf.fit(
        Xtrain, Ytrain_plain,
        max_epochs=epochs,
        patience=10,
        batch_size=256,
        virtual_batch_size=128
    )
    # Predict
    Y_pred_plain = clf.predict(Xtest)

    acc, pr, rec, f1 = print_metrics(Ytest_plain, Y_pred_plain, 'TabNet')
    return acc, pr, rec, f1, Y_pred_plain


class DANet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units):
        super(DANet, self).__init__()
        layers = []
        in_dim = input_dim

        for hidden in hidden_units:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden))
            in_dim = hidden

        layers.append(nn.Linear(in_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_and_test_danets(Xtrain, Ytrain, Xtest, Ytest):
    Ytrain_plain = np.argmax(Ytrain, axis=1)
    Ytest_plain = np.argmax(Ytest, axis=1)

    # Convert to PyTorch tensors
    Xtrain = torch.tensor(Xtrain, dtype=torch.float32)
    Ytrain_plain = torch.tensor(Ytrain_plain, dtype=torch.long)
    Xtest = torch.tensor(Xtest, dtype=torch.float32)
    Ytest_plain = torch.tensor(Ytest_plain, dtype=torch.long)
    # Create DataLoader
    batch_size = 64

    train_dataset = TensorDataset(Xtrain, Ytrain_plain)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # Model parameters
    input_dim = Xtrain.shape[1]
    output_dim = Ytrain.shape[1]
    hidden_units = [256, 128, 64]

    # Initialize the DANet model
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  # MPS for Apple M1
    print(f'Using device: {device}')
    model = DANet(input_dim, output_dim, hidden_units).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for rows, labels in train_loader:
            rows, labels = rows.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(rows)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_rows = Xtest.to(device)
        test_labels = Ytest_plain.to(device)
        outputs = model(test_rows)
        _, Y_pred_plain = torch.max(outputs, 1)
        #accuracy = accuracy_score(test_labels.cpu(), Y_pred_plain.cpu())
        #print(f'Accuracy: {accuracy * 100:.2f}%')

        acc, pr, rec, f1 = print_metrics(test_labels.cpu(), Y_pred_plain.cpu(), modelname='DANet')

    return acc, pr, rec, f1, Y_pred_plain.cpu()


class CNNBranch(nn.Module):
    def __init__(self, input_dim):
        super(CNNBranch, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x1 = self.pool1(self.conv1(x))
        x1 = self.flatten(x1)
        x2 = self.pool2(self.conv2(x))
        x2 = self.flatten(x2)
        return torch.cat((x1, x2), dim=1)


class DNNBranch(nn.Module):
    def __init__(self, input_dim):
        super(DNNBranch, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


class History:
    def __init__(self):
        self.history = {
            'accuracy': [],
            'val_accuracy': [],
            'loss': [],
            'val_loss': []
        }


class FusionModel(nn.Module):
    def __init__(self, dnn_input_dim, cnn_input_dim, num_classes):
        super(FusionModel, self).__init__()
        self.dnn_branch = DNNBranch(dnn_input_dim)
        self.cnn_branch = CNNBranch(cnn_input_dim)
        self.fc1 = nn.Linear(128 + 128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x_dnn, x_cnn):
        cnn_output = self.cnn_branch(x_cnn)
        dnn_output = self.dnn_branch(x_dnn)
        combined = torch.cat((cnn_output, dnn_output), dim=1)
        x = torch.relu(self.fc1(combined))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# class ConvNet(nn.Module):
#     def __init__(self, input_dim, num_classes, window_size=5):
#         print(f"Input dim: {input_dim}")
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=window_size, out_channels=64, kernel_size=3)
#         self.bn1 = nn.BatchNorm1d(64)  # BatchNorm after the first conv layer
#         self.pool1 = nn.MaxPool1d(kernel_size=2)
#
#         self.conv2 = nn.Conv1d(in_channels=window_size, out_channels=128, kernel_size=5)
#         self.bn2 = nn.BatchNorm1d(128)  # BatchNorm after the 2nd conv layer
#         self.pool2 = nn.MaxPool1d(kernel_size=2)
#
#         self.flatten = nn.Flatten()
#
#         self.fc1 = nn.Linear(64 * (input_dim // 2), 128)
#         self.dropout_1 = nn.Dropout(p=0.5)
#         self.fc2 = nn.Linear(128, 64)
#         self.dropout_2 = nn.Dropout(p=0.3)
#         self.fc3 = nn.Linear(64, num_classes)
#
#     def forward(self, x):
#         print(f"Input shape: {x.shape}")
#         # print(x.unsqueeze(1).shape)
#         # x = x.unsqueeze(1)
#         print(f"Input shape after unsqueeze: {x.shape}")
#         x1 = self.pool1(self.bn1(self.conv1(x)))  # Apply BatchNorm and then MaxPool
#         x1 = self.flatten(x1)
#
#         x2 = self.pool2(self.bn2(self.conv2(x)))  # Apply BatchNorm and then MaxPool
#         x2 = self.flatten(x2)
#
#         x = torch.cat((x1, x2), dim=1)
#
#         x = torch.relu(self.fc1(x))
#         x = self.dropout_1(x)
#         x = torch.relu(self.fc2(x))
#         x = self.dropout_2(x)
#         x = self.fc3(x)
#         return x

class ConvNet(nn.Module):
    def __init__(self, input_dim, num_classes, window_size):
        super(ConvNet, self).__init__()
        # print(f"Input dim: {input_dim}, Window size: {window_size}")
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # self.conv3 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=5, padding=1)
        # self.bn3 = nn.BatchNorm1d(128)
        # self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.flatten = nn.Flatten()

        conv_output_dim = (window_size // 2) // 2  # Adjust for pooling and kernel sizes
        # conv_output_dim_3 = (window_size // 2)  # for Conv3
        # print(f"Conv output dim: {conv_output_dim}, Conv output dim 3: {conv_output_dim_3}")

        self.fc1 = nn.Linear(128 * conv_output_dim, 128)
        # self.fc1 = nn.Linear(128 + 128, 128)
        self.dropout_1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout_2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, channels, length)
        # print(f"Input shape after permute: {x.shape}")

        # # branch 1
        # y = self.pool3(self.bn3(self.conv3(x)))
        # y = self.flatten(y)

        # branch 2
        x = self.pool1(self.bn1(self.conv1(x)))
        x = self.pool2(self.bn2(self.conv2(x)))
        x = self.flatten(x)

        # z = torch.cat((x, y), dim=1)
        # print(f"Shape after concat: {z.shape}, x: {x.shape}, y: {y.shape}")
        # x = torch.relu(self.fc1(z))

        x = torch.relu(self.fc1(x))
        x = self.dropout_1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout_2(x)
        x = self.fc3(x)
        return x


def train_and_test_ConvNet(Xtrain, Ytrain, Xtest, Ytest, outputpath, K, llambda,
                           num_epochs=40, acc_filename=None, loss_filename=None):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  # MPS for Apple M1
    print(f'Using device: {device}')

    Xtrain = rearrange_and_concat_data(Xtrain)
    Xtest = rearrange_and_concat_data(Xtest)

    Ytrain_plain = np.argmax(Ytrain, axis=1)
    Ytest_plain = np.argmax(Ytest, axis=1)

    # Convert to PyTorch tensors
    Xtrain = torch.tensor(Xtrain, dtype=torch.float32).to(device)
    Ytrain_plain = torch.tensor(Ytrain_plain, dtype=torch.long).to(device)
    Xtest = torch.tensor(Xtest, dtype=torch.float32).to(device)
    Ytest_plain = torch.tensor(Ytest_plain, dtype=torch.long).to(device)

    # Create DataLoader
    batch_size = 64
    train_dataset = TensorDataset(Xtrain, Ytrain_plain)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    test_dataset = TensorDataset(Xtest, Ytest_plain)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Model parameters
    input_dim = Xtrain.shape[2]  # number of features
    window_size = Xtrain.shape[1]  # window size
    output_dim = Ytrain.shape[1]

    # Initialize the ConvNet model
    # model = ConvNet(input_dim, num_classes=output_dim).to(device)
    model = ConvNet(input_dim, num_classes=output_dim, window_size=window_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    history = History()
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for rows, labels in train_loader:
            rows, labels = rows.to(device), labels.to(device)  #rows => input
            optimizer.zero_grad()
            outputs = model(rows)  # Add a channel dimension
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = correct_train / total_train
        train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        with torch.no_grad():
            test_rows = Xtest.to(device)
            test_labels = Ytest_plain.to(device)
            outputs = model(test_rows)
            loss = criterion(outputs, test_labels)
            _, Y_pred_plain = torch.max(outputs, 1)
            val_loss += loss.item()
            total_val += test_labels.size(0)
            correct_val += (Y_pred_plain == test_labels).sum().item()
            # acc, pr, rec, f1 = print_metrics(test_labels.cpu(), Y_pred_plain.cpu())

        val_accuracy = correct_val / total_val
        val_loss = val_loss / len(test_loader)

        # Store history
        history.history['accuracy'].append(train_accuracy)
        history.history['val_accuracy'].append(val_accuracy)
        history.history['loss'].append(train_loss)
        history.history['val_loss'].append(val_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')

        model.train()  # Switch back to training mode

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_rows = Xtest.to(device)
        test_labels = Ytest_plain.to(device)
        outputs = model(test_rows)
        _, Y_pred_plain = torch.max(outputs, 1)
        acc, pr, rec, f1 = print_metrics(test_labels.cpu(), Y_pred_plain.cpu(), modelname='CNN')

    plot_training_curve(history, outputpath, K, llambda, acc_filename, loss_filename, modelname='CNN')
    return acc, pr, rec, f1, Y_pred_plain.cpu()


def plot_training_curve(history, outputpath, K, llambda, acc_filename=None, loss_fielname=None, modelname='nn'):
    indices = np.arange(1, len(history.history['accuracy']) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(indices, history.history['accuracy'])
    plt.plot(indices, history.history['val_accuracy'])
    plt.title(f'Model Accuracy, K={K}, lambda = {llambda}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train acc', 'Validation acc'])
    if acc_filename is not None:
        plt.savefig(f'{outputpath}/{acc_filename}_{modelname}.png')
    else:
        plt.savefig(f'{outputpath}accuracy_K_{K}_lambda_{llambda}_{modelname}.png')

    plt.figure(figsize=(12, 6))
    plt.plot(indices, history.history['loss'])
    plt.plot(indices, history.history['val_loss'])
    plt.title(f'Model Loss, K={K}, lambda = {llambda}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train loss', 'Validation loss'])
    if loss_fielname is not None:
        plt.savefig(f'{outputpath}/{loss_fielname}_{modelname}.png')
    else:
        plt.savefig(f'{outputpath}loss_K_{K}_lambda_{llambda}_{modelname}.png')
