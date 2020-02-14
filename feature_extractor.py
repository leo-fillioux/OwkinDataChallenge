import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from metrics import cindex

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

# define the 3D CNN model that will be used to extract the features
class FeatureExtractor(nn.Module):
    def __init__(self, latent_size):
        super(FeatureExtractor, self).__init__()

        self.conv_layer_1 = nn.Conv3d(1, 2, kernel_size = (3, 3, 3), padding = 1)
        self.batch_norm_1 = nn.BatchNorm3d(2)
        self.max_pool_1 = nn.MaxPool3d((2, 2, 2))

        self.conv_layer_2 = nn.Conv3d(2, 4, kernel_size=(3, 3, 3), padding=1)
        self.batch_norm_2 = nn.BatchNorm3d(4)
        self.max_pool_2 = nn.MaxPool3d((2, 2, 2))

        self.conv_layer_3 = nn.Conv3d(4, 8, kernel_size=(3, 3, 3), padding=1)
        self.batch_norm_3 = nn.BatchNorm3d(8)
        self.max_pool_3 = nn.MaxPool3d((2, 2, 2))

        self.conv_layer_4 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=1)
        self.batch_norm_4 = nn.BatchNorm3d(16)
        self.max_pool_4 = nn.MaxPool3d((2, 2, 2))

        self.fc1 = nn.Linear(16*5*5*5, 1024)
        self.fc2 = nn.Linear(1024, latent_size)
        self.fc3 = nn.Linear(latent_size, 1)

    def forward(self, images):
        x = self.conv_layer_1(images.float())
        x = F.relu(self.batch_norm_1(x))
        x = self.max_pool_1(x)

        x = self.conv_layer_2(x)
        x = F.relu(self.batch_norm_2(x))
        x = self.max_pool_2(x)

        x = self.conv_layer_3(x)
        x = F.relu(self.batch_norm_3(x))
        x = self.max_pool_3(x)

        x = self.conv_layer_4(x)
        x = F.relu(self.batch_norm_4(x))
        x = self.max_pool_4(x)

        x = x.view(x.shape[0], -1)

        x = torch.sigmoid(self.fc1(x))
        latent = F.relu(self.fc2(x))
        x = F.relu(self.fc3(latent))

        return x, latent

def evaluateOnDev(model, data):
    """
    Evaluate the current model on the dev set
        parameters : model to test and dev data
        return the concordance index using the provided metrics
    """
    model.eval()
    images, radiomics, clinical_data, y_dev, patient_id = data

    images = torch.from_numpy(images).unsqueeze_(1)

    # evaluate the CNN on the images and get just the prediction"
    prediction, _ = model.forward(images)
    prediction = prediction.detach().numpy()
    nans = np.nan * np.ones(patient_id.shape)

    # prepare the DataFrame that will be used to generate the concordance index
    submission = pd.DataFrame(np.vstack((patient_id, prediction[:, 0], nans)).T)
    submission.columns = ['PatientID', 'SurvivalTime', 'Event']
    submission = submission.set_index(['PatientID'])
    model.train()
    if(submission.SurvivalTime > 0).all():
        return cindex(y_dev, submission)
    else:
        return 0

def train(model, train_data, dev_data, n_epoch, learning_rate, latent_size, verbose = True):
    """
    Train the model
        parameters : model, training set, dev set, number of epochs, learning rate, size of the second-to-last layer of the model
        return the best performing model on the dev set and the corresponding concordance index
    """
    # define the model, criterion and optimizer
    model = model(latent_size).cpu()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-5)

    train_error = []
    model.train()

    # this will be used to track the model giving the best c index on the dev set, and return this one
    best_c_index = 0
    best_model = None

    # loop of the epochs and the batches
    for epoch in range(n_epoch):
        epoch_average_loss = 0
        for (batch_x_images, batch_x_radiomics, batch_x_clinical, batch_y) in train_data:
            optimizer.zero_grad()
            survival_time, censor = batch_y[:, 0], batch_y[:, 1]
            batch_x_images.unsqueeze_(1) # this is used to add the 1 channel information to the image
            survival_time.unsqueeze_(-1)
            prediction, _ = model.forward(batch_x_images)
            loss = criterion(prediction.float(), survival_time.float())
            loss.backward()
            optimizer.step()
            epoch_average_loss += loss.item() * batch_x_images.shape[0] / len(train_data)
        c_index_dev = evaluateOnDev(model, dev_data)
        if verbose : print('\tC-index on the dev set for epoch {} : {:.4f}'.format(epoch+1, c_index_dev))
        # if the model is better than the current best one, remember this one
        if(c_index_dev > best_c_index):
            best_c_index = c_index_dev
            best_model = deepcopy(model)
        train_error.append(epoch_average_loss)

    # return the best model, and the corresponding concordance index
    return best_model, best_c_index

def getFeaturesFromTrainedModel(model, images):
    """
    Extract the features from the images using the model that has already been trained
        parameters : model, images
        return the features extracted from the image
    """
    images = torch.from_numpy(images)
    images.unsqueeze_(1)
    _, features = model(images)
    return features

def saveModel(model, c_index, name):
    """
    Save the feature extraction model
        parameters : model to be saved, corresponding concordance index, model name
    """
    save_path = '../Models/model_' + name + '_' + str(c_index)
    torch.save(model, save_path)
    print('Model saved as model_' + name + '_' + str(c_index))


def loadModel(name):
    """
    Load a model
        parameters : name of the model to be loaded
        return the loaded model
    """
    load_path = '../Models/' + name
    model = torch.load(load_path)
    model.eval()
    print('Loaded model ' + name)
    return model