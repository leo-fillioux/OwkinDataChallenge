import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing
import random
import SimpleITK as sitk

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

class FullDataLoader:
    def __init__(self, data_path, train = True, train_dev_split = 0.7, parameters = {}, batch_size = 16):
        self.data_path = data_path                                  # path where the data is stored
        self.train = train                                          # bool value
        self.train_dev_split = train_dev_split                      # percentage of data going in the training set
        self.batch_size = batch_size
        self.parameters = parameters                                # parameters used for normalization of the test set

        # load the clinical data and radiomics from CSV
        self.clinical_data = self.loadClinicalData(parameters)
        self.radiomics = self.loadRadiomics(parameters)

        # load the images
        self.images = self.loadImages()

        # load the labels
        if(self.train):
            self.labels = data = pd.read_csv(self.data_path + '../train_y.csv', index_col = 0)

    def loadClinicalData(self, parameters):
        """
        Load the clinical data CSV and normalize the data
            parameters : information to make sure that the one-hot encoding produces the same columns on the training and test set
            return the clinical data
        """

        data = pd.read_csv(self.data_path + 'features/clinical_data.csv', index_col = 0)
        data['Histology'] = data['Histology'].str.lower()
        data = pd.get_dummies(data, columns=['Histology'], drop_first = True)
        label_encoder = preprocessing.LabelEncoder()
        data['SourceDataset'] = label_encoder.fit_transform(data['SourceDataset'])
        data['age'] = data['age'].fillna((data['age'].median()))
        if(self.train):
            self.parameters['train_columns'] = data.columns
            return data
        else:
            if set(parameters['train_columns']) - set(data.columns):
                FileNotFoundError('There was a difference between the one-hot-encoded values in the training and test set.')
            else:
                return data


    def loadRadiomics(self, parameters):
        """
        Load the radiomics data CSV and normalize the data
            parameters : information on the normalization done on the training set to normalize the test set
            return the radiomics data
        """
        data = pd.read_csv(self.data_path + 'features/radiomics.csv', index_col = 0, header = 2)
        number_nans = np.sum(data.isna().sum())             # check that there are non NaNs
        all_real = data.applymap(np.isreal).all().all()     # and that there are only real values

        if(number_nans == 0 and all_real):
            if(self.train):
                radiomics_min, radiomics_max = data.min(), data.max()
                data = (data - radiomics_min) / (radiomics_max - radiomics_min)
                self.parameters['rad_train_min'] = radiomics_min
                self.parameters['rad_train_max'] = radiomics_max
                return data
            else:
                data = (data - parameters['rad_train_min']) / (parameters['rad_train_max'] - parameters['rad_train_min'])
                return data
        else:
            raise FileNotFoundError('File contained NaNs or an unexpected non real value.')

    def loadImages(self):
        """
        Load the images, multiply the image by the mask to keep only the part relevant to the tumor, normalize the images
            return the 92*92*92 images
        """

        images = []

        # get the min and max values to use for normalization, store the data in images
        max, min = -np.inf, np.inf
        if(self.train):
            for patient in self.clinical_data.index:
                patient = str(patient).zfill(3)
                archive = np.load(self.data_path + 'images/patient_' + patient + '.npz')
                images.append([archive['scan'], archive['mask']])
                max = np.max((max, np.max(np.array(archive['scan']))))
                min = np.min((min, np.min(np.array(archive['scan']))))
                self.parameters['max_images'], self.parameters['min_images'] = max, min

        # once we know the max and min values to use, normalize the image and apply the mask
        for i, data in enumerate(images):
            scan, mask = data
            img = (scan - self.parameters['min_images']) / (self.parameters['max_images'] - self.parameters['min_images'])
            images[i] = img*mask

        images = np.array(images)
        return images


    def augmentData(self, images, radiomics, clinical_data, y_values):
        """
        Given the images, radiomics and clinical data, augment the data by appying a random 3D translation of the image
        and adding random noise to the radiomics and clinical data
            return the new augmented data
        """

        images = list(images)
        for i, data in enumerate(zip(images, radiomics, clinical_data, y_values)):
            img, rad, clin, y = data

            img_cpy = img.copy()

            # apply a random translation ot the image
            translation_matrix = np.random.randint(-10, 10, (3, 1))
            transform = sitk.AffineTransform(3)
            transform.SetTranslation((translation_matrix[0].item(), translation_matrix[1].item(), translation_matrix[2].item()))
            img_cpy = sitk.GetImageFromArray(img_cpy)
            new_itk = sitk.Resample(img_cpy, transform)
            img_new = sitk.GetArrayFromImage(new_itk)

            # add noise to the image, but take the mask beforehand to reapply it after
            mask = (img_new != 0)
            noise = np.random.normal(size = img_new.shape)
            img_new = img_new * noise * mask
            #images.append(img_new)

            # add noise to the radiomics and clinical data
            rad_noise = np.random.uniform(-0.1, 0.1, rad.shape)
            clinical_noise = np.random.uniform(-0.1, 0.1, clin.shape)

            images[i] = img_new
            radiomics[i] = rad * rad_noise
            clinical_data[i] = clin * clinical_noise
        images = np.array(images)
        return images, radiomics, clinical_data, y_values


    def getParameters(self):
        """
        Return the parameters useful to make sure that the test set is normalized in the same way as the training set and has the same columns
        """
        return self.parameters


    def getData(self):
        """
        Return data loaders for the train and dev or for the test datasets
        """
        if(self.train):

            # generate the indices for the train and dev set
            total_size = self.clinical_data.shape[0]
            train_size = int(self.train_dev_split * total_size)
            train_indices = sorted(random.sample(range(0, total_size), train_size))
            dev_indices = [i for i in range(0, total_size) if i not in train_indices]

            # take everything at the train indices
            images_train = np.take(self.images, train_indices, axis = 0)
            radiomics_train = np.array(self.radiomics.iloc[train_indices])
            clinical_data_train = np.array(self.clinical_data.iloc[train_indices])
            y_train = np.take(self.labels, train_indices, axis = 0).values
            patient_id_train = np.take(self.labels, train_indices, axis=0).index

            images_train, radiomics_train, clinical_data_train, y_train = self.augmentData(images_train, radiomics_train, clinical_data_train, y_train)

            # take everything at the dev indices
            images_dev = np.take(self.images, dev_indices, axis = 0)
            radiomics_dev = np.array(self.radiomics.iloc[dev_indices])
            clinical_data_dev = np.array(self.clinical_data.iloc[dev_indices])
            y_dev = np.take(self.labels, dev_indices, axis = 0)
            patient_id_dev = np.take(self.labels, dev_indices, axis=0).index

            training_set = TensorDataset(torch.from_numpy(images_train), torch.from_numpy(radiomics_train),
                                         torch.from_numpy(clinical_data_train), torch.from_numpy(y_train))
            train_loader = DataLoader(training_set, batch_size = self.batch_size, shuffle = False)

            return train_loader, (images_train, radiomics_train, clinical_data_train, y_train, patient_id_train),\
                   (images_dev, radiomics_dev, clinical_data_dev, y_dev, patient_id_dev)

        else:
            # if we want the test dataset
            images_test = self.images
            patient_id = self.radiomics.index
            radiomics_test = np.array(self.radiomics)
            clinical_data_test = np.array(self.clinical_data)
        return (images_test, radiomics_test, clinical_data_test, patient_id)