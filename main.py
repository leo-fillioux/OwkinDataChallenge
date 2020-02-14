from data_loader import FullDataLoader
from feature_extractor import FeatureExtractor, train, getFeaturesFromTrainedModel
from final_prediction import finalPrediction
from metrics import cindex
import pandas as pd
import numpy as np
import argparse

def main(args):
    # constants
    TRAIN_PATH = '../Data/train_x/'
    TEST_PATH = '../Data/test_x/'
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    LATENT_SIZE = args.latent_size
    EPOCHS = args.epochs
    SILENT = args.silent

    # create the data loaders
    if not SILENT : print('STARTING TO LOAD THE DATA')
    train_dev_full_loader = FullDataLoader(TRAIN_PATH, train = True, batch_size = BATCH_SIZE)
    test_full_loader = FullDataLoader(TEST_PATH, train = False, parameters = train_dev_full_loader.getParameters())

    train_loader, train_data, dev_data = train_dev_full_loader.getData()
    test_loader = test_full_loader.getData()
    if not SILENT : print('DATA FULLY LOADED')

    # train the feature extractor
    if not SILENT : print('BEGINNING TRAINING THE FEATURE EXTRACTOR')
    model = FeatureExtractor

    best_model, best_c_index = train(model = model, train_data = train_loader, dev_data = dev_data, n_epoch = EPOCHS,
                                     learning_rate = LEARNING_RATE, latent_size = LATENT_SIZE)
    if not SILENT : print("TRAINING FINISHED WITH A BEST SCORE OF {:.4f}".format(best_c_index))

    # unpack the train and dev data
    images_train, radiomics_train, clinical_data_train, y_train, patient_id_train = train_data
    images_dev, radiomics_dev, clinical_data_dev, y_dev, patient_id_dev = dev_data

    # extract the features from the train and dev set
    if not SILENT : print("USING THE MODEL TO EXTRACT THE FEATURES FROM THE IMAGES")
    train_features = getFeaturesFromTrainedModel(best_model, images_train)
    dev_features = getFeaturesFromTrainedModel(best_model, images_dev)

    if not SILENT : print('MAKING FINAL PREDICTIONS')
    # make the final prediction on the train and dev sets
    train_prediction, pca, cox_model = finalPrediction(train_features, radiomics_train, clinical_data_train,
                                                       y_train, patient_id_train, pca = None, cox_model = None)
    dev_prediction, pca, cox_model = finalPrediction(dev_features, radiomics_dev, clinical_data_dev,
                                                     y_dev, patient_id_dev, pca, cox_model)

    patient_id_train = patient_id_train.values.reshape((-1, 1))
    y_train = pd.DataFrame((np.hstack((patient_id_train, y_train))))
    y_train.columns = ['PatientID', 'SurvivalTime', 'Event']
    y_train = y_train.set_index(['PatientID'])

    patient_id_dev = patient_id_dev.values.reshape((-1, 1))
    y_dev = pd.DataFrame((np.hstack((patient_id_dev, y_dev))))
    y_dev.columns = ['PatientID', 'SurvivalTime', 'Event']
    y_dev = y_dev.set_index(['PatientID'])

    # get the concordance index on the train and dev sets
    print("CONCORDANCE INDEX ON THE TRAINING SET", cindex(y_train, train_prediction))
    print("CONCORDANCE INDEX ON THE DEV SET", cindex(y_dev, dev_prediction))

    # create the final prediction for the test set
    if args.dont_submit:
        print("CREATING THE SUBMISSION FILE ON THE TEST SET")
        images_test, radiomics_test, clinical_data_test, patient_id_test = test_loader
        test_features = getFeaturesFromTrainedModel(best_model, images_test)
        test_prediction, pca, cox_model = finalPrediction(test_features, radiomics_test, clinical_data_test,
                                                         None, patient_id_test, pca, cox_model)
        test_prediction.to_csv('submission.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-size", type = int, default = 60, help = "Size of the feature vector extracted from the image")
    parser.add_argument("--epochs", type = int, default = 10, help = "Number of training epochs")
    parser.add_argument("--batch-size", type = int, default = 32, help = "Batch size used for training")
    parser.add_argument("--learning-rate", type = float, default = 0.001, help = "Learning rate for the optimizer")
    parser.add_argument("--silent", default = False, action = "store_true", help = "Go in silent mode and only print the results")
    parser.add_argument("--dont-submit", default=False, action="store_true", help = "Do not predict on the test set")
    args = parser.parse_args()
    main(args)