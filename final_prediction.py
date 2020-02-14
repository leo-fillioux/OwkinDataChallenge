import numpy as np
from sklearn.decomposition import PCA
from lifelines import CoxPHFitter
import pandas as pd

def applyPCA(image_features, radiomics, clinical_data, pca = None, pca_order = 80):
    """
    Apply the PCA to make sure that the features are all linearly independent
        parameters : features, PCA if applying the one fitted on train to apply on dev, order of the PCA
        return the transformed data and the PCA model
    """
    X = np.hstack((image_features.detach().numpy(), radiomics, clinical_data))

    # if there is already a pca given (i.e. we are not on the train set)
    if pca:
        X = pca.transform(X)

    # otherwise (i.e. we are on the train set), define the PCA, train it and apply it
    else:
        pca = PCA(n_components=pca_order)
        X = pca.fit_transform(X)
    return X, pca

def finalPrediction(image_features, radiomics, clinical_data, y, patient_id, pca = None, cox_model = None):
    """
    Apply the PCA and the cox model to the features extracted from the image and the other features
        parameters : features, y, patient id, PCA and CoxPH models if we are applying on the dev and train set
        return the submission as well as the PCA and CoxPH models
    """
    # apply the PCA to the data
    x, pca = applyPCA(image_features, radiomics, clinical_data, pca = None)

    # if the cox model is not given, fit it on the (x, y) pair (i.e. we are on the train set)
    if not cox_model:
        size = x.shape[1] + y.shape[1]
        final_data = pd.DataFrame(data=np.hstack((x, y)), columns=['col_' + str(i) for i in range(size)])
        cox_model = CoxPHFitter()
        cox_model.fit(final_data, duration_col='col_' + str(size - 2), event_col='col_' + str(size - 1), step_size=0.6)

    # then predict using the model
    size = x.shape[1]
    final_data = pd.DataFrame(data=x, columns=['col_' + str(i) for i in range(size)])
    prediction = cox_model.predict_expectation(final_data).values[:, 0]

    # put the prediction in a pandas DataFrame to submit or evaluate on the concoardance index
    nans = np.nan * np.ones(patient_id.shape)
    submission = pd.DataFrame(np.vstack((patient_id, prediction, nans)).T)
    submission.columns = ['PatientID', 'SurvivalTime', 'Event']
    submission = submission.set_index(['PatientID'])

    # return the submission as well as both model, that might be used on the dev or test set
    return submission, pca, cox_model