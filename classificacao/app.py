from datetime import datetime
from sklearn import metrics, preprocessing
from sklearn.ensemble import GradientBoostingClassifier
import os
import time
import numpy as np
import matplotlib.pyplot as plt

def getFeatures(path, filename):
    features = np.loadtxt(os.path.join(path, filename), delimiter=',')
    return features

def getLabels(path, filename):
    encodedLabels = np.loadtxt(os.path.join(path, filename), delimiter=',', dtype=int)
    return encodedLabels

def getEncoderClasses(path, filename):
    encoderClasses = np.loadtxt(os.path.join(path, filename), delimiter=',', dtype=str)
    return encoderClasses

def trainGradientBoosting(trainData, trainLabels):
    print('[INFO] Training the Gradient Boosting model...')
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)  # Exemplo: n_estimators=100, learning_rate=0.1
    startTime = time.time()
    gb_model.fit(trainData, trainLabels)
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Training done in {elapsedTime}s')
    return gb_model

def predictGradientBoosting(gb_model, testData):
    print('[INFO] Predicting...')
    startTime = time.time()
    predictedLabels = gb_model.predict(testData)
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Predicting done in {elapsedTime}s')
    return predictedLabels

def getCurrentFileNameAndDateTime():
    fileName = os.path.basename(__file__).split('.')[0]
    dateTime = datetime.now().strftime('-%d%m%Y-%H%M')
    return fileName + dateTime

def plotConfusionMatrix(encoderClasses, testEncodedLabels, predictedLabels):
    encoder = preprocessing.LabelEncoder()
    encoder.classes_ = encoderClasses
    test = encoder.inverse_transform(testEncodedLabels)
    pred = encoder.inverse_transform(predictedLabels)
    print(f'[INFO] Plotting confusion matrix and accuracy...')
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics.ConfusionMatrixDisplay.from_predictions(test, pred, ax=ax, colorbar=False, cmap=plt.cm.Greens)
    plt.suptitle('Confusion Matrix: ' + getCurrentFileNameAndDateTime(), fontsize=18)
    accuracy = metrics.accuracy_score(testEncodedLabels, predictedLabels) * 100
    plt.title(f'Accuracy: {accuracy}%', fontsize=18, weight='bold')
    plt.savefig('./results/' + getCurrentFileNameAndDateTime(), dpi=300)
    print(f'[INFO] Plotting done!')
    print(f'[INFO] Close the figure window to end the program.')
    plt.show(block=False)
    return accuracy


def main():
    mainStartTime = time.time()
    trainFeaturePath = './features_labels/train/'
    testFeaturePath = './features_labels/test/'
    featureFilename_Hu = 'features.csv'
    featureFilename_LBP = 'features.csv'
    labelFilename = 'labels.csv'
    encoderFilename = 'encoderClasses.csv'
    
    print(f'[INFO] ========= TRAINING PHASE ========= ')
    trainFeatures_Hu = getFeatures(trainFeaturePath, featureFilename_Hu)
    trainFeatures_LBP = getFeatures(trainFeaturePath, featureFilename_LBP)
    trainFeatures_combined = np.concatenate((trainFeatures_Hu, trainFeatures_LBP), axis=1)
    trainEncodedLabels = getLabels(trainFeaturePath, labelFilename)
    gb_model = trainGradientBoosting(trainFeatures_combined, trainEncodedLabels)
    
    print(f'[INFO] =========== TEST PHASE =========== ')
    testFeatures_Hu = getFeatures(testFeaturePath, featureFilename_Hu)
    testFeatures_LBP = getFeatures(testFeaturePath, featureFilename_LBP)
    testFeatures_combined = np.concatenate((testFeatures_Hu, testFeatures_LBP), axis=1)
    testEncodedLabels = getLabels(testFeaturePath, labelFilename)
    encoderClasses = getEncoderClasses(testFeaturePath, encoderFilename)
    predictedLabels = predictGradientBoosting(gb_model, testFeatures_combined)
    elapsedTime = round(time.time() - mainStartTime, 2)

    print(f'[INFO] Code execution time: {elapsedTime}s')
    accuracy = plotConfusionMatrix(encoderClasses, testEncodedLabels, predictedLabels)

    print(f'[INFO] Saving results...')
    with open('./results/results.txt', 'a') as file:
        file.write(f'[{getCurrentFileNameAndDateTime()}] Accuracy: {accuracy}%\n')

    print(f'[INFO] Saving done!')
    print(f'[INFO] Accuracy: {accuracy}%')

if __name__ == "__main__":
    main()