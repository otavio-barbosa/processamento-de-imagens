import time
from extract_lbp import extract_lbp
from grayHistogram_FeatureExtraction import encodeLabels, getData, saveData

mainStartTime = time.time()
trainImagePath = './images_split/train/'
testImagePath = './images_split/test/'
trainFeaturePath = './features_labels/train/'
testFeaturePath = './features_labels/test/'

print(f'[INFO] ========= TRAINING IMAGES ========= ')
trainImages, trainLabels = getData(trainImagePath)
trainEncodedLabels, encoderClasses = encodeLabels(trainLabels)
trainFeatures = extract_lbp(trainImages)
saveData(trainFeaturePath, trainEncodedLabels, trainFeatures, encoderClasses)

print(f'[INFO] =========== TEST IMAGES =========== ')
testImages, testLabels = getData(testImagePath)
testEncodedLabels, encoderClasses = encodeLabels(testLabels)
testFeatures = extract_lbp(testImages)
saveData(testFeaturePath, testEncodedLabels, testFeatures, encoderClasses)
elapsedTime = round(time.time() - mainStartTime, 2)
print(f'[INFO] Code execution time: {elapsedTime}s')