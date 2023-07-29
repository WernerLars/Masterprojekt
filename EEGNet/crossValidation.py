def crossValidation(Model, inputData, labelData):

    dividedInputData = np.vsplit(inputData,5)
    inputSplit1 = dividedInputData[0]
    inputSplit2 = dividedInputData[1]
    inputSplit3 = dividedInputData[2]
    inputSplit4 = dividedInputData[3]
    inputSplit5 = dividedInputData[4]

    dividedLabelData = np.hsplit(labelData, 5)
    labelSplit1 = dividedLabelData[0]
    labelSplit2 = dividedLabelData[1]
    labelSplit3 = dividedLabelData[2]
    labelSplit4 = dividedLabelData[3]
    labelSplit5 = dividedLabelData[4]

    splitList = [[inputSplit1, labelSplit1],
        [inputSplit2, labelSplit2],
        [inputSplit3, labelSplit3],
        [inputSplit4, labelSplit4],
        [inputSplit5, labelSplit5]]


    inputSplitDictionary =  { 0 : inputSplit1,
                    1 : inputSplit2,
                    2 : inputSplit3,
                    3 : inputSplit4,
                    4 : inputSplit5}

    labelSplitDictionary = {0 : labelSplit1,
                    1 : labelSplit2,
                    2 : labelSplit3,
                    3 : labelSplit4,
                    4 : labelSplit5}
    ValidationAccuracy =[]
    
    print()



    for i in range(len(splitList)):

        trainData = np.zeros((1,4))
        trainLabels = np.zeros((1))
        testTrainingData =np.zeros((1,4))
        testLabelData = np.zeros((1))

        for j in range(len(splitList)):
            if(i == j):
                testTrainingData = inputSplitDictionary[i]
                testLabelData = labelSplitDictionary[i]
            else:
                trainData = np.concatenate((trainData, inputSplitDictionary[i]), axis = 0)
                labelData = np.concatenate((trainLabels, labelSplitDictionary[i]), axis = 0)
        history = Model.fit(testTrainingData, testLabelData, epochs=10,batch_size=16, validation_data=(testTrainingData, testLabelData))
        currValAcc = history.history['val_accuracy']
        currValAccMean = statistics.mean(currValAcc)
        print(currValAcc)
        ValidationAccuracy.append(currValAccMean)
    print(currValAcc[0])
    crossValAcc = statistics.mean(ValidationAccuracy)

    return crossValAcc