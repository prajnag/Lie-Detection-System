import numpy as np
import random
from sklearn.metrics import confusion_matrix
import copy
np.random.seed(2)

#picking a random layer and updating the weights by 20% randomly 
def getRandomWeightsLocalSearch(ww, firstlayer, secondLayer, inputDim):
    l = [firstlayer, secondLayer]
    n = random.randint(0, len(l)-1)
    if l[n]==firstlayer:
        if(random.uniform(0,1) > 0.5):
                        sign = 1
        else:
                        sign = -1
        x_temp = random.randint(0, inputDim-1)
        y_temp = random.randint(0, firstlayer-1)
        ww[0][x_temp][y_temp] =ww[0][x_temp][y_temp]+ sign*ww[0][x_temp][y_temp]*0.20
        pass
    #if second layer is randomly chosen
    if l[n]==secondLayer:
        if(random.uniform(0,1) > 0.5):
                        sign = 1
        else:
                        sign = -1
        x_temp = random.randint(0, firstlayer-1)
        y_temp = random.randint(0, secondLayer-1)
        ww[2][x_temp][y_temp] =ww[2][x_temp][y_temp]+ sign*ww[2][x_temp][y_temp]*0.20
    return ww

#for global search, randomly updating all the values in all the layers 
def getRandomWeights(ww, firstlayer, secondLayer, inputDim): #randomises all the weights in every iterstion-global search
    ww[0] = np.random.rand(inputDim, firstlayer)
    ww[1] = np.random.rand(firstlayer)
    ww[2] = np.random.rand(firstlayer, secondLayer)
    ww[3] = np.random.rand(secondLayer)
    return ww


def test_model(model, X_train, y_train):
    y_pred = model.predict(X_train)
     # setting a confidence threshhold of 0.9
    y_pred_labels = list(y_pred > 0.9)
    for i in range(len(y_pred_labels)):
        if int(y_pred_labels[i]) == 1 : y_pred_labels[i] = 1
        else : y_pred_labels[i] = 0
    cm = confusion_matrix(y_train, y_pred_labels) #making a confusion martix out of ypred and ytrain
    accuracy = (cm[0][0] + cm[1][1])/(cm[0][0] + cm[0][1]+ cm[1][0] + cm[1][1])
    return (accuracy, cm)
       

def main():
    from datetime import datetime
    l = []
    def generateColumns(start, end):
        for i in range(start, end+1):
            l.extend([str(i)+'X', str(i)+'Y'])
        return l

    eyes = generateColumns(1, 12)
    # reading in the csv as a dataframe
    import pandas as pd
    df = pd.read_csv('/Users/prajnagirish/Desktop/aiEyes-master/Eyes.csv')
    # selecting the features and target
    X = df[eyes]
    y = df['truth_value']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 32)

    # Data Normalization
    from sklearn.preprocessing import StandardScaler as SC
    sc = SC()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

    # importing the required layers from keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Activation

    # layering up the cnn
    firstlayer = 4
    secondLayer = 1

    model = Sequential()
    model.add(Dense(firstlayer, input_dim = X_train.shape[1])) 
    model.add(Activation('relu'))
    model.add(Dense(secondLayer))
    model.add(Activation('sigmoid'))

    # model compilation
    opt = 'adam'
    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['mean_squared_error'])
    acc = 0
    ww = getRandomWeights(model.get_weights(), firstlayer, secondLayer, X_train.shape[1])
    ww1 = copy.deepcopy(ww) 
    ww1 = getRandomWeightsLocalSearch(ww1, firstlayer, secondLayer, X_train.shape[1])
    while True:
        #global search
        curAcc, cm = test_model(model, X_train, y_train)
        repeat = 0
        while (curAcc<0.95):

            ww1 = copy.deepcopy(ww) 
            if repeat>100:
                #doing 50 mutations
                for _ in range(50):
                    ww1 = getRandomWeightsLocalSearch(ww1, firstlayer, secondLayer, X_train.shape[1])
                    
            for _ in range(10):
                ww1 = getRandomWeightsLocalSearch(ww1, firstlayer, secondLayer, X_train.shape[1])
            #localsearch
            ww1 = getRandomWeightsLocalSearch(ww1, firstlayer, secondLayer, X_train.shape[1])
            model.set_weights(ww1)
            localAcc, cm = test_model(model, X_train, y_train)
            if localAcc >= curAcc:
                if localAcc == curAcc:
                    repeat += 1
                else:
                    repeat = 0
                ww = copy.deepcopy(ww1)
                curAcc = localAcc
                            
            if(curAcc)>=0.84:
                print("Final Accuracy is ", curAcc*100)
                y_pred = model.predict(X_train)
                y_pred_labels = list(y_pred > 0.9)
                for i in range(len(y_pred_labels)):
                    if int(y_pred_labels[i]) == 1 : y_pred_labels[i] = 1
                    else : y_pred_labels[i] = 0
                cm = confusion_matrix(y_train, y_pred_labels)
                print(cm)
                break
        break
if __name__ == "__main__":
    main()