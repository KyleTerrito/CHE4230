import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from source.data_preprocess import DataProcessing

class ModelBuilder(DataProcessing):
    def __init__(self, *args, **kwargs):
        super(ModelBuilder, self).__init__(*args, **kwargs)

    def ann(self, X_train,X_test,y_train,y_test):

        clf = MLPClassifier(hidden_layer_sizes=(20,2), learning_rate_init=0.01,max_iter= 500, random_state= 42)
        
        clf.fit(X_train,y_train)
        
        ann_predicted = clf.predict(X_test)

        error = 0
        for i in range(len(y_test)):
            error += np.sum(ann_predicted != y_test)

        total_accuracy = 1 - error / len(y_test)

        self.accuracy = accuracy_score(y_test,ann_predicted)

        return clf