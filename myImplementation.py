import csv
import time
import tracemalloc
import os

import seaborn as sns
from sklearn import metrics
import pandas as pd
import numpy as np
import math
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import accuracy_score


class Tree:
    """ The Tree class contains the all the values that are present on each node and the child nodes data."""
    def __init__(self, val, output, depth):
        self.val = val
        self.children = {}
        self.output = output
        self.depth = depth

    def add_child(self, selected_feature, node):
        self.children[selected_feature] = node


class Decision_tree_classification:
    """This Class contains all the required methods to implement the decision tree"""

    # Initialize the root of the tree to None
    def __init__(self):
        self.__root = None

    # This method takes 1d array as an input and
    # returns a dictionary with keys as unique values and their frequencies as values.
    def get_unique_freq(self, n):
        data_split = {}
        for i in n:
            if i in data_split:
                data_split[i] += 1
            else:
                data_split[i] = 1
        return data_split

    # This method takes the output 1d array and calculates the degree of uncertainty, i.e. entropy
    def get_entropy(self, n):
        get_items_freq = self.get_unique_freq(n)
        entropy = 0
        total = len(n)
        for i in get_items_freq:
            p = get_items_freq[i] / total
            entropy += p * math.log2(p)
        return -1 * entropy

    # This method returns the split ratio value, used to calculate the gain ratio
    def get_split_ratio(self, x, feature_index):
        values = set(x[:, feature_index])
        total_size = np.shape(x)[0]
        d = {}
        split_info = 0
        for i in range(x.shape[0]):
            if x[i][feature_index] not in d:
                d[x[i][feature_index]] = 1
            else:
                d[x[i][feature_index]] += 1
        for i in values:
            split_info += (d[i] / total_size) * math.log2(d[i] / total_size)
        return (-1) * split_info

    # This method is used to calculate the information gain for the given attribute
    def get_information_gain(self, x, y, feature_index):
        tot_info = self.get_entropy(y)
        values = set(x[:, feature_index])
        total_size = np.shape(x)[0]
        data_split = {}
        curr_info = 0
        df = pd.DataFrame(x)
        df[df.shape[1]] = y
        for i in range(x.shape[0]):
            if x[i][feature_index] not in data_split:
                data_split[x[i][feature_index]] = 1
            else:
                data_split[x[i][feature_index]] += 1
        for i in values:
            df1 = df[df[feature_index] == i]
            curr_info += (data_split[i] / total_size) * self.get_entropy(df1[df1.shape[1] - 1])
        return tot_info - curr_info

    # This method uss information gain and split ratio in order to calculate the gain ratio.
    def get_gain_ratio(self, x, y, feature_index):
        info_gain = self.get_information_gain(x, y, feature_index)
        split_ratio = self.get_split_ratio(x, feature_index)
        if split_ratio == 0:
            return math.inf
        else:
            return float(info_gain / split_ratio)

    # This method is used to calculate the gini index, which is then used to calculate the gini max value
    def get_gini_index(self, n):
        get_items_freq = self.get_unique_freq(n)
        gini = 0
        total = len(n)
        for i in get_items_freq:
            p = get_items_freq[i] / total
            gini += p ** 2
        return 1 - gini

    # This method is used to calculate the gini max value for the given attribute
    def get_gini_max(self, x, y, feature_index):
        total_gini_value = self.get_gini_index(y)
        values = set(x[:, feature_index])
        total_size = np.shape(x)[0]
        data_split = {}
        curr_gini = 0
        df = pd.DataFrame(x)
        df[df.shape[1]] = y
        for i in range(x.shape[0]):
            if x[i][feature_index] not in data_split:
                data_split[x[i][feature_index]] = 1
            else:
                data_split[x[i][feature_index]] += 1
        for i in values:
            df1 = df[df[feature_index] == i]
            curr_gini += (data_split[i] / total_size) * self.get_gini_index(df1[df1.shape[1] - 1])
        return total_gini_value - curr_gini

    # This method loops through all the features present for the data set passed and
    # calculates the metrics value for the given metric and returns the best attribute.
    # We use this best metric to split the data set into the child nodes.
    def get_best_selection_attribute(self, d, y, type, feature_list):
        max_value = -math.inf
        best_feature = None
        for i in feature_list:
            # We check if the feature has only 2 unique values and
            # assign the selection metric to max gini, as it is the best in case of multivalued attributes.
            if len(self.get_unique_freq(d[:, i])) == 2 or type == "gini":
                curr_gain = self.get_gini_max(d, y, i)
                if curr_gain > max_value:
                    best_feature = i
                    max_value = curr_gain
            elif type == "gain":
                curr_gain = self.get_gain_ratio(d, y, i)
                if curr_gain > max_value:
                    best_feature = i
                    max_value = curr_gain
            elif type == "infogain":
                curr_gain = self.get_information_gain(d, y, i)
                if curr_gain > max_value:
                    best_feature = i
                    max_value = curr_gain
        return best_feature

    # Decision tree is implemented in this method.
    # If data set belongs to the same output class, we mark the node as leaf and return the node with output class.
    # If there are no more attributes present to divide the data further,
    # mark the node as leaf and output the class that has the highest frequency among the data sets.
    # We then check for the best attribute and split the data using that attribute.
    # Recursively call the decision_tree method for the child nodes and return the current node in the end.
    # Remove the attribute from the feature list as this attribute cannot be used to divide the data further
    # along the depth of the tree.
    def decision_tree(self, d, out_list, metric_type, attribute_list, tree_depth, classes):
        # If the node consists of only one class.
        if len(set(out_list)) == 1:
            output = out_list[0]
            return Tree(None, output, tree_depth)

        # If there are no more features left to classify
        elif len(attribute_list) == 0:
            # print("Reached Leaf node with decision Tree depth = ", tree_depth)
            get_items_freq = self.get_unique_freq(out_list)
            curr_count = -math.inf
            output = None
            for i in classes:
                if i in get_items_freq:
                    frequency = get_items_freq[i]
                    if frequency > curr_count:
                        output = i
                        curr_count = frequency
            return Tree(None, output, tree_depth)

        best_attribute = self.get_best_selection_attribute(d, out_list, metric_type, attribute_list)
        freq_map = self.get_unique_freq(out_list)
        output = None
        max_count = -math.inf
        for i in classes:
            if i in freq_map:
                if freq_map[i] > max_count:
                    output = i
                    max_count = freq_map[i]
        values = set(d[:, best_attribute])
        df_complete = pd.DataFrame(d)
        df_x = pd.DataFrame(d)
        df_complete[df_complete.shape[1]] = out_list
        curr_node = Tree(best_attribute, output, tree_depth)
        index = attribute_list.index(best_attribute)
        attribute_list.remove(best_attribute)

        for i in values:
            dfx = df_x[df_x[best_attribute] == i]
            dfy = df_complete[df_complete[best_attribute] == i]
            node = self.decision_tree(dfx.to_numpy(), (dfy.to_numpy()[:, -1:]).flatten(), metric_type, attribute_list,
                                      tree_depth + 1, classes)
            curr_node.add_child(i, node)

        attribute_list.insert(index, best_attribute)
        return curr_node

    # Preprocessing method is used to append all the attributes, output classes present in the data set
    def preprocess_input_params(self, d, y, metric_type):
        features = [i for i in range(len(d[0]))]
        classes = set(y)
        initial_depth = 0
        self.__root = self.decision_tree(d, y, metric_type, features, initial_depth, classes)

    # This method is used to predict the output values for the given input
    def __predict_for(self, data, node):
        if len(node.children) == 0:
            return node.output
        val = data[node.val]
        if val not in node.children:
            return node.output
        return self.__predict_for(data, node.children[val])

    # This method is used for preprocessing to calculate the predicted output
    def predict(self, d):
        Y = [0 for i in range(len(d))]
        for i in range(len(d)):
            Y[i] = self.__predict_for(d[i], self.__root)
        return np.array(Y)

    # This method is used to calculate the precision of the model. It is scaled to 1.
    def score(self, X, Y):
        count = 0
        for i in range(len(Y)):
            if X[i] == Y[i]:
                count += 1
        return count / len(Y)

    # Print the tree in preorder traversal way.
    def print_tree_only(self, node, spacing=""):
        # Base case: we've reached a leaf
        if len(node.children) == 0:
            print(spacing + "Leaf Node: Attribute_Split: " , str(node.val) , " Tree Depth = ",node.depth , " Label Class: " , str(node.output))
            return

        # Print the Node with the number of children and attribute used to split
        print(spacing + "Regular Node: with ", len(node.children) , " Children and Attribute_Split: " , str(node.val),  " Tree Depth = ",node.depth)

        for i in node.children:
            # Call this function recursively on all the child branches
            print(spacing + '-->Child')
            self.print_tree_only(node.children[i], spacing + "  ")

    def print_tree(self):
        self.print_tree_only(self.__root, "")
# Testing example x = np.array([[0, 0],
#               [0, 1],
#               [1, 0]])
#
# y = np.array([0,
#               1,
#               1])
# x1 =([[1,1]])
# y1=np.array([1])
# print(x1)
# print(y1)
# clf1 = Decision_tree_classification()
# clf1.preprocessing(x, y)
# Y_pred = clf1.predict(x1)
# # print("Predictions :", Y_pred)
# print()
# print("Score :", clf1.score(y1, Y_pred)) # Score on training data
# print()
# clf1.print_tree()

from sklearn import model_selection
def preprocess_dataset(data,datatype):#preprocessing the dataframe before passing it to decision tree algo.
    #check summary of null values in the dataframe

    print(df.isnull().sum())
    #filling missing data with mode
    if datatype and df.isnull().values.any():
        data = data.replace('?', np.NaN)
        data.fillna(df.mode().iloc[0], inplace=True)
        return data

    #filling missing values with mean of each column
    if datatype and df.isnull().values.any():
        data.fillna(data.mean())

    #finding min, max, iqr's of data and splitting the values accordigly
    try:
        for (columnName, columnData) in data.items():
            if is_numeric_dtype(df[columnName]):
                min = df.describe().loc[['min']]
                iqr1 = df.describe().loc[['25%']]
                med = df.describe().loc[['50%']]
                iqr3 = df.describe().loc[['75%']]
                max = df.describe().loc[['max']]
                df[columnName] = pd.cut(x=df[columnName], bins=[min, iqr1, med, iqr3, max], labels=[f'>{min} & <{iqr1}',
                                                                                                    f'>{iqr1} & <{med}',
                                                                                                    f'>{med} & <{iqr3}',
                                                                                                    f'>{iqr3} & <{max}',
                                                                                                    f'>{max}'])
    except:
        print()

    return data



time_elapsed = {}
memory_usage = {}
dir = 'FinalDataSets/Dataset3-breast-cancer-2classes.csv'
df = pd.read_csv(dir)
df.describe()

outfolder = dir.partition('/')[2].partition('-')[0] + '_Predictions'
parent_dir = "Outputs/"
path = os.path.join(parent_dir, outfolder)
if not os.path.exists(path):
    os.mkdir(path)

print(df.isnull().sum())

is_categorical = True
if len(list(set(df.columns) - set(df._get_numeric_data().columns)))==0:
    is_categorical = False

if is_categorical:
    df = df.replace('?', np.NaN)
    df.fillna(df.mode().iloc[0], inplace=True)


cdf=df
preprocess_dataset(cdf,is_categorical)

lst = df.values.tolist()
tick = time.time()#start time
tracemalloc.start()
trainDF_x, testDF_x = model_selection.train_test_split(lst)
trainDF_y =[]
for l in trainDF_x:
    trainDF_y.append(l[-1])
    del l[-1]

testDF_y = []
for lst in testDF_x:
    testDF_y.append(lst[-1])
    del lst[-1]

accuracy_scores = {}

clf2 = Decision_tree_classification()
clf2.preprocess_input_params(np.array(trainDF_x), np.array(trainDF_y).flatten(), "gain")
ourmodel_pred = clf2.predict(np.array(testDF_x))
print("Predictions of our model with gain ratio: ", ourmodel_pred)
print("Accuracy Score of our model with gain ratio metric: {0:0.4f}".format( clf2.score(np.array(testDF_y), ourmodel_pred)))
tock = time.time() # end time
memory_usage['Our Model Memory Usage with GainRatio'] = tracemalloc.get_traced_memory()[0]
tracemalloc.stop()
time_elapsed['Our Model time with Gain Ratio'] = round((tock - tick) * 1000, 2)
accuracy_scores['Our Model Accuracy with Gain Ratio'] = clf2.score(np.array(testDF_y), ourmodel_pred)
print()

tracemalloc.start()
tick = time.time()
clf3 = Decision_tree_classification()
clf3.preprocess_input_params(np.array(trainDF_x), np.array(trainDF_y).flatten(), "gini")
ourmodel_pred_gini = clf3.predict(np.array(testDF_x))
print("Accuracy Score of our model with gini index metric: {0:0.4f}".format( clf3.score(np.array(testDF_y), ourmodel_pred_gini)))
tock = time.time()
memory_usage['Our Model Memory Usage with Gini Index'] = tracemalloc.get_traced_memory()[0]
tracemalloc.stop()
time_elapsed['Our Model time with Gini Index'] = round((tock - tick) * 1000, 2)
accuracy_scores['Our Model Accuracy with Gini Index'] = clf3.score(np.array(testDF_y), ourmodel_pred_gini)

tracemalloc.start()
tick = time.time()
clf5 = Decision_tree_classification()
clf5.preprocess_input_params(np.array(trainDF_x), np.array(trainDF_y).flatten(), "infogain")
ourmodel_pred_infogain = clf5.predict(np.array(testDF_x))
print("Accuracy Score of our model with Information Info Gain: {0:0.4f}".format( clf5.score(np.array(testDF_y), ourmodel_pred_infogain)))
tock = time.time()
memory_usage['Our Model Memory Usage with Info Gain'] = tracemalloc.get_traced_memory()[0]
tracemalloc.stop()
time_elapsed['Our Model time with Info  Gain'] = round((tock - tick) * 1000, 2)
accuracy_scores['Our Model Accuracy with Info Gain'] = clf5.score(np.array(testDF_y), ourmodel_pred_infogain)

if is_categorical:
    tracemalloc.start()
    comp_df = df
    comp_df.fillna(0)
    comp_df = pd.get_dummies(comp_df, drop_first=True)
    #print(comp_df)
    X = comp_df.iloc[:, :-1].values
    Y = comp_df.iloc[:, -1].values.reshape(-1, 1)
    tick = time.time()
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=.3, random_state=41)
    clf4 = tree.DecisionTreeClassifier()
    #print(X_train)
    clf4.fit(X_train, Y_train)
    inbuiltmodel_pred = clf4.predict(X_test)
    tock = time.time()
    print('Accuracy Score of Sklearn Decision Tree Model Categorical: {0:0.4f}'.format( metrics.accuracy_score(inbuiltmodel_pred, Y_test)))
    time_elapsed['SKlearn DT Model time'] = round((tock - tick) * 1000, 2)
    memory_usage['Sklearn DT Model Memory Usage'] = tracemalloc.get_traced_memory()[0]
    accuracy_scores['Sklearn DT Model Accuracy'] = metrics.accuracy_score(inbuiltmodel_pred, Y_test)
    tracemalloc.stop()

    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    tracemalloc.start()
    tick = time.time()
    svc = SVC()
    svc.fit(X_train, Y_train)
    svm_pred = svc.predict(X_test)
    tock = time.time()
    memory_usage['SVM Model Memory Usage'] = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    time_elapsed['SVM Model time'] = round((tock - tick) * 1000, 2)
    accuracy_scores['SVM Model Accuracy'] = accuracy_score(svm_pred, Y_test)
    print('Accuracy Score of Sklearn SVM Model Categorical: {0:0.4f}'.format(accuracy_score(svm_pred, Y_test)))

else:
    # Performing inbuilt DT on same dataset::
    tracemalloc.start()
    tick = time.time()
    clf4 = tree.DecisionTreeClassifier()
    clf4.fit(np.array(trainDF_x), np.array(trainDF_y))
    inbuiltmodel_pred = clf4.predict(np.array(testDF_x))
    tock = time.time()
    memory_usage['Inbuilt DT Model Memory Usage'] = tracemalloc.get_traced_memory()[0]
    time_elapsed['Inbuilt DT Model time'] = round((tock - tick) * 1000, 2)
    tracemalloc.stop()
    accuracy_scores['Sklearn DT Model Accuracy'] = accuracy_score(np.array(testDF_y), inbuiltmodel_pred)
    print('Accuracy Score of Sklearn Decision tree Model Numerical: {0:0.4f}' .format(accuracy_score(np.array(testDF_y), inbuiltmodel_pred)))

    # """""performing SVM on same dataset"""""
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    tracemalloc.start()
    tick = time.time()
    svc = SVC()
    svc.fit(np.array(trainDF_x), np.array(trainDF_y).flatten())
    svm_pred = svc.predict(np.array(testDF_x))
    tock = time.time()
    memory_usage['SVM Model Memory Usage'] = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    time_elapsed['SVM Model time'] = round((tock - tick) * 1000, 2)
    accuracy_scores['SVM Model Accuracy'] = accuracy_score(testDF_y, svm_pred)
    print('Accuracy Score of Sklearn SVM Model Numerical: {0:0.4f}'.format(accuracy_score(testDF_y, svm_pred)))

    #Tensorflow Implementation for numerical data
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers, datasets, models
    import numpy as np
    import pandas as pd
    from sklearn import model_selection, metrics

    ten_df = df
    tick = time.time()
    tracemalloc.start()
    X = ten_df.iloc[:, :-1].values
    Y = ten_df.iloc[:, -1].values.reshape(-1, 1)
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=.2, random_state=41)
    print(len(X_train))
    print(len(Y_train))
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(9, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='softmax')
    ])
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=50, epochs=100)
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)
    tensor_pred = model.predict(X_test)
    tensor_pred = np.argmax(tensor_pred, axis=1)
    y_test = np.argmax(Y_test, axis=1)
    cm = metrics.confusion_matrix(y_test, tensor_pred)
    print(cm)
    print('Accuracy Score of Tensorflow Classification:: {0:0.4f}'.format( accuracy))
    print(metrics.confusion_matrix(y_test, tensor_pred))
    print(metrics.classification_report(y_test, tensor_pred))
    tock = time.time()
    time_elapsed['TensorFlow Model time'] = round((tock - tick) * 1000, 2)
    memory_usage['Tensorflow model memory usage'] = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    accuracy_scores['Tensorflow Model Accuracy'] = accuracy



#Saving predictions generated by the three models
try:
    save_df = pd.DataFrame({"Original Test Data": testDF_y, "OurModel Predictions": ourmodel_pred,
                            "InBuilt DT Predictions:": inbuiltmodel_pred, "SVM Model Predictions": svm_pred})
    save_df.to_csv(f'Outputs/{outfolder}/AllPredictions.csv', index=False)
except:
    print()

table_path = dir.partition('/')[2].partition('.')[0]
with open(f'Outputs/{outfolder}/Tables_{table_path}.csv', 'w') as f:
    data = ["Model", "Time Elapsed(ms)","Memory usage(kb)"]
    writer = csv.writer(f)
    writer.writerow(data)
    for key in time_elapsed.keys() :
        f.write("%s,%s\n" % (key, time_elapsed[key]))
    writer.writerow('')
    data = ["Model", "Memory usage(Kb)"]
    writer = csv.writer(f)
    writer.writerow(data)
    for key in memory_usage.keys():
        f.write("%s,%s\n" % (key, memory_usage[key]*0.001))
    writer.writerow('')
    data = ["Model", "Accuracy"]
    writer = csv.writer(f)
    writer.writerow(data)
    for key in accuracy_scores.keys():
        f.write("%s,%s\n" % (key, accuracy_scores[key]))
    writer.writerow('')

print('---------------ACCURACY SCORES OF EACH MODEL-----------------')
print("Accuracy Score of our model: {0:0.4f}".format( clf2.score(np.array(testDF_y), ourmodel_pred)))
if is_categorical:
    print('Accuracy Score of Sklearn Decision Tree Model Categorical: {0:0.4f}'.format(
        metrics.accuracy_score(inbuiltmodel_pred, Y_test)))
    print('Accuracy Score of Sklearn SVM Model Categorical: {0:0.4f}'.format(accuracy_score(svm_pred, Y_test)))
else:
    print('Accuracy Score of Sklearn Decision tree Model Numerical: {0:0.4f}'.format(
        clf2.score(np.array(testDF_y), inbuiltmodel_pred)))
    print('Accuracy Score of Sklearn SVM Model Numerical: {0:0.4f}'.format(accuracy_score(testDF_y, svm_pred)))
    print('Accuracy Score of Tensorflow Classification:: {0:0.4f}'.format( accuracy))

print('--------------------------------------------------------------')
print()
print('---------------TIME ELAPSED FOR EACH MODEL--------------------')
print(time_elapsed)
print('--------------------------------------------------------------')
print()
print('---------------MEMORY USAGE BY EACH MODEL---------------------')
print(memory_usage)
print('--------------------------------------------------------------')
print()

#if len(set(trainDF_y))==2:
# Confusion matrix only for datasets with only 2 classes
print('---------------CONFUSION MATRIX GENERATED -----------------')
import matplotlib.pyplot as plt

from sklearn import metrics

confusion_matrix = metrics.confusion_matrix(np.array(testDF_y).flatten(), np.array(ourmodel_pred).flatten())
#cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])

labels = set(np.array(testDF_y))
print(labels)
cm_df = pd.DataFrame(confusion_matrix,
                     index = list(labels),
                     columns = list(labels))
plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()
plt.savefig(f'Outputs/{outfolder}/ConfusionMatrix.png')
#plt.show()

from sklearn.metrics import roc_curve, roc_auc_score
try:
    print('---------------ROC Curve GENERATED -----------------')
    false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(testDF_y, ourmodel_pred)
    plt.subplots(1, figsize=(4, 4))
    plt.title('Receiver Operating Characteristic - DecisionTree')
    plt.plot(false_positive_rate1, true_positive_rate1)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(f'Outputs/{outfolder}/ROCCurve.png')
    #plt.show()
except:
    print('ROC CURVE NOT GENERATED')