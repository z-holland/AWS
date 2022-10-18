# -*- coding: utf-8 -*-
"""
adult data set project

"""


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import pandas as pd

print("Adult data projext")

print("\nload data sets")

# load adult train data
train_data =  pd.read_csv("adult.data", header=None)
#print(train_data)

# assign columns
train_data.columns = ['age','workclass','fnlwgt','education','education-num','marital-status',  
'occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary']
print(train_data)

# assign output classes
classes = [' <=50K',' >50K'] 

# remove rows with unknown '?'
print("train data with '?:",train_data[(train_data == ' ?')].count(axis=1).sum())
train_data = train_data[~(train_data == ' ?').any(axis=1)]
print("after cleaning:",train_data[(train_data == ' ?')].count(axis=1).sum())

# reindex rows
train_data.reset_index(inplace=True);

# print train data
#print(train_data)
print("train data records: ",len(train_data))

# load adult test data
test_data =  pd.read_csv("adult.test", header=None)
#print(test_data)

# assign columns
test_data.columns = ['age','workclass','fnlwgt','education','education-num','marital-status',  
'occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary']
print(test_data)

# remove rows with unknown '?'
print("train data with '?:",test_data[(test_data == ' ?')].count(axis=0).sum())
test_data = test_data[~(test_data == ' ?').any(axis=1)]
print("after cleaning:",test_data[(test_data == ' ?')].count(axis=0).sum())

# reindex rows
test_data.reset_index(inplace=True);

# print train data
#print(test_data)
print("number testdata records: ",len(test_data))


# apply one-hot encoding to transform data columns 
# workclass,education,marital-status,occupation,relationship,race,sex,native-country
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# make encodees
labelEncoder = LabelEncoder()
oneHotEncoder = OneHotEncoder(handle_unknown='ignore')

####################
# encode train set #
####################

print("\nencode train set")

# encode sex column
print("encoding column: sex")
train_data['sex-code'] = labelEncoder.fit_transform(train_data['sex'])

# drop sex table
train_data.drop('sex', axis=1, inplace=True)

# encode native-country column
print("encoding column: native-country")
train_data['native-country-code'] = labelEncoder.fit_transform(train_data['native-country'])

# drop native-country table
train_data.drop('native-country', axis=1, inplace=True)


# list of columns to encode for one-hot-decoding
encode_columns = ['workclass','education','marital-status','occupation','relationship','race']


# for each column
for column in encode_columns:
    
    # print column
    print("one-hot-encoding column: " + column)
 
    # encode workclass column
    encoding = oneHotEncoder.fit_transform(train_data[[column]])
    # encoding as an array
    encoding = encoding.toarray()
    # make a data frame of encoded results
    encoded = pd.DataFrame(encoding)
    # get encoded column names
    encoded_columns = oneHotEncoder.get_feature_names_out([column])
    # assign encoded column names
    encoded.columns=encoded_columns
    # add encoded columns to original  dataframe
    train_data = train_data.join(encoded)
    # drop original columnh
    train_data.drop(column, axis=1, inplace=True)
    

####################
# encode test set #
####################

print("\nencode test set")


# encode sex column
print("encoding column: sex")
test_data['sex-code'] = labelEncoder.fit_transform(test_data['sex'])

# drop sex table
test_data.drop('sex', axis=1, inplace=True)

# encode native-country column
print("encoding column: native-country")
test_data['native-country-code'] = labelEncoder.fit_transform(test_data['native-country'])

# drop native-country table
test_data.drop('native-country', axis=1, inplace=True)


# list of columns to encode to one-hot-decoding
encode_columns = ['workclass','education','marital-status','occupation','relationship','race']

# for each column
for column in encode_columns:
    
    
    # print column
    print("one-hot-encoding column: " + column)
 
    # encode workclass column
    encoding = oneHotEncoder.fit_transform(test_data[[column]])
    # encoding as an array
    encoding = encoding.toarray()
    # make a data frame of encoded results
    encoded = pd.DataFrame(encoding)
    # get encoded column names
    encoded_columns = oneHotEncoder.get_feature_names_out([column])
    # assign encoded column names
    encoded.columns=encoded_columns
    # add encoded columns to original  dataframe
    test_data = test_data.join(encoded)
    # drop original columnh
    test_data.drop(column, axis=1, inplace=True)


#############################
# descision tree classifier #
#############################

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


print("\nDescision Tree classifier")


# train data X,y
X_train = train_data[train_data.columns.drop('salary')]
y_train = train_data['salary']

# test data  X,y
X_test = test_data[test_data.columns.drop('salary')]
y_test = test_data['salary']


# make decision tree Classifier
dt = tree.DecisionTreeClassifier()

# fit data
dt.fit(X_train, y_train)

# calculte prediction
predicted = dt.predict(X_test);
#print( predicted)

# calculte confusion matrix
cm = confusion_matrix(y_test, predicted)
#print(cm)
tp = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tn = cm[1][1]

# print tp,fp,fn,tn
print('tp:', tp,'fp: ', fp, 'fn:', fn, 'tn:', tn)

# calculate accuracy  (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, predicted)
print ('Accuracy: ', accuracy)

# precision tp / (tp + fp)
precision = tp/(tp+fp)
print('Precision: ', precision)

# recall: tp / (tp + fn)
recall = tp / (tp+fn)
print('Recall: ', recall)

# f1: 2*Precision * Recall / (Prescion + Recall) 
f1 = 2*precision * recall / (precision + recall)
print('F1 score: ', f1)



#############################
# Naive Bayesian Classifier #
#############################


from sklearn.naive_bayes import GaussianNB

print("\nNaive Bayesian Classifier")

# train data X,y
X_train = train_data[train_data.columns.drop('salary')]
y_train = train_data['salary']

# test data  X,y
X_test = test_data[test_data.columns.drop('salary')]
y_test = test_data['salary']


# make naive bayes Classifier
nb = GaussianNB()

# fit train data
nb.fit(X_train, y_train)

# calculate prediction
predicted = nb.predict(X_test);
#print( predicted)

# calculte confusion matrix
cm = confusion_matrix(y_test, predicted)
#print(cm)
tp = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tn = cm[1][1]

print('tp:', tp,'fp: ', fp, 'fn:', fn, 'tn:', tn)

# show accuracy
accuracy = accuracy_score(y_test, predicted)
print ('Accuracy: ', accuracy)

# precision tp / (tp + fp)
precision = tp/(tp+fp)
print('Precision: ', precision)
# recall: tp / (tp + fn)
recall = tp / (tp+fn)
print('Recall: ', recall)
# f1: 2*Precision * Recall / (Prescion + Recall) 
f1 = 2*precision * recall / (precision + recall)
print('F1 score: ', f1)



########################################
# encode numeric data using mean value #
########################################


# encode numeric train set columns using mean value
print("\nencoding train numeric columns on mean")
numeric_columns = ['age','education-num','fnlwgt','capital-gain','capital-loss','hours-per-week']
for column in numeric_columns:
    print(column)
    train_data[column] = (train_data[column] < train_data[column].mean()).astype(int)


# encode numeric columns test set using mean value
print("encoding test numeric columns on mean")
numeric_columns = ['age','education-num','fnlwgt','capital-gain','capital-loss','hours-per-week']
for column in numeric_columns:
    print(column)
    test_data[column] = (test_data[column] < test_data[column].mean()).astype(int)



########################
# K - Means Classifier #
########################


from sklearn.cluster import KMeans


print("\nK-Means Classifier")

# train data X,y
X_train = train_data[train_data.columns.drop('salary')]
y_train = train_data['salary']


# test data  X,y
X_test = test_data[test_data.columns.drop('salary')]
y_test = test_data['salary']


# fit train data for k = 3,5,10
klist = [3,5,10]


# for each k
for k in klist:
    print("k = ",k)
    
    # make KMeans Classifier
    kmeans = KMeans(n_clusters=k)
    
    # fit train data
    kmeans.fit(X_train, y_train)
    
    # centroid cluster centers
    centroids = kmeans.cluster_centers_
    print("centroids")
    print(centroids)
    


##################
# KNN Classifier #
##################


from sklearn.neighbors import KNeighborsClassifier

print("\nKnn Classifier")

# train data X,y
X_train = train_data[train_data.columns.drop('salary')]
y_train = train_data['salary']

# last 10 records of test data  X,y
X_test = test_data[test_data.columns.drop('salary')][-10:]
y_test = test_data['salary'][-10:]


# fit train data for k = 3,5,10
klist = [3,5,10]

# for eack k
for k in klist:
    print("k = ",k)
    
    # make KMeans Classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # fit train data
    knn.fit(X_train, y_train)
    
    # show test case prediction
    predicted = dt.predict(X_test);
    #print( predicted)
    
    # calculte confusion matrix
    cm = confusion_matrix(y_test,predicted)
    #print(cm)
    tp = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[1][1]

    # print tp,fp,fn,tn
    print('tp:', tp,'fp: ', fp, 'fn:', fn, 'tn:', tn)

    # calculate accuracy  (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, predicted)
    print ('Accuracy: ', accuracy)

    # precision tp / (tp + fp)
    precision = tp/(tp+fp)
    print('Precision: ', precision)
    # recall: tp / (tp + fn)
    recall = tp / (tp+fn)
    print('Recall: ', recall)
    # f1: 2*Precision * Recall / (Prescion + Recall) 
    f1 = 2*precision * recall / (precision + recall)
    print('F1 score: ', f1)

##################
# SVM Classifier #
##################


from sklearn import svm

print("\nSVM Classifier")


# train data X,y
X_train = train_data[train_data.columns.drop('salary')]
y_train = train_data['salary']

# test data  X,y
X_test = test_data[test_data.columns.drop('salary')]
y_test = test_data['salary']

# make SVM Classifier
sv = svm.SVC()

# fit train data
sv.fit(X_train, y_train)

# predict test case
predicted = sv.predict(X_test);
#print( predicted)

# calculte confusion matrix
cm = confusion_matrix(y_test, predicted)
#print(cm)
tp = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tn = cm[1][1]

# print tp,fp,fn,tn
print('tp:', tp,'fp: ', fp, 'fn:', fn, 'tn:', tn)

# show accuracy  (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, predicted)
print ('Accuracy: ', accuracy)

# precision tp / (tp + fp)
precision = tp/(tp+fp)
print('Precision: ', precision)
# recall: tp / (tp + fn)
recall = tp / (tp+fn)
print('Recall: ', recall)
# f1: 2*Precision * Recall / (Prescion + Recall) 
f1 = 2*precision * recall / (precision + recall)
print('F1 score: ', f1)


##################
# MLP Classifier #
##################


from sklearn.neural_network import MLPClassifier

print("\nMLP Classifier")

# train data X,y
X_train = train_data[train_data.columns.drop('salary')]
y_train = train_data['salary']

# test data  X,y
X_test = test_data[test_data.columns.drop('salary')]
y_test = test_data['salary']


# make MLP Classifier
mlp = MLPClassifier(random_state=1, max_iter=300)

# fit train data
mlp.fit(X_train, y_train)

# print proabilities
proababilities = mlp.predict_proba(X_test)
print("probabilities")
print(proababilities)

# predict
predicted = mlp.predict(X_test)
#print(predicted)

# calculte confusion matrix
cm = confusion_matrix(y_test, predicted)
#print(cm)
tp = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tn = cm[1][1]

# print tp,fp,fn,tn
print('tp:', tp,'fp: ', fp, 'fn:', fn, 'tn:', tn)

# calculate accuracy  (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, predicted)
print ('Accuracy: ', accuracy)

# precision tp / (tp + fp)
precision = tp/(tp+fp)
print('Precision: ', precision)
# recall: tp / (tp + fn)
recall = tp / (tp+fn)
print('Recall: ', recall)
# f1: 2*Precision * Recall / (Prescion + Recall) 
f1 = 2*precision * recall / (precision + recall)
print('F1 score: ', f1)


############################
# Random Forest Classifier #
############################


from sklearn.ensemble import RandomForestClassifier

print("\nRandom Forest Classifier")


# train data X,y
X_train = train_data[train_data.columns.drop('salary')]
y_train = train_data['salary']

# test data  X,y
X_test = test_data[test_data.columns.drop('salary')]
y_test = test_data['salary']

# make SVM Classifier
rf = RandomForestClassifier(random_state=0)

# fit train data
rf.fit(X_train, y_train)

# predict test case
predicted = rf.predict(X_test);
#print( predicted)

# calculte confusion matrix
cm = confusion_matrix(y_test, predicted)
#print(cm)
tp = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tn = cm[1][1]

# print tp,fp,fn,tn
print('tp:', tp,'fp: ', fp, 'fn:', fn, 'tn:', tn)

# show accuracy  (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, predicted)
print ('Accuracy: ', accuracy)

# precision tp / (tp + fp)
precision = tp/(tp+fp)
print('Precision: ', precision)
# recall: tp / (tp + fn)
recall = tp / (tp+fn)
print('Recall: ', recall)
# f1: 2*Precision * Recall / (Prescion + Recall) 
f1 = 2*precision * recall / (precision + recall)
print('F1 score: ', f1)

