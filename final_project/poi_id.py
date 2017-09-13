#!/usr/bin/python
import pprint
import sys
from time import time
import matplotlib.pyplot as py
import numpy
import pickle
import warnings

##importing the sklearn functions
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from tester import dump_classifier_and_data

warnings.filterwarnings("ignore")
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi']  # You will need to use more features

## all units are in US dollars
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                      'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                      'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

## units are generally number of emails messages; notable exception is 'email_address', which is a text string
email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                  'shared_receipt_with_poi']

features_list = features_list + financial_features + email_features  ## all features in list

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

dataPoints = float(len(my_dataset))  ## count of data points
featureCount = float(len(features_list))  ## count of all features

## calculation of poi's and non-poi's
pois = 0.
nonPois = 0.
for i in my_dataset.values():
    if i["poi"] == 1:
        pois += 1.
    else:
        nonPois += 1.


## detect nan values of each person, with a given percentage
## return a list of detected persons
def nanDetectorForKeys(percentage):
    nans = {}

    for i in my_dataset:  ##loop through persons
        nan = 0.
        for t in my_dataset[i]:  ## loop through values
            if my_dataset[i][t] == 'NaN' or t == '':  ## if value is Nan
                nan += 1  ## add to nan

        if nan / featureCount > float(percentage) / 100:  ## if nan counts are bigger then the given percentage
            nans[i] = nan  ## add it to nans list

    return nans  ##return list


## detect nan values of each feature, with a given percentage
## return a list of detected features
def nanDetectorForValues(percentage):
    nans = {}

    allTogether = {}
    for feature in features_list:  ## loop through all features
        allTogether[feature] = list()  ## set a list for each feature
        for i in my_dataset.values():  ## append all values for each feataure to the dictionary
            (allTogether[feature]).append(i[feature])

    for i in allTogether:  ## loop through all features, and values
        arr = allTogether[i]
        nan = 0.
        for val in arr:  ## for each value
            if val == 'NaN':  ## check if its NaN
                nan += 1.  ##add +1 if it is

        # print nan,dataPoints,percentage,nan/dataPoints > percentage/100
        if nan / dataPoints > percentage / 100:  ## check if nan count is bigger then the given percentage
            nans[i] = nan  ## add to nans list

    return nans  ## return list


## data exploration

print "Number of Data Points : ", dataPoints
print "Number of Poi's : ", pois
print "Number of non-Poi's : ", nonPois
print "Number of missing Poi values : ", dataPoints - (pois + nonPois)
print "Poi percentage : %", round((pois / (nonPois + pois)), 2) * 100  ## poi's percentage for all records
print "Number of Features : ", featureCount

nanValues = nanDetectorForValues(50)
print "Features which at least %50 of their values are Nan : "
print nanValues

nanKeys = nanDetectorForKeys(90)
print "Individuals which at least %90 of their features are Nan : "
print nanKeys


def plot(feature1, feature2):  ## visualising two features
    features = [feature1, feature2]  ## make a list of two
    data = featureFormat(data_dict, features)  ## format given features
    for point in data:
        f1 = point[0]
        f2 = point[1]
        py.scatter(f1, f2)  ## plot scatter with each record
    py.xlabel(feature1)
    py.ylabel(feature2)
    py.show()  ##show the plot


def explore():  ## exploring features with each other in scatter plot
    ##financial
    plot("salary", "bonus")
    plot("deferral_payments", "total_payments")
    plot("loan_advances", "restricted_stock_deferred")
    plot("deferred_income", "expenses")
    plot("exercised_stock_options", "total_stock_value")
    plot("long_term_incentive", "restricted_stock")
    plot("other", "director_fees")
    ##emails
    plot("from_this_person_to_poi", "from_poi_to_this_person")
    plot("shared_receipt_with_poi", "to_messages")
    plot("from_messages", "to_messages")


# explore()

## there seems to be a true outlier which shows up in every scatter that plotted by financial features.
for i in data_dict:
    if data_dict[i]["salary"] not in ['', 'NaN'] and data_dict[i]["salary"] > 1000000:  ## detect the outliers
        print i, data_dict[i]["salary"]  ## print the values

## there are four records, but the interesting one is the TOTAL key. looks like it's the total of all records
## others maybe be valid data points, keeping them

## removing outliers and Nans

for i in nanKeys:
    my_dataset.pop(i)  ##remove the keys with lots of NaN values, which in this case is %90 or more NaN percentage

my_dataset.pop("TOTAL")  ## remove the total outlier


## find the index of given feature
def findIndex(feature):
    index = 0
    for i in features_list:  ## loop through the feature list, until the given feature is matched
        if i == feature:
            break
        index += 1
    return index


## remove email_address feature, since it's a string value and unique for every record
del features_list[findIndex("email_address")]


## features with NaN values are not cleaned, bevause they will be handled in "featureFormat"


# explore() ## explore the data again, after cleaning some records.

## after removing outliers, there are still some records which looks like outliers, but in this case
## they could lead us to the poi's indeed, so i'll leave them as it is.

## calculate fractions of given -related- values for poi frequency
def computeFraction(param1, param2):
    fraction = 0.
    if float(param1) > 0. and float(param2) > 0.:  ## if both values are greater then 0
        if float(param2) == 0. or float(param1) == 0.:
            fraction = 0.
        else:
            fraction = float(param1) / float(param2)

    return fraction


for i in my_dataset:  ## for every data points
    ## add these three new features

    ## frequency for the messages received from a poi
    my_dataset[i]["fraction_from_poi"] = computeFraction(my_dataset[i]["from_poi_to_this_person"],
                                                         my_dataset[i]["from_messages"])
    ## frequency for the messages sent to a poi
    my_dataset[i]["fraction_to_poi"] = computeFraction(my_dataset[i]["from_this_person_to_poi"],
                                                       my_dataset[i]["to_messages"])

    ## frequency for the income other then the regular salary
    try:
        salary = float(my_dataset[i]["salary"])
        totalPayments = float(my_dataset[i]["total_payments"])
        my_dataset[i]["fraction_to_salary"] = computeFraction(totalPayments - salary, totalPayments)

    except:
        my_dataset[i]["fraction_to_salary"] = 0

##add new features to the features_list
features_list = features_list + ["fraction_from_poi", "fraction_to_poi", "fraction_to_salary"]

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

## scale the features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)


##select best features to use in algorithms with KBest
def selectFeatures(nParam):
    kBest = SelectKBest(k=nParam)
    kBest.fit_transform(features, labels)
    kResult = zip(kBest.get_support(), kBest.scores_, features_list[1:])
    return list(sorted(kResult, key=lambda x: x[1], reverse=True))


results5 = selectFeatures(5)
results10 = selectFeatures(10)
results15 = selectFeatures(15)
resultsAll = selectFeatures("all")

pprint.pprint(results5)
pprint.pprint(results10)
pprint.pprint(results15)
pprint.pprint(resultsAll)

## running kBest with different "k" values shows that, the scores are not changing
# it just changes the count of the features to return from the list
# so i'll use the top 10 features

nFeatures = 0

for trueFalse, score, feature in results10:
    if not trueFalse:
        del features_list[findIndex(feature)]
    else:
        nFeatures += 1 ## count of the selected features
        print "| ", feature, " | ", score, " |"

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

randomState = 42  ## random state to be used in classifiers and test split


## returns stored classifiers and their parameters
def createClassifiersAndParams():
    rec = {}  ## store all in this list

    def addToDict(name, clf, params):  ## helper for adding to the list
        ## name : name of the classifier
        ## clf : classifier object
        ## params : parameters object
        rec[name] = {"clf": clf,
                     "params": params}

    ##naive bayes
    addToDict("NaiveBayes", GaussianNB(), {})

    ##support vector machines
    addToDict("SVM", SVC(), {'kernel': ['poly', 'rbf', 'sigmoid'],
                             'cache_size': [7000],
                             'tol': [0.0001, 0.001, 0.005, 0.05],
                             'decision_function_shape': ['ovo', 'ovr'],
                             'random_state': [randomState],
                             'C': [100, 1000, 10000]
                             })

    ##DecisionTree,
    addToDict("DecisionTree", DecisionTreeClassifier(), {'criterion': ['gini', 'entropy'],
                                                         'splitter': ['best', 'random'],
                                                         'min_samples_split': [2, 10, 20],
                                                         'max_depth': [None, 2, 4, 8, 16],
                                                         'min_samples_leaf': [1, 3, 5, 7, 9],
                                                         'max_leaf_nodes': [None, 6, 12, 24],
                                                         'random_state': [randomState]})
    #
    ##AdaBoost
    addToDict("AdaBoost", AdaBoostClassifier(), {'n_estimators': [25, 50, 100],
                                                 'algorithm': ['SAMME', 'SAMME.R'],
                                                 'learning_rate': [.2, .5, 1, 1.4, 2.],
                                                 'random_state': [randomState]})
    ##Random Forest
    addToDict("RandomForest", RandomForestClassifier(), {'n_estimators': [5, 10, 25],
                                                         'criterion': ['gini', 'entropy'],
                                                         'max_features': ['auto', 'sqrt', 'log2', 3, nFeatures],
                                                         'min_samples_split': [2, 10, 20],
                                                         'class_weight': ["balanced_subsample", "balanced"],
                                                         # 'max_depth': [None, 2],
                                                         # 'min_samples_leaf': [1, 3],
                                                         # 'max_leaf_nodes': [None, 2],
                                                         'random_state': [randomState]})

    ##KNeighbors
    addToDict("KNeighbors", KNeighborsClassifier(), {'n_neighbors': [2, 4, nFeatures],
                                                     'weights': ['uniform', 'distance'],
                                                     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                                                     'leaf_size': [10, 25, 50],
                                                     'p': [2, 3, 4]})

    ##KMeans
    addToDict("KMeans", KMeans(), {'n_clusters': [4, 8, 16],
                                   'init': ['k-means++', 'random'],
                                   'max_iter': [50, 150, 300],
                                   'tol': [0.0001, 0.0005, 0.001, 0.005],
                                   'precompute_distances': [True, False],
                                   'random_state': [randomState],
                                   'copy_x': [True, False],
                                   'algorithm': ['full', 'elkan']})

    ##LogisticRegression
    addToDict("LogisticRegression", LogisticRegression(), {'penalty': ['l1', 'l2'],
                                                           'tol': [0.0001, 0.0005, 0.001, 0.005],
                                                           'C': [1, 10, 100, 1000, 10000, 100000, 1000000],
                                                           'fit_intercept': [True, False],
                                                           'solver': ['liblinear'],
                                                           'class_weight': [None, 'balanced'],
                                                           'random_state': [randomState]
                                                           })

    return rec


def train(clf, params, features_train, labels_train):  ## trainer function
    # train
    t0 = time()  ## timer for calculating trainin time
    clft = GridSearchCV(clf, params)  ## grid search with parameters
    clft = clft.fit(features_train, labels_train)  ## training the searcher for best fit

    # print "training time:", round(time() - t0, 3), "s"  ##print the training time
    return clft, (time() - t0)  ## return best parameters


def predict(clf, features_test):  ## predictor function
    # predict
    t0 = time()  ##timer for calculating the prediction time
    pred = clf.predict(features_test)  ## predict the result for test features
    # print "predicting time:", round(time() - t0, 3), "s"  ## print the prediction time
    return pred, (time() - t0)  ## return all predictions


def scores(pred, labels_test):  ## scoring function
    ## inspired from tester script

    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    total_predictions = 0
    accuracy = 0.
    precision = 0.
    recall = 0.
    f1 = 0.
    f2 = 0.

    ## get the prediction details
    for prediction, truth in zip(pred, labels_test):
        if prediction == 0 and truth == 0:
            true_negatives += 1
        elif prediction == 0 and truth == 1:
            false_negatives += 1
        elif prediction == 1 and truth == 0:
            false_positives += 1
        elif prediction == 1 and truth == 1:
            true_positives += 1
        else:
            print "Warning: Found a predicted label not == 0 or 1."
            print "All predictions should take value 0 or 1."
            print "Evaluating performance for processed predictions:"
            break

    try:
        ##calculate each metric
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0 * (true_positives + true_negatives) / total_predictions
        precision = 1.0 * true_positives / (true_positives + false_positives)
        recall = 1.0 * true_positives / (true_positives + false_negatives)
        f1 = 2.0 * true_positives / (2 * true_positives + false_positives + false_negatives)
        f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)

    except:
        print "Got a divide by zero"

    ## print the values
    # print 'accuracy = ', accuracy
    # print 'precision = ', precision
    # print 'recall = ', recall
    # print 'f1 = ', f1
    # print 'f2 = ', f2

    return accuracy, precision, recall, f1, f2  ## return all scores


## function for doing everything in a single area
## gets classifier, their parameters, train and test features
## returns the best-fit classifier, its parameters and the scores
def doTheMath(clf, params, features_train, labels_train, features_test, labels_test):
    clft, trainTime = train(clf, params, features_train, labels_train)  ## call train function with given params
    preds, predictTime = predict(clft, features_test)  ## make predictions with given classifier

    accuracy, precision, recall, f1, f2 = scores(preds, labels_test)  ## calculate the scores

    ## return best-fit classifier, and its parameters , also the score values
    return clft.best_estimator_, clft.best_params_, accuracy, precision, recall, f1, f2, trainTime, predictTime


allClassifiers = createClassifiersAndParams()  ## create the list of classifiers

## format the data with only selected features
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

## split the data to training and testing sets
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

print "Train set length:", len(features_train)
print "Test set length:", len(features_test)

## scale the features
scaler = MinMaxScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.fit_transform(features_test)

for x in allClassifiers:  ##loop through all classifiers
    print x, " running.."

    clft = allClassifiers[x]["clf"]  ##get the classifier
    params = allClassifiers[x]["params"]  ##get the parameters

    ## call the function for all processes
    estimator, params, accuracy, precision, recall, f1, f2, trainTime, predictTime = doTheMath(clft,
                                                                                               params,
                                                                                               features_train,
                                                                                               labels_train,
                                                                                               features_test,
                                                                                               labels_test)

    ## record results in dictionary
    allClassifiers[x]["clf"] = estimator
    allClassifiers[x]["params"] = params
    allClassifiers[x]["accuracy"] = accuracy
    allClassifiers[x]["precision"] = precision
    allClassifiers[x]["recall"] = recall
    allClassifiers[x]["f1"] = f1
    allClassifiers[x]["f2"] = f2
    allClassifiers[x]["trainTime"] = trainTime
    allClassifiers[x]["predictTime"] = predictTime

    ## calculate a new score, with a formula for choosing best classifiers, i made it up.
    ## f1 score, accuracy, precision and recall values are important, but in this case, time is important too.
    ## score = (f1*precision*recall*accuracy) / (total time)
    ## this makes sense to me.
    score = (allClassifiers[x]["f1"] * allClassifiers[x]["precision"] * \
                            allClassifiers[x]["recall"] * allClassifiers[x]["accuracy"]) / \
            (allClassifiers[x]["trainTime"] + allClassifiers[x]["predictTime"])

    ## store new score in dictionary
    allClassifiers[x]["my_score"] = score

    ##printing scores to use in .md
    print "| ", x, " | ", round(score, 4), " | ", round(allClassifiers[x]["accuracy"],3), " | ", round(allClassifiers[x]["precision"],3), \
        " | ", round(allClassifiers[x]["recall"],3), " | ", round(allClassifiers[x]["f1"],3), " | ", round(allClassifiers[x]["f2"],3), " | ", \
        round(allClassifiers[x]["trainTime"],3), " | ", round(allClassifiers[x]["predictTime"],3), " |"




### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

## choosing best classifier
for i in allClassifiers:
    if i in ['NaiveBayes', 'DecisionTree', 'LogisticRegression']:
        print allClassifiers[i]["clf"]


## best 3 algorithm tested and validated with cross validtion using script included in tester.py
## all those steps are runned in tunning.py script, and this returns the best algorithm, which is clearly Naive one :)
from tuning import tuning
clf = tuning()

## test script result are as above
# GaussianNB(priors=None)
# 	Accuracy: 0.82860	Precision: 0.34322	Recall: 0.31250	F1: 0.32714	F2: 0.31820
# 	Total predictions: 15000	True positives:  625	False positives: 1196	False negatives: 1375	True negatives: 11804



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
