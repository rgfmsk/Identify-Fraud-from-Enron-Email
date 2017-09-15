## detect nan values of each person, with a given percentage
## return a list of detected persons
import sys
import operator
import numpy as np
import matplotlib.pyplot as py

sys.path.append("../tools/")
from feature_format import featureFormat


def nanDetectorForKeys(dataset, features, percentage):
    nans = {}
    featureCount = len(features)
    for i in dataset:  ##loop through persons
        nan = 0.
        for t in dataset[i]:  ## loop through values
            if dataset[i][t] == 'NaN' or t == '':  ## if value is Nan
                nan += 1  ## add to nan

        if nan / featureCount > float(percentage) / 100:  ## if nan counts are bigger then the given percentage
            nans[i] = nan  ## add it to nans list

    return nans  ##return list


## detect nan values of each feature, with a given percentage
## return a list of detected features
def nanDetectorForValues(dataset, features, percentage):
    nans = {}
    dataPoints = float(len(dataset))
    allTogether = {}
    for feature in features:  ## loop through all features
        allTogether[feature] = list()  ## set a list for each feature
        for i in dataset.values():  ## append all values for each feataure to the dictionary
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


def plotScatter(data_dict, feature1, feature2):  ## visualising two features
    features = [feature1, feature2]  ## make a list of two
    data = featureFormat(data_dict, features)  ## format given features
    for point in data:
        f1 = point[0]
        f2 = point[1]
        py.scatter(f1, f2)  ## plot scatter with each record
    py.xlabel(feature1)
    py.ylabel(feature2)
    py.show()  ##show the plot


def plotBar(results):  ## visualising two features
    lists = {}
    for i in results:
        lists[i[2]] = i[1]

    fig = py.figure()
    lists_sorted = sorted(lists.items(), key=operator.itemgetter(1),
                          reverse=True)  # sorted by key, return a list of tuples

    x, y = zip(*lists_sorted)

    ind = np.arange(len(x))  # the x locations for the groups
    width = .5  # the width of the bars: can also be len(x) sequence

    py.bar(ind, y, width, color='#d62728')
    py.ylabel('Scores')
    py.title('Features')
    py.xticks(ind, x, rotation='vertical')
    py.subplots_adjust(bottom=0.5)
    py.savefig("features.png")
    # py.show()


def explore(data_dict):  ## exploring features with each other in scatter plot
    ##financial
    plotScatter(data_dict, "salary", "bonus")
    plotScatter(data_dict, "deferral_payments", "total_payments")
    plotScatter(data_dict, "loan_advances", "restricted_stock_deferred")
    plotScatter(data_dict, "deferred_income", "expenses")
    plotScatter(data_dict, "exercised_stock_options", "total_stock_value")
    plotScatter(data_dict, "long_term_incentive", "restricted_stock")
    plotScatter(data_dict, "other", "director_fees")
    ##emails
    plotScatter(data_dict, "from_this_person_to_poi", "from_poi_to_this_person")
    plotScatter(data_dict, "shared_receipt_with_poi", "to_messages")
    plotScatter(data_dict, "from_messages", "to_messages")


## find the index of given feature
def findIndex(feature, features_list):
    index = 0
    for i in features_list:  ## loop through the feature list, until the given feature is matched
        if i == feature:
            break
        index += 1
    return index


## calculate fractions of given -related- values for poi frequency
def computeFraction(param1, param2):
    fraction = 0.
    if float(param1) > 0. and float(param2) > 0.:  ## if both values are greater then 0
        if float(param2) == 0. or float(param1) == 0.:
            fraction = 0.
        else:
            fraction = float(param1) / float(param2)

    return fraction
