# Identify Fraud from Enron Email

## Summary

The goal of this project is to build a model to identify Person of Interests (POI's) from given 
Enron email data, by using some machine learning algorithms. 
The data includes 146 different individual's email and financial records. 
The data also includes the feature labeled "poi" for these individuals, which means we already 
know the POI's. There are total of 21 features, 6 email features , 14 financial features and 
the poi label.
By using these informations, we want to build an algorithm to make predictions to identify POI's,
maybe for another similiar cases too.

But there are some problems with the dataset of course. Not every individuals have these features 
filled, or not every feature is actually useful for our model and also there is an outlier. 

- Found the features with half of the values are not valid in the data set : 
`ALL FEATURES EXCEPT 'poi' :)`
- Found the individuals whose features are %90 NaN : 
`'LOCKHART EUGENE E'`
- Found the outlier : 
`'TOTAL'`

With these results, i removed the record of TOTAL and EUGENE. 
And for the NaN features, i decided to use feature selection while training my models.

## Features

After exploring some visualisations, i decided to add three new features to the dataset.

Email features seemed very useful for calculating fractions of emails from & to POI's, so i added two
features labeled as :
- `'fraction_to_poi'`
- `'fraction_from_poi'`

And other thing that drawed my attention is the salary and total payments features.
Total payments features looks like the total of all other financial features. And the salary is the 
only stable feature in them. Other features seems to me as extra incomes. So i calculated the incomes 
except the salaries fraction of each individuals, labeled as :
- `'fraction_to_salary'`

Then i scaled the features using sklearn's `MinMaxScaler`. Because every SO questions i read about 
machine learning algorithms and their performance issues, scaling is the most suggested process.

After scaling the features, i used sklearn's `SelectKBest` feature selection algorithm to pick 
10 best features that i can use in my processes, and these 10 features came up :

| Selected Features | Scores |
| ---------------------- | ----- |
|  exercised_stock_options  |  25.0975415287  |
|  total_stock_value  |  24.4676540475  |
|  bonus  |  21.0600017075  |
|  salary  |  18.575703268  |
|  deferred_income  |  11.5955476597  |
|  long_term_incentive  |  10.0724545294  |
|  restricted_stock  |  9.34670079105  |
|  total_payments  |  8.86672153711  |
|  shared_receipt_with_poi  |  8.74648553213  |
|  loan_advances  |  7.24273039654  |

The features that i added to my features list didn't come up in this table, so it seems other 
features are better indetifiers then mines.


## Algorithm

I used sklearn's `GridSearchCV` to try different algorithms with a number of different
parameters to fit our dataset. Stored all algorithms with their best parameter settings in a
dictionary with the score metrics. 
It's hard to say something about the performances of each variation, so i also stored 
each parameter's training and prediction times in that dictionary.

I also calculated a different score metric, by using all calculated results and the times.
Using this new score, i'll try to use best three algorithms.

My dictionary includes all of the above :
 - SVM
 - RandomForest
 - KMeans
 - NaiveBayes
 - DecisionTree
 - LogisticRegression
 - AdaBoost
 - KNeighbors

| Algorithm | My Score | Accuracy | Precision | Recall | f1 | f2 | Training Time (s) | Prediction Time(s) |
| --------- | ----- | ----- | ----- | ----- | ----- | ----- |----| ---- |
|  SVM  |  0.0  |  0.86  |  0.0  |  0.0  |  0.0  |  0.0  |  0.91  |  0.0  |
|  RandomForest  |  0.0  |  0.89  |  0.5  |  0.2  |  0.29  |  0.23  |  20.86  |  0.0  |
|  KMeans  |  0.0  |  0.0  |  0.0  |  0.0  |  0.0  |  0.0  |  44.49  |  0.0  |
|  NaiveBayes  |  5.3  |  0.64  |  0.24  |  1.0  |  0.38  |  0.61  |  0.01  |  0.0  |
|  DecisionTree  |  0.01  |  0.91  |  0.67  |  0.4  |  0.5  |  0.43  |  8.14  |  0.0  |
|  LogisticRegression  |  0.02  |  0.89  |  0.5  |  0.4  |  0.44  |  0.42  |  3.81  |  0.0  |
|  AdaBoost  |  0.0  |  0.86  |  0.33  |  0.2  |  0.25  |  0.22  |  13.38  |  0.0  |
|  KNeighbors  |  0.0  |  0.89  |  0.0  |  0.0  |  0.0  |  0.0  |  1.84  |  0.0  |


I intended to use best three of these algorithms. But the list already includes three algorithms,
whose scores are greater then zero. So, i'll keep going with 
`NaiveBayes`, `LogisticRegression` and `DecisionTree`.

## Tuning Parameters

Parameter tuning is a neccesary step in machine learning. By tuning the parameters, we can 
improve our predictions or we can improve our training and prediction times.

But also, by over-tuning the parameters, we can also make our algorithm to act worse. 
The algorithm can be trained so well and get a overfit shape. So, while everything is great 
in test sets, it can behave really bad in big data sets and make misleading choices. 
So i think this step is kind of a trial-denial process. Yet, the feature i used in this project is
just making these trials at once and gives the best result from all possible choices. And that is 
GridSearchCV.

## Validation

Validation is a process, which we test our algorithms our own. 
By using the provided dataset, if we train our models with a small size of test points, we'll probably
get higher rates of score metrics, such as precision, recall and f1. But this does not mean, 
our model predicts very good. It predicts what it learns. And this is the classic mistake we can make. 

We can say that if we train our model with a huge data, then we'll probably get better results. 
But training with huge data sets, is not always possible and also the time it takes to train 
our model would increase too.

So, with this validation step we test our algorithm with different variations of the test data by 
generating new data points. And provided StratifiedSuffleSplit feature does a really great job generating
new data points. It doesn't just splits the data, it also randomizes the data set and generates different
data points, and generates as much as we want.

Now we can test our algorithm with the new test set, and see the results. Since the test size got higher,
it's normal to see the score metrics to drop. I did pick three different algorithms with high scores already,
so i ran this validation for each of them. And among them, i choosed the best one, which is in this case
`'Gaussian Naive-Bayes'`. Other two algorithms prediction metrics dropped more with bigger test sizes.

## Metrics & Performance

I used all available score metrics in a way, while choosing the one algorithm.
I made up a formula, that makes sense to me as above:

> (`f1_score` * `f2_score` * `precision` * `recall`) / (`Total Training and Prediction Time`)

I mentioned the metrics above for the provided dataset for each algorithms. But after using validations 
with different test sets, the one i choosed `Naive Bayes` gives the following scores with test 
sizes [100,500,1000,2000], which meets the project rubric scores for precision and recall (0.3) :

- `Accuracy`: 0.81733	`Precision`: 0.32381	`Recall`: 0.34000	`F1`: 0.33171	`F2`: 0.33663
- `Accuracy`: 0.82867	`Precision`:: 0.34426	`Recall`: 0.31500	`F1`: 0.32898	`F2`: 0.32045
- `Accuracy`: 0.82860	`Precision`:: 0.34322	`Recall`: 0.31250	`F1`: 0.32714	`F2`: 0.31820
- `Accuracy`: 0.82570	`Precision`:: 0.33503	`Recall`: 0.31200	`F1`: 0.32311	`F2`: 0.31635

And Naive-Bayes also, the fastest algorithm among others.


## References:
- [Introduction to Machine Learning (Udacity)](https://www.udacity.com/course/viewer#!/c-ud120-nd)
- [MITx Analytics Edge](https://www.edx.org/course/analytics-edge-mitx-15-071x-0)
- [scikit-learn Documentation](http://scikit-learn.org/stable/documentation.html)

## Files
- `poi_id.py`: main submission file
- `tester.py`: producing test results for submission, provided by Udacity
- `tuning.py`: cross validations of the results, and getting the best algorithm
- `final project/`: dataset files and pickle objects
- `tools/`: feature formatting funtions included