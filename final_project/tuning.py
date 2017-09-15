import warnings

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings("ignore")

## prints the results if its executed from this script
def tuning(printTestResults):

    ## best scored classifier
    clf1 = GaussianNB(priors=None)

    ## second one
    clf2 = SVC(C=1000, cache_size=7000, class_weight=None, coef0=0.0,
               decision_function_shape='ovo', degree=3, gamma='auto', kernel='poly',
               max_iter=-1, probability=False, random_state=42, shrinking=True,
               tol=0.0001, verbose=False)

    ## third one
    clf3 = AdaBoostClassifier(algorithm='SAMME', base_estimator=None, learning_rate=0.5,
                              n_estimators=50, random_state=42)

    ## others
    clf4 = LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
                              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                              penalty='l2', random_state=42, solver='liblinear', tol=0.0001,
                              verbose=False, warm_start=False)

    clf5 = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                                  max_features=None, max_leaf_nodes=6, min_impurity_decrease=0.0,
                                  min_impurity_split=None, min_samples_leaf=1,
                                  min_samples_split=2, min_weight_fraction_leaf=0.0,
                                  presort=False, random_state=42, splitter='random')

    clf6 = RandomForestClassifier(bootstrap=True, class_weight='balanced',
                                  criterion='gini', max_depth=None, max_features='auto',
                                  max_leaf_nodes=None, min_impurity_decrease=0.0,
                                  min_impurity_split=None, min_samples_leaf=1,
                                  min_samples_split=2, min_weight_fraction_leaf=0.0,
                                  n_estimators=10, n_jobs=1, oob_score=False, random_state=42,
                                  verbose=False, warm_start=False)

    clf7 = KNeighborsClassifier(algorithm='auto', leaf_size=10, metric='minkowski',
                                metric_params=None, n_jobs=1, n_neighbors=4, p=2,
                                weights='uniform')

    clf8 = KMeans(algorithm='full', copy_x=True, init='k-means++', max_iter=50,
                  n_clusters=16, n_init=10, n_jobs=1, precompute_distances=True,
                  random_state=42, tol=0.0001, verbose=False)

    from tester import test_classifier, load_classifier_and_data  ##import tester scripts to run cross validation

    _, my_dataset, features_list = load_classifier_and_data()  ##load the dataset and features

    if printTestResults:

        # trying naive bayes with different test sets
        test_classifier(clf1, my_dataset, features_list, folds=100)
        # GaussianNB(priors=None)
        #	Accuracy: 0.85077	Precision: 0.52679	Recall: 0.29500	F1: 0.37821	F2: 0.32346
        #	Total predictions: 1300	True positives:   59	False positives:   53	False negatives:  141	True negatives: 1047

        test_classifier(clf1, my_dataset, features_list, folds=500)
        # GaussianNB(priors=None)
        #	Accuracy: 0.84846	Precision: 0.51152	Recall: 0.33300	F1: 0.40339	F2: 0.35799
        #	Total predictions: 6500	True positives:  333	False positives:  318	False negatives:  667	True negatives: 5182

        test_classifier(clf1, my_dataset, features_list, folds=1000)
        # GaussianNB(priors=None)
        #	Accuracy: 0.84677	Precision: 0.50312	Recall: 0.32300	F1: 0.39342	F2: 0.34791
        #	Total predictions: 13000	True positives:  646	False positives:  638	False negatives: 1354	True negatives: 10362

        test_classifier(clf1, my_dataset, features_list, folds=2000)
        # GaussianNB(priors=None)
        #	Accuracy: 0.84831	Precision: 0.51088	Recall: 0.32875	F1: 0.40006	F2: 0.35399
        #	Total predictions: 26000	True positives: 1315	False positives: 1259	False negatives: 2685	True negatives: 20741



        ## trying svm

        test_classifier(clf2, my_dataset, features_list, folds=100)
        # SVC(C=1000, cache_size=7000, class_weight=None, coef0=0.0,
        #  decision_function_shape='ovo', degree=3, gamma='auto', kernel='poly',
        #  max_iter=-1, probability=False, random_state=42, shrinking=True,
        #  tol=0.0001, verbose=False)
        #	Accuracy: 0.85538	Precision: 0.67647	Recall: 0.11500	F1: 0.19658	F2: 0.13789
        #	Total predictions: 1300	True positives:   23	False positives:   11	False negatives:  177	True negatives: 1089

        test_classifier(clf2, my_dataset, features_list, folds=500)
        # SVC(C=1000, cache_size=7000, class_weight=None, coef0=0.0,
        #  decision_function_shape='ovo', degree=3, gamma='auto', kernel='poly',
        #  max_iter=-1, probability=False, random_state=42, shrinking=True,
        #  tol=0.0001, verbose=False)
        #	Accuracy: 0.85508	Precision: 0.63810	Recall: 0.13400	F1: 0.22149	F2: 0.15914
        #	Total predictions: 6500	True positives:  134	False positives:   76	False negatives:  866	True negatives: 5424

        test_classifier(clf2, my_dataset, features_list, folds=1000)
        # SVC(C=1000, cache_size=7000, class_weight=None, coef0=0.0,
        #  decision_function_shape='ovo', degree=3, gamma='auto', kernel='poly',
        #  max_iter=-1, probability=False, random_state=42, shrinking=True,
        #  tol=0.0001, verbose=False)
        #	Accuracy: 0.85415	Precision: 0.62745	Recall: 0.12800	F1: 0.21262	F2: 0.15224
        #	Total predictions: 13000	True positives:  256	False positives:  152	False negatives: 1744	True negatives: 10848

        test_classifier(clf2, my_dataset, features_list, folds=2000)
        # SVC(C=1000, cache_size=7000, class_weight=None, coef0=0.0,
        #  decision_function_shape='ovo', degree=3, gamma='auto', kernel='poly',
        #  max_iter=-1, probability=False, random_state=42, shrinking=True,
        #  tol=0.0001, verbose=False)
        #	Accuracy: 0.85427	Precision: 0.62726	Recall: 0.13000	F1: 0.21537	F2: 0.15450
        #	Total predictions: 26000	True positives:  520	False positives:  309	False negatives: 3480	True negatives: 21691



        ## trying logistic regression
        test_classifier(clf3, my_dataset, features_list, folds=100)
        # AdaBoostClassifier(algorithm='SAMME', base_estimator=None, learning_rate=0.5,
        #          n_estimators=50, random_state=42)
        #	Accuracy: 0.85000	Precision: 0.54386	Recall: 0.15500	F1: 0.24125	F2: 0.18086
        #	Total predictions: 1300	True positives:   31	False positives:   26	False negatives:  169	True negatives: 1074

        test_classifier(clf3, my_dataset, features_list, folds=500)
        # AdaBoostClassifier(algorithm='SAMME', base_estimator=None, learning_rate=0.5,
        #          n_estimators=50, random_state=42)
        #	Accuracy: 0.84415	Precision: 0.48148	Recall: 0.16900	F1: 0.25019	F2: 0.19421
        #	Total predictions: 6500	True positives:  169	False positives:  182	False negatives:  831	True negatives: 5318

        test_classifier(clf3, my_dataset, features_list, folds=1000)
        # AdaBoostClassifier(algorithm='SAMME', base_estimator=None, learning_rate=0.5,
        #          n_estimators=50, random_state=42)
        #	Accuracy: 0.84315	Precision: 0.47137	Recall: 0.16050	F1: 0.23946	F2: 0.18489
        #	Total predictions: 13000	True positives:  321	False positives:  360	False negatives: 1679	True negatives: 10640

        test_classifier(clf3, my_dataset, features_list, folds=2000)

        # AdaBoostClassifier(algorithm='SAMME', base_estimator=None, learning_rate=0.5,
        #          n_estimators=50, random_state=42)
        #	Accuracy: 0.84308	Precision: 0.47114	Recall: 0.16325	F1: 0.24248	F2: 0.18779
        #	Total predictions: 26000	True positives:  653	False positives:  733	False negatives: 3347	True negatives: 21267


    def testClassifier(clf, fold):
        test_classifier(clf, my_dataset, features_list, folds=fold)

    if printTestResults:
        for i in [100, 500, 1000, 2000]:
            testClassifier(clf1, i)

        for i in [100, 500, 1000, 2000]:
            testClassifier(clf2, i)

        for i in [100, 500, 1000, 2000]:
            testClassifier(clf3, i)

        for i in [100, 500, 1000, 2000]:
            testClassifier(clf4, i)

        for i in [100, 500, 1000, 2000]:
            testClassifier(clf5, i)

        for i in [100, 500, 1000, 2000]:
            testClassifier(clf6, i)

        for i in [100, 500, 1000, 2000]:
            testClassifier(clf7, i)

        for i in [100, 500, 1000, 2000]:
            testClassifier(clf8, i)

    ##
    # after running all classifiers with different sizes of folds, the best algorithm is still naive bayes according to
    # the both precision and recall scores
    ##
    return clf1


if __name__ == '__main__':
    tuning(True)
