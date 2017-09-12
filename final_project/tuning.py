from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

def tuning():
    ## best scored classifier
    clf1 = GaussianNB(priors=None)

    ## second one
    clf2 = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
                              max_features=None, max_leaf_nodes=None,
                              min_impurity_decrease=0.0, min_impurity_split=None,
                              min_samples_leaf=1, min_samples_split=2,
                              min_weight_fraction_leaf=0.0, presort=False, random_state=42,
                              splitter='random')
    ## third one
    clf3 = LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
                          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                          penalty='l2', random_state=42, solver='liblinear', tol=0.0001,
                          verbose=0, warm_start=False)

    from tester import test_classifier, load_classifier_and_data ##import tester scripts to run cross validation

    _, my_dataset, features_list = load_classifier_and_data() ##load the dataset and features

    #trying naive bayes with different test sets
    test_classifier(clf1, my_dataset, features_list, folds=100)
    # GaussianNB(priors=None)
    # 	Accuracy: 0.81733	Precision: 0.32381	Recall: 0.34000	F1: 0.33171	F2: 0.33663
    # 	Total predictions: 1500	True positives:   68	False positives:  142	False negatives:  132	True negatives: 1158

    test_classifier(clf1, my_dataset, features_list, folds=500)
    # GaussianNB(priors=None)
    # 	Accuracy: 0.82867	Precision: 0.34426	Recall: 0.31500	F1: 0.32898	F2: 0.32045
    # 	Total predictions: 7500	True positives:  315	False positives:  600	False negatives:  685	True negatives: 5900

    test_classifier(clf1, my_dataset, features_list, folds=1000)
    # GaussianNB(priors=None)
    # 	Accuracy: 0.82860	Precision: 0.34322	Recall: 0.31250	F1: 0.32714	F2: 0.31820
    # 	Total predictions: 15000	True positives:  625	False positives: 1196	False negatives: 1375	True negatives: 11804

    test_classifier(clf1, my_dataset, features_list, folds=2000)
    # GaussianNB(priors=None)
    # 	Accuracy: 0.82570	Precision: 0.33503	Recall: 0.31200	F1: 0.32311	F2: 0.31635
    # 	Total predictions: 30000	True positives: 1248	False positives: 2477	False negatives: 2752	True negatives: 23523



    ## trying decision trees

    test_classifier(clf2, my_dataset, features_list, folds=100)
    # DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
    #             max_features=None, max_leaf_nodes=None,
    #             min_impurity_decrease=0.0, min_impurity_split=None,
    #             min_samples_leaf=1, min_samples_split=2,
    #             min_weight_fraction_leaf=0.0, presort=False, random_state=42,
    #             splitter='random')
    # 	Accuracy: 0.85867	Precision: 0.39655	Recall: 0.11500	F1: 0.17829	F2: 0.13403
    # 	Total predictions: 1500	True positives:   23	False positives:   35	False negatives:  177	True negatives: 1265

    test_classifier(clf2, my_dataset, features_list, folds=500)
    # DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
    #             max_features=None, max_leaf_nodes=None,
    #             min_impurity_decrease=0.0, min_impurity_split=None,
    #             min_samples_leaf=1, min_samples_split=2,
    #             min_weight_fraction_leaf=0.0, presort=False, random_state=42,
    #             splitter='random')
    # 	Accuracy: 0.85720	Precision: 0.38206	Recall: 0.11500	F1: 0.17679	F2: 0.13369
    # 	Total predictions: 7500	True positives:  115	False positives:  186	False negatives:  885	True negatives: 6314

    test_classifier(clf2, my_dataset, features_list, folds=1000)
    # DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
    #             max_features=None, max_leaf_nodes=None,
    #             min_impurity_decrease=0.0, min_impurity_split=None,
    #             min_samples_leaf=1, min_samples_split=2,
    #             min_weight_fraction_leaf=0.0, presort=False, random_state=42,
    #             splitter='random')
    # 	Accuracy: 0.85733	Precision: 0.38636	Recall: 0.11900	F1: 0.18196	F2: 0.13812
    # 	Total predictions: 15000	True positives:  238	False positives:  378	False negatives: 1762	True negatives: 12622

    test_classifier(clf2, my_dataset, features_list, folds=2000)
    # DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
    #             max_features=None, max_leaf_nodes=None,
    #             min_impurity_decrease=0.0, min_impurity_split=None,
    #             min_samples_leaf=1, min_samples_split=2,
    #             min_weight_fraction_leaf=0.0, presort=False, random_state=42,
    #             splitter='random')
    # 	Accuracy: 0.85870	Precision: 0.40229	Recall: 0.12300	F1: 0.18840	F2: 0.14283
    # 	Total predictions: 30000	True positives:  492	False positives:  731	False negatives: 3508	True negatives: 25269



    ## trying logistic regression
    test_classifier(clf3, my_dataset, features_list, folds=100)
    # LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
    #           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
    #           penalty='l2', random_state=42, solver='liblinear', tol=0.0001,
    #           verbose=0, warm_start=False)
    # 	Accuracy: 0.76600	Precision: 0.14218	Recall: 0.15000	F1: 0.14599	F2: 0.14837
    # 	Total predictions: 1500	True positives:   30	False positives:  181	False negatives:  170	True negatives: 1119

    test_classifier(clf3, my_dataset, features_list, folds=500)
    # LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
    #           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
    #           penalty='l2', random_state=42, solver='liblinear', tol=0.0001,
    #           verbose=0, warm_start=False)
    # 	Accuracy: 0.76333	Precision: 0.13060	Recall: 0.13700	F1: 0.13372	F2: 0.13567
    # 	Total predictions: 7500	True positives:  137	False positives:  912	False negatives:  863	True negatives: 5588

    test_classifier(clf3, my_dataset, features_list, folds=1000)
    # LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
    #           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
    #           penalty='l2', random_state=42, solver='liblinear', tol=0.0001,
    #           verbose=0, warm_start=False)
    # 	Accuracy: 0.76387	Precision: 0.13039	Recall: 0.13600	F1: 0.13314	F2: 0.13484
    # 	Total predictions: 15000	True positives:  272	False positives: 1814	False negatives: 1728	True negatives: 11186

    test_classifier(clf3, my_dataset, features_list, folds=2000)
    # LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
    #           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
    #           penalty='l2', random_state=42, solver='liblinear', tol=0.0001,
    #           verbose=0, warm_start=False)
    # 	Accuracy: 0.76437	Precision: 0.13069	Recall: 0.13575	F1: 0.13317	F2: 0.13471
    # 	Total predictions: 30000	True positives:  543	False positives: 3612	False negatives: 3457	True negatives: 22388


    ##
    # after running all classifiers with different sizes of folds, the best algorithm is still naive bayes according to
    # the both precision and recall scores
    ##
    return clf1

if __name__ == '__main__':
    tuning()