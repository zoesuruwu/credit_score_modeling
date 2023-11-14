from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
# ML
import statsmodels.api as sm
from imblearn.over_sampling import BorderlineSMOTE
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, plot_roc_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, train_test_split


def Find_Optimal_Cutoff(target, predicted):
    """Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame(
        {
            "tf": pd.Series(tpr - (1 - fpr), index=i),
            "threshold": pd.Series(threshold, index=i),
        }
    )
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

    return list(roc_t["threshold"])


def logistic_reg(X, Y):
    # Splitting 70% for train and 30% for test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, train_size=0.7, test_size=0.3, random_state=100
    )
    # smf
    # model = smf.logit("admit ~ gre + gpa + C(rank)", data=df).fit()
    # 1. Logistic regression model
    lr1 = sm.Logit(y_train, sm.add_constant(X_train)).fit()
    print(lr1.summary())
    # Odd ratio:
    coeff = lr1.params
    odd = np.exp(coeff)
    print("Odd ratio:")
    print(odd)
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.barh(coeff.index, coeff.values)
    plt.title("Bar chart of feature importance(coefficients)")
    plt.tight_layout()
    plt.show()
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.barh(odd.index, odd.values)
    plt.title("Bar chart of odd ratio")
    plt.tight_layout()
    plt.show()

    train_predictions = lr1.predict(sm.add_constant(X_train))
    test_predictions = lr1.predict(sm.add_constant(X_test))
    # Find optimal probability threshold from train dataset
    threshold = Find_Optimal_Cutoff(y_train, train_predictions)[0]

    print("Threshold: ", threshold)
    # Classify default or non-default from probability
    train_prediction_binary = train_predictions.map(lambda x: 1 if x > threshold else 0)
    test_predictions_binary = test_predictions.map(lambda x: 1 if x > threshold else 0)
    tests = ["Train", "Test"]
    prob_y = [y_train, y_test]
    binary_y = [train_prediction_binary, test_predictions_binary]
    for i in range(len(tests)):
        print("*" * 20 + tests[i] + "*" * 20)
        print("Accuracy:", metrics.accuracy_score(prob_y[i], binary_y[i]))
        print("Precision:", metrics.precision_score(prob_y[i], binary_y[i]))
        print("Recall:", metrics.recall_score(prob_y[i], binary_y[i]))
        lr_f1 = f1_score(prob_y[i], binary_y[i])
        print("Logistic regression: f1=%.3f" % (lr_f1))
        cm = metrics.confusion_matrix(prob_y[i], binary_y[i])
        print("Confusion matrix: ")
        cm = pd.DataFrame(
            index=["Observed non-default", "Observed default"],
            columns=["Predicted non-default", "Predicted default"],
            data=cm,
        )
        print(cm)

    ## ROC
    # no-kill
    ns_probs = [0 for _ in range(len(y_test))]
    # keep probabilities for the positive outcome only
    lr_probs = test_predictions
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    lr_auc = np.round(lr_auc, 4)
    # summarize scores
    print("Bad model: ROC AUC=%.3f" % (ns_auc))
    print("Logistic: ROC AUC=%.3f" % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle="--", label="Bad model")
    plt.plot(lr_fpr, lr_tpr, marker=".", label="Logistic")
    # axis labels
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # bad model = no classification power
    plt.title("Bad model: ROC AUC = {}, Logistics: ROC AUC = {}".format(ns_auc, lr_auc))
    # show the legend
    plt.legend()
    # show the plot
    plt.show()


def smote_borderline(X, y):
    # summarize class distribution
    counter = Counter(y)
    print("Original data set default-non-default proportion: ")
    print(counter)
    # transform the dataset
    oversample = BorderlineSMOTE()
    X, y = oversample.fit_resample(X, y)
    # summarize the new class distribution
    counter = Counter(y)
    print("After SMOTE data set default-non-default proportion: ")
    print(counter)
    return X, y


def hyper_parm_RF(X_validate, y_validate):
    # Number of trees in Random Forest
    rf_n_estimators = [int(x) for x in np.linspace(200, 1000, 5)]

    # Maximum number of levels in tree
    rf_max_depth = [int(x) for x in np.linspace(5, 55, 11)]
    # Add the default as a possible value
    rf_max_depth.append(None)

    # Number of features to consider at every split
    rf_max_features = ["sqrt", "log2"]

    # Minimum number of samples required to split a node
    rf_min_samples_split = [5, 7, 10, 14]

    # number of samples that should be present in the leaf node after splitting a node
    rf_min_samples_leaf = [4, 6, 8, 12]

    # what fraction of the original dataset is given to any individual tree
    rf_max_samples = [0.1, 0.2, 0.3]

    # maximum terminal nodes after splitting, if reached, the tree stops growing
    rf_max_leaf_nodes = [8, 12, 16, 20]
    rf_grid = {
        "n_estimators": rf_n_estimators,
        "max_depth": rf_max_depth,
        "max_features": rf_max_features,
        "min_samples_leaf": rf_min_samples_leaf,
        "min_samples_split": rf_min_samples_split,
        "max_samples": rf_max_samples,
        "max_leaf_nodes": rf_max_leaf_nodes,
    }
    rf_base = RandomForestClassifier()

    # Create the random search Random Forest
    rf_random = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=rf_grid,
        n_iter=60,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    # Fit the random search model
    rf_random.fit(X_validate, y_validate)

    # View the best parameters from the random search
    return rf_random.best_params_


def print_score(true, pred, train=True):
    if train:
        clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")

    elif train == False:
        clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
        print("Test Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")


def ABS_SHAP(df_shap, df):
    # import matplotlib as plt
    # Make a copy of the input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop("index", axis=1)

    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i], df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list), pd.Series(corr_list)], axis=1).fillna(
        0
    )
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns = ["Variable", "Corr"]
    corr_df["Sign"] = np.where(corr_df["Corr"] > 0, "red", "blue")

    # Plot it
    shap_abs = np.abs(shap_v)
    k = pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ["Variable", "SHAP_abs"]
    k2 = k.merge(corr_df, left_on="Variable", right_on="Variable", how="inner")
    k2 = k2.sort_values(by="SHAP_abs", ascending=True)
    colorlist = k2["Sign"]
    ax = k2.plot.barh(
        x="Variable", y="SHAP_abs", color=colorlist, figsize=(5, 6), legend=False
    )
    ax.set_xlabel("SHAP Value (Red = Positive Impact)")
    plt.tight_layout()
    plt.show()


def shap_plot(model, S, j):
    explainerModel = shap.TreeExplainer(model)
    shap_values_Model = explainerModel.shap_values(S)
    p = shap.force_plot(
        explainerModel.expected_value, shap_values_Model[j], S.iloc[[j]]
    )
    return p


def random_forest(X, y):
    # Source 1: https://towardsdatascience.com/cross-validation-and-hyperparameter-tuning-how-to-optimise-your-machine-learning-model-13f005af9d7d
    # Source 2: https://pierpaolo28.github.io/blog/blog25/
    # train-validation-test set
    # grid search
    # Perform first split
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Perform the second split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_temp, y_train_temp, test_size=0.25, random_state=42
    )
    # Hyperparameter tuning in the validation set
    # hyper_optimal = hyper_parm_RF(X_valid, y_valid)
    hyper_optimal = {
        "n_estimators": 200,
        "min_samples_split": 14,
        "min_samples_leaf": 12,
        "max_samples": 0.3,
        "max_leaf_nodes": 20,
        "max_features": "sqrt",
        "max_depth": 30,
    }
    print("Optimized Hyperparameter: ")
    print(hyper_optimal)
    # Train model Random Forest
    # RandomForestRegressor instead of RandomForestClassifier
    rf_reg = RandomForestRegressor(
        n_estimators=hyper_optimal["n_estimators"],
        min_samples_split=hyper_optimal["min_samples_split"],
        max_features=hyper_optimal["max_features"],
        max_depth=hyper_optimal["max_depth"],
        min_samples_leaf=hyper_optimal["min_samples_leaf"],
        max_leaf_nodes=hyper_optimal["max_leaf_nodes"],
        max_samples=hyper_optimal["max_samples"],
        criterion="mae",
        bootstrap=True,
        random_state=42,
    )

    rf_clf = RandomForestClassifier(
        n_estimators=hyper_optimal["n_estimators"],
        min_samples_split=hyper_optimal["min_samples_split"],
        max_features=hyper_optimal["max_features"],
        max_depth=hyper_optimal["max_depth"],
        min_samples_leaf=hyper_optimal["min_samples_leaf"],
        max_leaf_nodes=hyper_optimal["max_leaf_nodes"],
        max_samples=hyper_optimal["max_samples"],
        criterion="gini",
        bootstrap=True,
        random_state=42,
    )
    print("Based on RandomForestClassifier: ")
    rf_clf.fit(X_train, y_train)
    y_train_pred = rf_clf.predict(X_train)
    y_test_pred = rf_clf.predict(X_test)

    # Confusion matrix + F1 score
    print_score(y_train, y_train_pred, train=True)
    print_score(y_test, y_test_pred, train=False)
    # ROC curve
    plot_roc_curve(rf_clf, X_test, y_test)
    y_proba = rf_clf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_clf.predict_proba(X_test)[:, 1], average="macro")
    rf_auc = np.round(rf_auc, 4)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label="Random forest")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC AUC = {}".format(rf_auc))
    plt.legend(loc="best")
    plt.show()

    # SHAP
    df = pd.DataFrame(X_train)
    df["Target"] = y_train
    df_test = pd.DataFrame(X_test)
    df_test["Target"] = y_test
    df_sub = df.sample(n=10000, random_state=2)
    X_train_sub = df_sub.loc[:, df_sub.columns != "Target"]
    df_test_sub = df_test.sample(n=10000, random_state=2)
    df_test_sub = df_test_sub.reset_index(drop=True)
    X_test_sub = df_test_sub.loc[:, df_test_sub.columns != "Target"]

    print("Based on RandomForestRegressor: ")
    rf_reg.fit(X_train_sub, df_sub["Target"])
    rf_explainer = shap.TreeExplainer(rf_reg)
    shap_values = rf_explainer.shap_values(X_train_sub)

    test_explainer = shap.TreeExplainer(rf_reg)
    test_shap_values = test_explainer.shap_values(X_test_sub)
    print("In total: the test set has: {} data points".format(len(X_test)))
    print("Shap values calculated using 10000 sample size")

    # Force plot
    shap.force_plot(
        np.round(test_explainer.expected_value),
        np.round(test_shap_values[0], 2),
        np.round(X_test_sub.iloc[[0]], 2),
        matplotlib=True,
        show=False,
    )
    plt.title("First observation - default")
    plt.tight_layout()
    print("First observation is defaulted? {}".format(df_test_sub["Target"][0]))
    shap.force_plot(
        np.round(test_explainer.expected_value),
        np.round(test_shap_values[9], 2),
        np.round(X_test_sub.iloc[[9]], 2),
        matplotlib=True,
        show=False,
    )
    plt.title("10th observation - non-default")
    plt.tight_layout()
    print("10th observation is defaulted? {}".format(df_test_sub["Target"][9]))
    print("Base value(y): {}".format(df_test_sub["Target"].mean()))
    predictor_df = pd.DataFrame(
        index=X_test_sub.columns, columns=["Predictor_mean", "first_obs", "10th_obs"]
    )
    predictor_df["Predictor_mean"] = np.mean(X_test_sub)
    predictor_df["first_obs"] = X_test_sub.iloc[0, :].values
    predictor_df["10th_obs"] = X_test_sub.iloc[9, :].values
    print(predictor_df)
    f = plt.figure()
    shap.summary_plot(test_shap_values, X_test_sub)
    f.tight_layout()
    f = plt.figure()
    shap.summary_plot(shap_values, X_train_sub, plot_type="bar")
    f.tight_layout()
    # positive/negative global plot1
    ABS_SHAP(shap_values, X_train_sub)
