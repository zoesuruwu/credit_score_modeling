import logging
import warnings

import datatable as dt
import numpy as np
import pandas as pd

from data_cleaning import data_pre_propressing
from model_building import logistic_reg, random_forest, smote_borderline


def main():
    logger = logging.getLogger()
    warnings.filterwarnings("ignore")
    pd.set_option("display.max_columns", None)
    np.set_printoptions(suppress=False)
    logger.info("Starting with data-pre prosessing: ")
    data_pre_propressing(logger)
    df = dt.fread("cleaned_accepted_2007_to_2018Q4.csv").to_pandas()
    X = df.drop(["loan_status"], axis=1)
    Y = df["loan_status"].map({"Non-defaulted": 0, "Defaulted": 1})
    dummy = ["grade_B", "grade_C", "grade_D+", "term_60 months"]
    for var in dummy:
        X[var] = X[var].map({True: 1, False: 0})

    ### ML models
    # Over-sampling smote: https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
    X, Y = smote_borderline(X, Y)
    # Logistic regression with SMOTE
    print("-" * 50 + "Logistic regression with SMOTE balanced data" + "-" * 50)
    logistic_reg(X, Y)
    ### Random forest
    random_forest(X, Y)
    # SHAP source: 1. https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d
    #              2. https://towardsdatascience.com/explain-any-models-with-the-shap-values-use-the-kernelexplainer-79de9464897a
    #              3. https://towardsdatascience.com/using-shap-values-to-explain-how-your-machine-learning-model-works-732b3f40e137

    ### add codes in Model_building here


if __name__ == "__main__":
    main()
