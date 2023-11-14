## move all codes from main to here
import datatable as dt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def drop_correlated(df, num_Columns):
    num_Columns.remove("loan_status")
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[num_Columns].corr(), annot=True, cmap="viridis")
    plt.tight_layout()
    plt.show()

    """ data anlysis: these two variables have correlation of 0.71:
    open_acc: The number of open credit lines in the borrower's credit file.
    total_acc: The total number of credit lines currently in the borrower's credit file
    -> Keep total_acc
    """
    df = df.drop("open_acc", axis=1)
    """ 2. data anlysis: these two variables have correlation of 0.69:
    pub_rec: Number of derogatory public records
    pub_rec_bankruptcies: Number of public record bankruptcies
    -> Keep pub_rec_bankruptcies
    """
    df = df.drop("pub_rec", axis=1)
    return df


def drop_outliers_plot(df, logger):
    """
    each variable is pre-examined and the criteria for dropping outliers are evaluated case by case
    """
    logger.info("Drop outliers -- Starting with variable annual_inc")
    print("annual_inc: drop where values are more than 300000, the proportion is: ")
    print(np.sum(df["annual_inc"] > 300000) / len(df["annual_inc"]) * 100)
    print("And number of obs: {}".format(len(df["annual_inc"] > 300000)))
    df = df[df["annual_inc"] <= 300000]
    sns.displot(
        data=df,
        x="annual_inc",
        hue="loan_status",
        bins=80,
        height=5,
        aspect=3,
        kde=True,
    )
    plt.title("Distribution of annual income of customers")
    plt.tight_layout()
    plt.show()

    logger.info("Drop outliers -- Starting with variable int_rate")
    sns.displot(
        data=df, x="int_rate", hue="loan_status", bins=80, height=5, aspect=3, kde=True
    )
    plt.title("Distribution of interest rate of customers")
    plt.tight_layout()
    plt.show()

    logger.info("Drop outliers -- Starting with variable loan_amnt")
    print("loan_amnt: drop more than 36000 with pct: ")
    print(np.sum(df["loan_amnt"] > 36000) / len(df["loan_amnt"]) * 100)
    df = df[df["loan_amnt"] <= 36000]
    sns.displot(data=df, x="loan_amnt", hue="loan_status", bins=80, height=5, aspect=3)
    plt.title("Distribution of loan amount of customers")
    plt.tight_layout()
    plt.show()

    logger.info("Drop outliers -- Starting with variable num_actv_bc_tl")
    print("num_actv_bc_tl: drop more than 10 with pct: ")
    print(np.sum(df["num_actv_bc_tl"] > 10) / len(df["num_actv_bc_tl"]) * 100)
    df = df[df["num_actv_bc_tl"] <= 10]
    df["num_actv_bc_tl"] = df["num_actv_bc_tl"].astype("int")
    sns.countplot(data=df, x="num_actv_bc_tl", hue="loan_status")
    plt.title(
        "Distribution of number of currently active bankcard accounts of customers",
        size=10,
    )
    plt.tight_layout()
    plt.show()

    logger.info("Drop outliers -- Starting with variable mort_acc")
    print("mort_acc: drop more than 10 with pct: ")
    print(np.sum(df["mort_acc"] > 10) / len(df["mort_acc"]) * 100)
    df = df[df["mort_acc"] <= 10]
    df["mort_acc"] = df["mort_acc"].astype("int")
    sns.countplot(data=df, x="mort_acc", hue="loan_status")
    plt.title("Distribution of number of mortgage accounts of customers", size=10)
    plt.tight_layout()
    plt.show()

    logger.info("Drop outliers -- Starting with variable tot_cur_bal")
    print("tot_cur_bal: drop more than 1000000 with pct: ")
    print(np.sum(df["tot_cur_bal"] > 1000000) / len(df["tot_cur_bal"]) * 100)
    df = df[df["tot_cur_bal"] <= 1000000]
    sns.displot(
        data=df,
        x="tot_cur_bal",
        hue="loan_status",
        bins=80,
        height=5,
        aspect=3,
        kde=True,
    )
    plt.title(
        "Distribution of the total current balance of all accounts of customers",
        size=10,
    )
    plt.tight_layout()
    plt.ticklabel_format(style="plain", axis="x")
    plt.show()

    logger.info("Drop outliers -- Starting with variable pub_rec_bankruptcies")
    print("pub_rec_bankruptcies: drop more than 3")
    print(
        np.sum(df["pub_rec_bankruptcies"] > 3) / len(df["pub_rec_bankruptcies"]) * 100
    )
    print("And number of obs: {}".format(len(df["pub_rec_bankruptcies"] > 3)))
    df = df[df["pub_rec_bankruptcies"] <= 3]
    df["pub_rec_bankruptcies"] = df["pub_rec_bankruptcies"].astype("int")
    sns.countplot(data=df, x="pub_rec_bankruptcies", hue="loan_status")
    plt.title(
        "Distribution of number of public record bankruptcies of customers", size=10
    )
    plt.tight_layout()
    plt.show()

    # revol_util: how much you currently owe divided by your credit limit.
    logger.info("Drop outliers -- Starting with variable revol_util")
    print("revol_util: drop more than 120")
    print(np.sum(df["revol_util"] > 120) / len(df["revol_util"]) * 100)
    df = df[df["revol_util"] <= 120]
    sns.displot(
        data=df,
        x="revol_util",
        hue="loan_status",
        bins=80,
        height=5,
        aspect=3,
        kde=True,
    )
    plt.title(
        "Distribution of the revolving line utilization rate of customers", size=10
    )
    plt.tight_layout()
    plt.show()

    logger.info("Drop outliers -- Starting with variable total_acc")
    print("total_acc: drop more than 60")
    print(np.sum(df["total_acc"] > 60) / len(df["total_acc"]) * 100)
    df = df[df["total_acc"] <= 60]
    sns.displot(
        data=df, x="total_acc", hue="loan_status", bins=80, height=5, aspect=3, kde=True
    )
    plt.title("Distribution of total number of credit lines of customers")
    plt.tight_layout()
    plt.show()
    # earliest_cr_line
    df["earliest_cr_line"] = df["earliest_cr_line"].astype("int")
    sns.countplot(data=df, x="earliest_cr_line", hue="loan_status")
    plt.title(
        "Distribution of the earliest reported credit line year of customers", size=10
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return df


def calc_iv(df, feature, target, pr=0):
    lst = []
    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append(
            [
                feature,
                val,
                df[df[feature] == val].count()[feature],
                df[(df[feature] == val) & (df[target] == "Defaulted")].count()[feature],
            ]
        )

    data = pd.DataFrame(lst, columns=["Variable", "Value", "All", "Bad"])
    data = data[data["Bad"] > 0]
    data["Share"] = data["All"] / data["All"].sum()
    data["Bad Rate"] = data["Bad"] / data["All"]
    data["Distribution Good"] = (data["All"] - data["Bad"]) / (
        data["All"].sum() - data["Bad"].sum()
    )
    data["Distribution Bad"] = data["Bad"] / data["Bad"].sum()
    data["WoE"] = np.log(data["Distribution Good"] / data["Distribution Bad"])
    data["IV"] = (
        data["WoE"] * (data["Distribution Good"] - data["Distribution Bad"])
    ).sum()
    data = data.sort_values(by=["Variable", "Value"], ascending=True)
    return data["IV"].values[0]


def drop_barchart(df):
    """
    ['grade', 'home_ownership', 'application_type', 'purpose', 'term']
    """
    ### Calculate Information value
    variables = ["grade", "home_ownership", "application_type", "purpose", "term"]
    IV_df = pd.Series(index=variables)
    for i in variables:
        IV_df[i] = calc_iv(df, i, "loan_status", pr=0)
    print(IV_df)
    poor = 0.15
    print(
        "Information values smaller than {0} classified as poor classification power".format(
            poor
        )
    )
    print(IV_df[IV_df < poor])
    df = df.drop(IV_df[IV_df < poor].sort_values().head(3).index, axis=1)
    #'grade'
    subgrade_order = sorted(df["grade"].unique())
    sns.countplot(x="grade", data=df, order=subgrade_order)
    plt.title("Loan grade")
    plt.tight_layout()
    plt.show()
    grouped = (
        df[["loan_status", "grade"]].groupby(["loan_status", "grade"]).size().unstack()
    )
    ax = grouped.plot(kind="bar", stacked=True)
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    plt.ylabel("Count")
    plt.xlabel("Loan status")
    plt.title("Loan grade")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    df["grade"].replace({"D": "D+", "E": "D+", "F": "D+", "G": "D+"}, inplace=True)

    # term
    sns.countplot(x="term", data=df)
    plt.title("Term")
    plt.tight_layout()
    plt.show()
    grouped = (
        df[["loan_status", "term"]].groupby(["loan_status", "term"]).size().unstack()
    )
    ax = grouped.plot(kind="bar", stacked=True)
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    plt.ylabel("Count")
    plt.xlabel("Loan status")
    plt.title("Term")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    print("Drop columns: [home_ownership, application_type, purpose]")

    return df


def imputing_missing(df):
    # Check for missing values %
    missing_pct = ((df.isnull().sum() / len(df)) * 100).sort_values()
    missing_pct = missing_pct[missing_pct > 0]
    print("Missing values in pct: \n", missing_pct)
    # 'revol_util': fill the missing values based on mean of each loan status group
    df["revol_util"] = df["revol_util"].fillna(
        df.groupby("loan_status")["revol_util"].transform("mean")
    )
    # 'num_actv_bc_tl': not much difference between each loan status group, so take the mean value of num_actv_bc_tl
    df["num_actv_bc_tl"] = df["num_actv_bc_tl"].fillna(
        np.round(df["num_actv_bc_tl"].mean(), 0)
    )
    # 'tot_cur_bal': fill the missing values based on mean of each loan status group
    df["tot_cur_bal"] = df["tot_cur_bal"].fillna(
        df.groupby("loan_status")["tot_cur_bal"].transform("mean")
    )
    print(missing_pct)
    return df


def numerical_transform(df, numerical_var):
    """
    1. only take log for variables ['annual_inc', 'tot_cur_bal' ], since it is right-skewed
    2. Standardize all numeric variables
    """
    print("Transform numerical variable: ")
    print(numerical_var)
    df["annual_inc"] = np.log(df["annual_inc"])
    df["tot_cur_bal"] = np.log(df["tot_cur_bal"])
    df.replace([np.inf, -np.inf], 0, inplace=True)
    std_scaler = StandardScaler()
    df[numerical_var] = std_scaler.fit_transform(df[numerical_var])
    for var in numerical_var:
        print("Descriptive statistics for variable {}".format(var))
        print(df.groupby("loan_status")[var].describe())
    return df


def categorical_transform(df, categorical_var):
    print("Transform categorical variables: ")
    print(categorical_var)
    for var in categorical_var:
        dummies_df = pd.get_dummies(df[var], prefix=var, drop_first=True)
        df = pd.concat([df.drop(var, axis=1), dummies_df], axis=1)
    return df


def undersampling(df, df_non_default, pct):
    df_non_default = df_non_default.sample(
        int(len(df_non_default) * pct), random_state=1
    )
    df_default = df[df["loan_status"] == "Defaulted"]
    df = pd.concat([df_default, df_non_default], axis=0)
    # randomly assign
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    print(
        "After under-sampling the non-defaulted loans (*0.6), the dimension of the data is: "
    )
    print(df.shape)
    counts_defaulted = len(df[df["loan_status"] == "Defaulted"])
    counts_non_defaulted = len(df[df["loan_status"] == "Non-defaulted"])
    counts = [counts_defaulted, counts_non_defaulted]
    percentage = [i / len(df) * 100 for i in counts]
    print("Percentage of Defaulted and Non-defaulted loans: ")
    print(percentage)
    return df


def data_pre_propressing(logger):
    dsk_df = dt.fread("2007_to_2018Q4.csv")
    df = dsk_df.to_pandas()
    num_Columns = df.select_dtypes(exclude=["object"]).columns.tolist() + [
        "loan_status"
    ]
    print("Before data-preprocessing, the dimension of data: ")
    print(df.shape)
    print("Features in the dataset are: ")
    print(df.columns)
    """
    Task 1: Numerical: Drop the variables based on the correlation matrix plot
    """

    df = drop_correlated(df, num_Columns)
    print("Drop the variables based on the correlation matrix plot, the dimension: ")
    print(df.shape)
    """
    Task 2: Impute missing values
    """
    df = imputing_missing(df)

    """
    Task 3: Drop outlier and plot distribution for numerical values
    """

    df = drop_outliers_plot(df, logger)
    print("After dropping outliers the dimension:")
    print(df.shape)

    """
    Task 4: Categorical: Drop the columns that have imbalance for categories of each variable and imbalance for loan_status     
    Information value criteria: Information Value gives us the strength of a predictor, i.e., how strongly or weakly will it be able predict a class.
    """
    df = drop_barchart(df)
    print(
        "After dropping weak categorical columns, the dimension is: {}".format(df.shape)
    )
    """
    Task 5: Data transformation: 
    - take log for numerical variables if the distribution is right-skewed
    - Standardize for numerical variables
    - Create dummies for categorical variables:
    
    """
    numerical_var = df.select_dtypes(exclude=["object"]).columns.tolist()
    df = numerical_transform(df, numerical_var)
    categorical_var = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_var.remove("loan_status")
    df = categorical_transform(df, categorical_var)

    # Plot the distribution of loan status
    counts_defaulted = len(df[df["loan_status"] == "Defaulted"])
    counts_non_defaulted = len(df[df["loan_status"] == "Non-defaulted"])
    counts = [counts_defaulted, counts_non_defaulted]
    percentage = [i / len(df) * 100 for i in counts]
    print("Percentage of Defaulted and Non-defaulted loans: ")
    print(percentage)
    sns.countplot(x="loan_status", data=df)
    plt.title("Loan status")
    plt.tight_layout()
    plt.show()
    ### Under-sampling for Non-defaulted, because it is out of proportion
    df_non_default = df[df["loan_status"] == "Non-defaulted"]
    df = undersampling(df, df_non_default, 0.6)
    dsk_logistic = dt.Frame(df)
    dsk_logistic.to_csv("cleaned_accepted_2007_to_2018Q4.csv")
