import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

def object_features_report(dataframe:pd.DataFrame):
    type_features = dataframe.select_dtypes(include=["object"]).columns

    print("********************Unique Counts********************")
    uniq_counts = dataframe[type_features].nunique()
    print(uniq_counts)

    print("\n********************Unique Values********************")
    for f in type_features:
        uniq_values = dataframe[f].unique()
        print(f"Feature '{f}': {uniq_values}")

    print("\n********************Value Counts********************")
    for f in type_features:
        val_counts = dataframe[f].value_counts()
        print(f"Feature '{f}' values: {val_counts}\n")


def split_dataset(target_feature:str, dataframe:pd.DataFrame, test_percentage:float, seed:int, stratify_feature:str):
    X_train, X_test, y_train, y_test = train_test_split(dataframe.drop(target_feature, axis=1),dataframe[target_feature],
                                                    test_size=test_percentage, random_state=seed,
                                                    stratify=dataframe[stratify_feature])
    
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    return X_train, X_test, y_train, y_test


def encoding_nominal_features(features_train:pd.DataFrame, features_test:pd.DataFrame, nominal_features:List):
    oh_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first")

    oh_encoder.fit(features_train[nominal_features])
    X_train_encoded = oh_encoder.transform(features_train[nominal_features])
    X_test_encoded = oh_encoder.transform(features_test[nominal_features])

    feature_names = oh_encoder.get_feature_names_out(input_features=nominal_features)
    X_train_df = pd.DataFrame(X_train_encoded, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_encoded, columns=feature_names)

    left_features = [f for f in features_train.columns if f not in nominal_features]
    X_train = pd.concat([features_train[left_features].reset_index(drop=True), X_train_df], axis=1)
    X_test = pd.concat([features_test[left_features].reset_index(drop=True), X_test_df], axis=1)

    print("One Hot Encoding with handle_unknown='ignore' and drop='first' done.")
    return X_train, X_test


def encoding_ordinal_features(features_train:pd.DataFrame, features_test:pd.DataFrame, ordinal_features:List, categories_list:Dict):
    ord_encoder = OrdinalEncoder(categories=categories_list, handle_unknown="use_encoded_value", unknown_value=-1)

    ord_encoder.fit(features_train[ordinal_features])
    features_train[ordinal_features] = ord_encoder.transform(features_train[ordinal_features])
    features_test[ordinal_features]= ord_encoder.transform(features_test[ordinal_features])

    print("Ordinal Encoding with handle_unknown='use_encoded_value' and unknown_value=-1 done.")
    return features_train, features_test


def scaling_numerical_features(features_train:pd.DataFrame, features_test:pd.DataFrame, features_to_scale:List):
    mm_scaler = MinMaxScaler()
    mm_scaler.fit(features_train[features_to_scale])
    features_train[features_to_scale] = mm_scaler.transform(features_train[features_to_scale])
    features_test[features_to_scale] = mm_scaler.transform(features_test[features_to_scale])

    print("Min Max Scaling done.")
    return features_train, features_test


def graphing_correlation_matrix(dataframe:pd.DataFrame, zoom:int=1):
    plt.figure(figsize=(4*zoom, 3*zoom))
    sns.heatmap(
        dataframe.corr(),
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
    )
    plt.title("Correlation Matrix")


def evaluating_model(model:any, model_name:str, X:pd.DataFrame, y:pd.DataFrame, label_data:str):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"**********{model_name} Metrics ({label_data}):**********")
    print(f"Root Mean Squared Error: {np.sqrt(mse):.4f}")
    print(f"Square R: {r2:.4f}")


def visualizing_model_performance(model:any, model_name:str, X:pd.DataFrame, y:pd.DataFrame, label_data:str):
    plt.figure(figsize=(6, 4))
    plt.scatter(y, model.predict(X), alpha=0.5, label=label_data, s=10)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--", linewidth=2, label="Perfect Prediction")
    plt.xlabel("Real Prices")
    plt.ylabel("Predicted Prices")
    plt.title(f"{model_name} - Real vs Predicted Prices - {label_data}")
    plt.legend()
    plt.grid(True)