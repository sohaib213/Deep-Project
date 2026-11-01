import pandas as pd
import numpy as np

ORIGIN_MAP = {'Torgersen': 0, 'Biscoe': 1, 'Dream': 2}

def load_and_preprocess_task2(dataset_path="penguins.csv"):
    df = pd.read_csv(dataset_path)

    features = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass', 'OriginLocation']
    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    df['OriginLocation'] = df['OriginLocation'].map(ORIGIN_MAP)

    classes = ['Adelie', 'Chinstrap', 'Gentoo']
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for i, cls in enumerate(classes):
        cls_data = df[df['Species'] == cls].drop_duplicates().reset_index(drop=True)

        label = np.zeros(3)
        label[i] = 1.0
        cls_data['Label'] = [label] * len(cls_data)

        train_data = pd.concat([train_data, cls_data.iloc[:30]], ignore_index=True)
        test_data = pd.concat([test_data, cls_data.iloc[30:50]], ignore_index=True)

    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)

    X_train = train_data[features].values.astype(float)
    y_train = np.stack(train_data['Label'].values)
    X_test = test_data[features].values.astype(float)
    y_test = np.stack(test_data['Label'].values)

    scaler_mean = np.mean(X_train, axis=0)
    scaler_std = np.std(X_train, axis=0)
    scaler_std[scaler_std == 0] = 1.0

    X_train = (X_train - scaler_mean) / scaler_std
    X_test = (X_test - scaler_mean) / scaler_std

    return X_train, y_train, X_test, y_test, scaler_mean, scaler_std, classes