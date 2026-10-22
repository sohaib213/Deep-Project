import pandas as pd
import numpy as np

ORIGIN_MAP = {'Torgersen': 0, 'Biscoe': 1, 'Dream': 2}

def load_and_preprocess(feature1, feature2, class1, class2, dataset_path="penguins.csv"):
    df = pd.read_csv(dataset_path)
        
    df = df[df['Species'].isin([class1, class2])].copy()
    df['Label'] = np.where(df['Species'] == class1, 1, -1)
    
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    
    for cls in [class1, class2]:
        cls_data = df[df['Species'] == cls].sample(frac=1, random_state=42).reset_index(drop=True)
        train_data = pd.concat([train_data, cls_data.iloc[:30]], ignore_index=True)
        test_data = pd.concat([test_data, cls_data.iloc[30:50]], ignore_index=True)
        
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
    features_to_impute = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass', 'OriginLocation']
    
    for col in features_to_impute:
        if pd.api.types.is_numeric_dtype(train_data[col]):
            train_mean = train_data[col].mean()
            train_data[col] = train_data[col].fillna(train_mean)
            test_data[col] = test_data[col].fillna(train_mean)
        else:
            train_mode = train_data[col].mode()[0]
            train_data[col] = train_data[col].fillna(train_mode)
            test_data[col] = test_data[col].fillna(train_mode)
            
    train_data['OriginLocation'] = train_data['OriginLocation'].map(ORIGIN_MAP)
    test_data['OriginLocation'] = test_data['OriginLocation'].map(ORIGIN_MAP)
            
    X_train = train_data[[feature1, feature2]].values.astype(float)
    y_train = train_data['Label'].values
    X_test = test_data[[feature1, feature2]].values.astype(float)
    y_test = test_data['Label'].values
    
    scaler_mean = np.mean(X_train, axis=0)
    scaler_std = np.std(X_train, axis=0)
    scaler_std[scaler_std == 0] = 1.0  
    
    X_train = (X_train - scaler_mean) / scaler_std
    X_test = (X_test - scaler_mean) / scaler_std
    
    return X_train, y_train, X_test, y_test, scaler_mean, scaler_std