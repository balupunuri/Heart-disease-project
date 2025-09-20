# heart_disease.py
# Standalone script to train several ML models on the heart disease dataset
# Usage: python heart_disease.py --data ./data/heart.csv --out model.joblib

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y

def build_pipelines():
    scaler = StandardScaler()
    rf = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', scaler),
                   ('clf', RandomForestClassifier(random_state=42))])
    lr = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', scaler),
                   ('clf', LogisticRegression(max_iter=1000, random_state=42))])
    svc = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', scaler),
                   ('clf', SVC(probability=True, random_state=42))])
    return {'random_forest': rf, 'logistic_regression': lr, 'svm': svc}

def main(args):
    X, y = load_data(args.data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipes = build_pipelines()

    best_models = {}
    for name, pipe in pipes.items():
        print(f"Training {name} ...")
        if name == 'random_forest':
            params = {'clf__n_estimators':[50,100], 'clf__max_depth':[None,5,10]}
        elif name == 'svm':
            params = {'clf__C':[0.1,1,10], 'clf__kernel':['rbf','linear']}
        else:
            params = {'clf__C':[0.1,1,10]}
        gs = GridSearchCV(pipe, params, cv=5, scoring='accuracy', n_jobs=-1)
        gs.fit(X_train, y_train)
        preds = gs.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(name, "best params:", gs.best_params_)
        print(classification_report(y_test, preds))
        best_models[name] = (gs.best_estimator_, acc)

    # choose best by accuracy and save
    best_name, (best_model, best_acc) = max(best_models.items(), key=lambda kv: kv[1][1])
    print(f"Best model: {best_name} with accuracy {best_acc:.4f}")
    joblib.dump(best_model, args.out)
    print("Saved model to", args.out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/heart.csv', help='path to CSV data file')
    parser.add_argument('--out', default='best_model.joblib', help='output path for saved model')
    args = parser.parse_args()
    main(args)
