# prepare_data.py

import pandas as pd
import numpy as np
from utils.logger import logger
import sys


def load_and_prepare_data(filepath='../data/application_train.csv'):
    logger.info("Prepare data")
    logger.info("Loading dataset from {}", filepath)

    df = pd.read_csv(filepath)
    
    selected_features = [
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
        'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH',
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
        'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
        'CNT_CHILDREN', 'NAME_EDUCATION_TYPE',
    ]

    X = df[selected_features].copy()
    y = df['TARGET'].copy()

    X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].replace(365243, np.nan)

    X['AGE_YEARS'] = (-X['DAYS_BIRTH'] / 365.25).round(1)
    X['YEARS_EMPLOYED'] = (-X['DAYS_EMPLOYED'] / 365.25).round(1)
    X['YEARS_ID_PUBLISH'] = (-X['DAYS_ID_PUBLISH'] / 365.25).round(1)
    X = X.drop(columns=['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH'])

    X['CODE_GENDER'] = X['CODE_GENDER'].map({'M': 0, 'F': 1}).fillna(0).astype(int)
    X['FLAG_OWN_CAR'] = X['FLAG_OWN_CAR'].map({'N': 0, 'Y': 1}).astype(int)
    X['FLAG_OWN_REALTY'] = X['FLAG_OWN_REALTY'].map({'N': 0, 'Y': 1}).astype(int)

    education_map = {
        'Lower secondary': 0, 'Secondary / secondary special': 1,
        'Incomplete higher': 2, 'Higher education': 3, 'Academic degree': 4,
    }

    X['EDUCATION_LEVEL'] = X['NAME_EDUCATION_TYPE'].map(education_map).fillna(1).astype(int)
    X = X.drop(columns=['NAME_EDUCATION_TYPE'])

    X = X.fillna(X.median())

    X['CREDIT_INCOME_RATIO'] = X['AMT_CREDIT'] / (X['AMT_INCOME_TOTAL'] + 1)
    X['ANNUITY_INCOME_RATIO'] = X['AMT_ANNUITY'] / (X['AMT_INCOME_TOTAL'] + 1)
    X['CREDIT_GOODS_RATIO'] = X['AMT_CREDIT'] / (X['AMT_GOODS_PRICE'] + 1)
    
    logger.info(f"Data prepared: X shape={X.shape}, y shape={y.shape}")
    logger.info("Default rate: {:.2%}", y.mean())
    return X, y