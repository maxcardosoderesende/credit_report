# trin_model.py

from utils.logger import logger
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score


def model_gbm(df_X, target):

    logger.info("Starting GBM model training...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df_X, target, test_size=0.2, random_state=42, stratify=target
    )

    # Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        ))
    ])

    # Train
    pipeline.fit(X_train, y_train)

    logger.info("GBM Model training completed")

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)

    logger.info("Model AUC: {:.4f}", auc)
    logger.debug("\n{}", classification_report(y_test, y_pred))

    # Save model
    BASE_DIR = os.path.abspath("..")
    MODEL_DIR = os.path.join(BASE_DIR, "model")

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "credit_model.pkl")
    joblib.dump(pipeline, model_path)
    logger.success("Model successfully saved at {}", model_path)

    # Save feature list
    feature_columns = list(X_train.columns)
    feature_path = os.path.join(MODEL_DIR, "feature_columns.pkl")
    joblib.dump(feature_columns, feature_path)
    logger.success("Feature list saved at {}", feature_path)

    # Save feature importance
    feature_importance = pipeline.named_steps['classifier'].feature_importances_
    logger.info("Feature importance computed")

    return pipeline, auc, feature_importance, feature_columns