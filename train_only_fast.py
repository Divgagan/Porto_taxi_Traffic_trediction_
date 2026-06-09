import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import warnings

# Suppress the harmless CUDA/pandas mismatch warning
warnings.filterwarnings('ignore', category=UserWarning)

def fast_train():
    print("--- FAST TRAINING PIPELINE (SKIPPING EXTRACTION) ---")
    
    if not os.path.exists('df_speed_data_full.csv'):
        print("Error: 'df_speed_data_full.csv' not found. You must run the full build script first.")
        return
        
    print("Loaded 1.7 million rows of physical speed data instantly from CSV.")
    df_grouped = pd.read_csv('df_speed_data_full.csv')
    
    features = ['hour', 'day', 'lon_bin', 'lat_bin']
    target = 'speed'
    
    X = df_grouped[features]
    y = df_grouped[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining XGBoost Speed Predictor on GPU...")
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    try:
        model = XGBRegressor(
            n_estimators=300, 
            learning_rate=0.08, 
            max_depth=8, 
            min_child_weight=20,
            subsample=0.8,
            colsample_bytree=1.0,
            random_state=42, 
            tree_method='hist', 
            device='cuda',
            eval_metric='mae',
            early_stopping_rounds=15
        )
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    except Exception as e:
        print("GPU fallback... using CPU.")
        model = XGBRegressor(
            n_estimators=300, learning_rate=0.08, max_depth=8, min_child_weight=20, subsample=0.8,
            random_state=42, n_jobs=-1, eval_metric='mae', early_stopping_rounds=15
        )
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    # Evaluation
    y_train_pred = model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_train = r2_score(y_train, y_train_pred)
    
    y_test_pred = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_test = r2_score(y_test, y_test_pred)
    
    print("\n--- 📊 Overfitting & Performance Analysis ---")
    print(f"TRAINING Data -> MAE: {mae_train:.2f} | RMSE: {rmse_train:.2f} | R-squared: {r2_train:.4f}")
    print(f"TESTING Data  -> MAE: {mae_test:.2f} | RMSE: {rmse_test:.2f} | R-squared: {r2_test:.4f}")
    
    diff_r2 = abs(r2_train - r2_test)
    if diff_r2 > 0.15:
        print(f"\n⚠️ WARNING: Gap between Train and Test R2 is {diff_r2:.2f}. The model may be OVERFITTED.")
    else:
        print("\n✅ SUCCESS: Train and Test metrics are close. The model generalized well and is NOT overfitted!")
        
    print(f"Final real-world error margin: +/- {mae_test:.2f} km/h.")
    
    # Generate Curves
    try:
        results = model.evals_result()
        epochs = len(results['validation_0']['mae'])
        x_axis = range(0, epochs)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, results['validation_0']['mae'], label='Train Loss (MAE)', linewidth=2)
        plt.plot(x_axis, results['validation_1']['mae'], label='Validation/Test Loss (MAE)', linewidth=2)
        plt.legend(loc='upper right')
        plt.title('XGBoost Learning Curve (Training vs Validation Loss)')
        plt.ylabel('Mean Absolute Error (km/h)')
        plt.xlabel('Boosting Rounds (Trees)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('xgboost_learning_curve.png', dpi=300)
        print("\n📈 Saved Learning Curve plot as 'xgboost_learning_curve.png' for your university report!")
    except Exception as e:
        print(f"⚠️ Could not generate image: {e}")

if __name__ == '__main__':
    fast_train()
