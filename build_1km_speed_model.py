import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import matplotlib.pyplot as plt

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance in kilometers between two points on the earth."""
    R = 6371.0 # Earth radius in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def build_model():
    print("--- STARTING 1KM SPEED PIPELINE ---")
    
    # 1. Coordinate Boundaries for Porto
    min_lon, max_lon = -8.73, -8.57
    min_lat, max_lat = 41.10, 41.25

    # 2. Scale down Grid to precisely 1 km x 1 km
    print("Step 1: Calculating 1km physical grid...")
    width_km = haversine(min_lat, min_lon, min_lat, max_lon)
    height_km = haversine(min_lat, min_lon, max_lat, min_lon)
    
    num_lon_bins = int(np.ceil(width_km))
    num_lat_bins = int(np.ceil(height_km))
    
    lon_bins = np.linspace(min_lon, max_lon, num_lon_bins + 1)
    lat_bins = np.linspace(min_lat, max_lat, num_lat_bins + 1)
    
    joblib.dump(lon_bins, 'lon_bins_1km.pkl')
    joblib.dump(lat_bins, 'lat_bins_1km.pkl')
    
    print(f"Grid created: {num_lon_bins} columns x {num_lat_bins} rows (Each cell is ~1km²)")
    
    # 3. Process Large Dataset in Chunks
    print("\nStep 2: Processing GPS sequences into physical speeds...")
    print("This will take a few minutes. Extracting speeds from 1.7 Million trips...")
    
    all_data = []
    chunk_num = 1
    start_time = time.time()
    
    # We load chunksize=100,000 for RAM efficiency
    for chunk in pd.read_csv('train.csv', chunksize=100000):
        print(f"  -> Processing Chunk {chunk_num} (Elapsed Time: {round(time.time()-start_time)}s)...")
        chunk = chunk[chunk['POLYLINE'] != '[]'].copy()
        
        # Parse Dates
        chunk['DATETIME'] = pd.to_datetime(chunk['TIMESTAMP'], unit='s')
        chunk['hour'] = chunk['DATETIME'].dt.hour
        chunk['day'] = chunk['DATETIME'].dt.dayofweek
        
        # Vectorized Polyline Parsing
        chunk['POLYLINE'] = chunk['POLYLINE'].apply(json.loads)
        
        for idx, row in chunk.iterrows():
            pts = row['POLYLINE']
            if len(pts) < 2:
                continue # Need at least 2 points to calculate speed
                
            # Convert list of pairs to numpy array
            pts_arr = np.array(pts)
            lons = pts_arr[:, 0]
            lats = pts_arr[:, 1]
            
            # Calculate distance between every consecutive point
            dist_km = haversine(lats[:-1], lons[:-1], lats[1:], lons[1:])
            
            # Calculate speed: Distance / (15 seconds / 3600 seconds)
            speeds = dist_km / (15.0 / 3600.0)
            
            # Filter clean data: Speeds > 0 and <= 120 km/h (highway speed limit)
            valid_mask = (speeds > 0) & (speeds <= 120) & \
                         (lons[:-1] >= min_lon) & (lons[:-1] <= max_lon) & \
                         (lats[:-1] >= min_lat) & (lats[:-1] <= max_lat)
            
            if not np.any(valid_mask):
                continue
            
            valid_lons = lons[:-1][valid_mask]
            valid_lats = lats[:-1][valid_mask]
            valid_speeds = speeds[valid_mask]
            
            # Bin the coordinates into our 1km grid
            lon_b = np.digitize(valid_lons, lon_bins) - 1
            lat_b = np.digitize(valid_lats, lat_bins) - 1
            
            # Store observations
            hour = row['hour']
            day = row['day']
            for lb, lab, s in zip(lon_b, lat_b, valid_speeds):
                all_data.append([hour, day, lb, lab, s])
                
        chunk_num += 1

    print("Data extraction complete!")
    
    # 4. Aggregation
    print("\nStep 3: Aggregating Average Speeds...")
    df = pd.DataFrame(all_data, columns=['hour', 'day', 'lon_bin', 'lat_bin', 'speed'])
    
    # We find the mean speed of all cars in a specific 1km cell at a specific hour
    df_grouped = df.groupby(['hour', 'day', 'lon_bin', 'lat_bin'])['speed'].mean().reset_index()
    
    df_grouped.to_csv('df_speed_data_full.csv', index=False)
    print(f"Generated physical speed dataset! Records: {len(df_grouped)}")
    
    # 5. Model Training (GPU Enabled)
    print("\nStep 4: Training XGBoost Speed Predictor...")
    features = ['hour', 'day', 'lon_bin', 'lat_bin']
    target = 'speed'
    
    X = df_grouped[features]
    y = df_grouped[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Using 'hist' tree_method and 'cuda' device to utilize local GPU
    print("Initializing GPU for XGBoost...")
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
        print(f"GPU initialization failed. Falling back to CPU. Reason: {e}")
        model = XGBRegressor(
            n_estimators=300, 
            learning_rate=0.08, 
            max_depth=8, 
            min_child_weight=20,
            subsample=0.8,
            random_state=42, 
            n_jobs=-1,
            eval_metric='mae',
            early_stopping_rounds=15
        )
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    # 6. Evaluation (Checking for Overfitting)
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
    
    # 7. Generate Learning Curves for Report
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
        print(f"\n⚠️ Could not generate learning curve image: {e}")
    
    # 7. Save production assets
    joblib.dump(model, 'xgb_speed_model_prod.pkl')
    print("\n✅ Success! New Physical Pipeline Output: 'xgb_speed_model_prod.pkl'.")

if __name__ == '__main__':
    build_model()
