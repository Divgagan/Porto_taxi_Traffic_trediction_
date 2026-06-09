# API_KEY = "AIzaSyBjls59IknGLPsSNof5otEzQqTTlTqsn2M"  # <-- PUT YOUR KEY HERE


import pandas as pd
import joblib
import googlemaps
import polyline
import numpy as np
import pytz
import warnings
import os
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template
from flask_caching import Cache

# --- 1. CONFIGURATION & MODEL LOADING ---
print("Loading 'GRID' models and API keys...")
API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "AIzaSyBjls59IknGLPsSNof5otEzQqTTlTqsn2M")
gmaps = googlemaps.Client(key=API_KEY, timeout=5, retry_timeout=5)

grid_model = joblib.load('xgb_speed_model_prod.pkl')
# Older pickled XGBoost models may miss attributes expected by newer xgboost.
if not hasattr(grid_model, 'feature_weights'):
    grid_model.feature_weights = None

# Force CPU inference to prevent CUDA device mismatch crash in Flask.
try:
    grid_model.set_params(device='cpu')
except Exception as e:
    print(f"Warning: could not force XGBoost CPU mode: {e}")
lon_bins = joblib.load('lon_bins_1km.pkl')
lat_bins = joblib.load('lat_bins_1km.pkl')

app = Flask(__name__)
PORTO_TZ = pytz.timezone('Europe/Lisbon')

# --- 2. NEW: CONFIGURE CACHE ---
app.config["CACHE_TYPE"] = "simple" # Use a simple in-memory cache
app.config["TEMPLATES_AUTO_RELOAD"] = True
cache = Cache(app)
print("ROOT PATH:", app.root_path)
print("TEMPLATE FOLDER:", app.template_folder)
print("INDEX PATH:", os.path.join(app.root_path, app.template_folder, 'index.html'))
# ------------------------------

# --- 3. HELPER FUNCTIONS ---
@cache.memoize(timeout=3600)
def get_route_data(source, dest):
    """Fetch and cache Google route geometry so repeated dashboard runs are fast."""
    directions = gmaps.directions(source, dest, mode="driving")
    route_points = [p for p in polyline.decode(directions[0]['overview_polyline']['points'])]
    dist_meters = directions[0]['legs'][0]['distance']['value']
    return route_points, dist_meters / 1000.0

# --- 4. NEW: Add Caching Decorator ---
@cache.memoize(timeout=3600) # Cache this function for 1 hour
def get_ml_speed(route_points_tuple, hour, day):
    """Gets the *average* ML speed score (km/h) for a route."""
    # Convert tuple back to list for processing
    route_points = list(route_points_tuple) 
    
    if not route_points: return 0
    unique_cells = set()
    lon_grid_size, lat_grid_size = len(lon_bins), len(lat_bins)

    for lat, lon in route_points:
        lon_bin = np.digitize(lon, lon_bins)
        lat_bin = np.digitize(lat, lat_bins)
        if (0 < lon_bin < lon_grid_size) and (0 < lat_bin < lat_grid_size):
            unique_cells.add((lon_bin, lat_bin))
            
    if not unique_cells: return 0
    features_list = []
    for lon_bin, lat_bin in unique_cells:
        features_list.append({'hour': hour, 'day': day, 'lon_bin': lon_bin, 'lat_bin': lat_bin})
    
    X_predict = pd.DataFrame(features_list)
    predictions = grid_model.predict(X_predict)
    return np.mean(predictions)

# --- 5. NEW: Add Caching Decorator ---
@cache.memoize(timeout=3600) # Cache this function for 1 hour
def get_google_travel_time(source, dest, day, hour):
    """Gets Google's predicted travel time for a *future* day/hour."""
    now = PORTO_TZ.localize(datetime.now())
    target_day = int(day)
    days_ahead = (target_day - now.weekday() + 7) % 7
    target_dt = (now + timedelta(days=days_ahead)).replace(hour=hour, minute=0, second=0, microsecond=0)
    
    if target_dt < now:
        target_dt += timedelta(days=7)
        
    try:
        directions_result = gmaps.directions(source, dest, mode="driving", departure_time=target_dt)
        leg = directions_result[0]['legs'][0]
        # Use duration_in_traffic if available, otherwise fall back to regular duration
        if 'duration_in_traffic' in leg:
            duration_sec = leg['duration_in_traffic']['value']
        else:
            duration_sec = leg['duration']['value']
        return duration_sec / 60
    except Exception as e:
        print(f"Google API error (hour={hour}): {e}")
        return -1

# --- 6. API ENDPOINTS ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/plan_trip', methods=['POST'])
def plan_trip():
    try:
        data = request.json
        source = data['source']
        dest = data['destination']
        day = int(data['day'])
        
        route_points = []
        dist_km = 1.0 # default fallback
        try:
            route_points, dist_km = get_route_data(source, dest)
        except Exception as e:
            print(f"Error getting route polyline: {e}")
            
        plan = []
        for hour in range(24):
            try:
                avg_speed = get_ml_speed(tuple(route_points), hour, day)
                if avg_speed <= 0: avg_speed = 30.0
                ai_travel_time_min = (dist_km / avg_speed) * 60.0
                
                departure_time = f"{hour:02d}:00"
                arrival_dt = datetime(2025, 1, 1, hour, 0) + timedelta(minutes=ai_travel_time_min)
                arrival_time = arrival_dt.strftime("%H:%M")
                
                plan.append({
                    'hour': hour,
                    'departure_time': departure_time,
                    'travel_time_min': round(ai_travel_time_min),
                    'arrival_time': arrival_time,
                    'congestion_score': round(avg_speed),
                })
            except Exception as e:
                print(f"Error processing hour {hour}: {e}")
                plan.append({
                    'hour': hour, 'departure_time': f"{hour:02d}:00",
                    'travel_time_min': -1, 'arrival_time': 'Err',
                    'congestion_score': 0
                })
        
        # Physics-based free-flow baseline (50 km/h = no traffic)
        # This replaces the Google API call which was crashing the server
        free_flow_eta = round((dist_km / 50.0) * 60)
        
        return jsonify({ 'route_path': route_points, 'plan': plan, 'google_peak_eta': f"{free_flow_eta} (No-Traffic)" })
    
    except Exception as e:
        print(f"FATAL plan_trip error: {e}")
        return jsonify({'error': str(e), 'plan': [], 'route_path': []}), 500

# --- 8. RUN THE APP ---
if __name__ == '__main__':
    app.run(debug=False, use_reloader=False, threaded=True, port=5000)
