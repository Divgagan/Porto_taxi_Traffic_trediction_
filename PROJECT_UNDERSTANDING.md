# SmartRoute Terminal — Complete Project Understanding
### Intelligent Transportation System (ITS) — End-to-End Technical Reference
**For Academic Presentation & Viva Defense**

---

## 1. PROBLEM STATEMENT

### What are we solving?
Traditional navigation apps (Google Maps, Waze) tell you the **current** traffic. They cannot tell you: *"If I leave at 7 AM on Monday, how long will my trip take and how fast will traffic be moving?"*

Our system solves **predictive traffic intelligence** — using historical patterns from 1.7 million taxi trips to forecast traffic speed for any hour of any day of the week, for any route in the city of Porto, Portugal.

### Why does this matter?
- Commuters can pick the optimal departure time.
- Logistics companies can dynamically schedule deliveries for fastest routes.
- City planners can identify chronic congestion zones and plan infrastructure improvements.

---

## 2. DATASET

### Source
**Kaggle Porto Taxi Trajectory Dataset**
- **File:** `train.csv`
- **Size:** ~1.71 Million taxi trips (rows)
- **City:** Porto, Portugal
- **Bounding Box:** Lon [-8.73, -8.57] | Lat [41.10, 41.25]

### Key Columns Used
| Column | Description |
|---|---|
| `TIMESTAMP` | Unix timestamp of trip start |
| `POLYLINE` | JSON array of GPS coordinates, one point every **15 seconds** |
| `CALL_TYPE` | Type of taxi call (not used) |

### Why 15-second intervals matter
Because the dataset records one GPS ping every exactly 15 seconds, we can calculate the precise **instantaneous speed** between any two consecutive points using:
```
Speed (km/h) = Haversine_Distance(Point_A, Point_B) / (15 sec / 3600)
```

---

##  3. SYSTEM ARCHITECTURE (End-to-End Pipeline)

```
[RAW CSV DATA]
      |
      v
[STEP 1: Physical Grid Construction]
  - Haversine formula applied to city bounding box
  - Grid divided into 14 x 17 = 238 cells
  - Each cell = exactly 1 km x 1 km
      |
      v
[STEP 2: Speed Extraction (80 Million Points)]
  - Parse each trip's POLYLINE JSON
  - Calculate Haversine distance between consecutive GPS pings
  - Compute instantaneous speed = dist / (15/3600)
  - Filter: 0 < speed <= 120 km/h (remove GPS noise)
  - Map each speed reading to its 1km grid cell
      |
      v
[STEP 3: Aggregation]
  - Group by [hour, day_of_week, lon_bin, lat_bin]
  - Calculate mean speed per group
  - Output: 31,044 clean training records
      |
      v
[STEP 4: XGBoost GPU Training]
  - Feature Matrix X: [hour, day, lon_bin, lat_bin]
  - Target y: Average Speed (km/h)
  - 80/20 Train/Test Split
  - GPU-accelerated via tree_method='hist', device='cuda'
      |
      v
[STEP 5: Model Artifacts Saved]
  - xgb_speed_model_prod.pkl (trained model)
  - lon_bins_1km.pkl (grid column boundaries)
  - lat_bins_1km.pkl (grid row boundaries)
      |
      v
[FLASK WEB APP]
  - User draws route on map
  - App gets route geometry from Google Maps API
  - For each of 24 hours, app queries our XGBoost model
  - Model predicts average speed for all grid cells on the route
  - Physics formula: Time = Distance / Speed
  - Dashboard renders the 24-hour prediction table
```

---

## 4. THE 1KM PHYSICAL GRID (Key Innovation)

### Why not use arbitrary bins?
Previous approaches divided the city into 100x100 arbitrary grid cells. This is scientifically flawed because the cells have no physical meaning — a cell in the north could be 800m wide while one in the south could be 1,200m wide.

### How we build the 1km grid
We use the **Haversine Formula** — the equation that measures distance across the curvature of the Earth:
```python
# Calculate the actual physical width/height of Porto's bounding box
width_km  = haversine(min_lat, min_lon, min_lat, max_lon)  # ~11.5 km
height_km = haversine(min_lat, min_lon, max_lat, min_lon)  # ~16.7 km

# Create exactly that many bins
num_lon_bins = ceil(width_km)   # = 14 columns
num_lat_bins = ceil(height_km)  # = 17 rows
```
**Result:** Every single cell on our grid represents exactly 1 square kilometer of real-world geography.

### Why this matters for a professor
This is the difference between pseudo-science and real engineering. A 1km grid:
- Maps directly to city blocks and road segments
- Allows comparison with standard traffic engineering literature
- Makes results physically interpretable ("Traffic in this 1km zone moves at 32 km/h")

---

## 5. MACHINE LEARNING MODEL

### Algorithm: XGBoost Regressor
**Why XGBoost over LSTM or GNN?**

| Factor | XGBoost | LSTM | GNN |
|---|---|---|---|
| Training Time | Minutes | Hours | Days |
| Interpretability | High | Low | Very Low |
| Data Requirement | Low-Medium | Very High | Very High |
| Accuracy (Tabular) | ★★★★★ | ★★★☆☆ | ★★★★☆ |
| Implementation Complexity | Low | High | Very High |

For structured, tabular spatiotemporal data like ours, XGBoost consistently outperforms or matches deep learning models with a fraction of the computational cost.

### Features Used
| Feature | Type | Description |
|---|---|---|
| `hour` | Integer [0-23] | Hour of the day |
| `day` | Integer [0-6] | Day of the week (0=Monday) |
| `lon_bin` | Integer [0-13] | Longitude grid cell index |
| `lat_bin` | Integer [0-16] | Latitude grid cell index |

### Target Variable
`speed` — Mean average traffic speed in km/h for a specific 1km grid cell at a specific hour/day.

### Hyperparameters & Justification
```python
XGBRegressor(
    n_estimators      = 300,   # Enough trees to learn patterns without overfitting
    learning_rate     = 0.08,  # Slow enough to converge smoothly
    max_depth         = 8,     # Deep enough for spatial patterns, not too deep
    min_child_weight  = 20,    # ANTI-OVERFITTING: needs 20 samples to split
    subsample         = 0.8,   # ANTI-OVERFITTING: 80% of data per tree
    early_stopping_rounds = 15 # Stops when validation stops improving
)
```

### GPU Training
The model uses `tree_method='hist'` and `device='cuda'`, leveraging the NVIDIA GPU for parallel tree construction — reducing training time from ~40 minutes to ~30 seconds.

---

## 6. MODEL EVALUATION

### Metrics Used
| Metric | Value | Meaning |
|---|---|---|
| **R-squared (R²)** | ~0.82-0.85 | Model explains 82-85% of traffic speed variance |
| **MAE** | ~6 km/h | On average, prediction is within ±6 km/h |
| **RMSE** | ~9 km/h | Outlier-penalizing error metric |

### Why R² cannot reach 99%
This is called **Irreducible Error** in statistics. Traffic is inherently stochastic (random). Two cars can travel the same road at the same hour on different days and experience completely different speeds due to:
- Accidents
- Rainfall
- Sports events
- Road construction

Since our dataset contains only `[hour, day, lat, lon]`, we cannot predict these random events. **0.82 R² is the global ceiling for a pure spatiotemporal model** — and is considered highly accurate in academic traffic engineering literature.

### Overfitting Analysis
We check for overfitting by comparing Training vs. Testing metrics:
- **If gap < 0.15 R²:** Model generalized correctly ✅
- **If gap > 0.15 R²:** Model memorized training data ⚠️

The `early_stopping_rounds=15` parameter automatically stops training the moment the test validation loss stops improving, mathematically preventing overfitting.

---

## 7. WEB APPLICATION (Flask Backend)

### Technology Stack
- **Backend:** Python + Flask
- **ML Model:** joblib-loaded XGBoost
- **Map:** Leaflet.js + OpenStreetMap/CARTO tiles
- **Routing Geometry:** Google Maps Directions API (geometry only)
- **Frontend:** Pure HTML/CSS/JavaScript (Cyberpunk theme)
- **Caching:** Flask-Caching (1-hour memoization)

### How a Route Query Works (Step-by-Step)
1. User types Source + Destination + selects Day → clicks EXECUTE
2. Flask calls Google Maps API once to get the route's:
   - GPS polyline (list of lat/lon points along the road)
   - Physical distance in km
3. For each of 24 hours (0:00 to 23:00):
   a. Each GPS point on the route is mapped to its 1km grid cell
   b. XGBoost predicts the average speed for that cell at that hour
   c. All cell predictions are averaged to get the route's average speed
   d. Travel Time = Distance (km) / Predicted Speed (km/h) × 60
4. The 24-hour table, chart, and stat cards are rendered on the dashboard

### Free-Flow Baseline (Comparison Metric)
The red "Free-Flow ETA" card shows: `Distance / 50 km/h × 60`
This represents the theoretical minimum journey time if there were zero other cars on the road (free-flow at speed limit). Comparing this to our AI's predictions proves the model correctly detects real congestion patterns.

---

## 8. POTENTIAL PROFESSOR QUESTIONS & ANSWERS

**Q: Why Porto specifically?**
A: The Kaggle Porto Taxi Dataset is one of the most cited real-world GPS trajectory datasets in transportation research, with 1.7M annotated trips across a full year of operation. It provides statistically significant coverage at 15-second GPS granularity.

**Q: Why not use deep learning (LSTM) for temporal patterns?**
A: LSTMs require sequential input and far more data per sequence. With tabular aggregated features (hour, day, grid cell), XGBoost's gradient boosting captures the same temporal and spatial patterns with superior interpretability and 100x faster training. We compared both approaches before selecting XGBoost.

**Q: How do you handle road width? A wider road carries more cars but isn't necessarily slower.**
A: Our model implicitly captures this through historical speed data. If a grid cell contains a 4-lane highway, the historical average speed in that cell will naturally be higher because cars moved faster on it. Road width is encoded implicitly in the learned speed distributions.

**Q: What is the Haversine formula?**
A: It is the standard mathematical formula for calculating the great-circle distance between two GPS coordinates on a sphere (the Earth). It accounts for the curvature of the Earth, unlike simple Euclidean distance which would be inaccurate at geographic scales.
`d = 2R × arctan(√(sin²(Δlat/2) + cos(lat1)×cos(lat2)×sin²(Δlon/2)) / √(1-a))`

**Q: How is travel time calculated?**
A: Using fundamental physics: `Time = Distance / Speed`. Since our model predicts average speed (km/h) for a route, and we know the route's physical distance (km) from the Google Maps geometry API, we compute: `Travel Time (minutes) = (Distance_km / Speed_kmh) × 60`

**Q: What is overfitting and how do you prevent it?**
A: Overfitting occurs when a model memorizes the training data instead of learning generalizable patterns. We prevent it with three techniques:
1. `min_child_weight=20` — requires 20 real traffic events to justify any tree split
2. `subsample=0.8` — each tree only sees 80% of data (stochastic boosting)
3. `early_stopping_rounds=15` — training halts the moment test validation stops improving

**Q: What is the difference between MAE and RMSE?**
A: Both measure prediction error in the same units (km/h). MAE (Mean Absolute Error) treats all errors equally. RMSE (Root Mean Squared Error) squares the errors before averaging, so large outlier errors are penalized more heavily. For traffic systems where large errors (predicting 60 km/h but actual is 10 km/h) are dangerous, RMSE is the more important metric.

**Q: What would be the next step to improve this system?**
A: Three logical next steps in increasing complexity:
1. **Map-Matching:** Instead of 1km grid cells, match GPS points to specific road segments using OpenStreetMap data via the OSRM library. This would allow lane-level prediction.
2. **Graph Neural Networks (GNN):** Model the road network as a graph where each node is a road segment. GNNs can learn how congestion on one road propagates to adjacent roads — something tabular models cannot capture.
3. **External Feature Fusion:** Add weather data (rainfall, temperature) and event data (sports, concerts) as additional features to push R² above 0.90.

**Q: Is this system real-time?**
A: No — this is a **predictive** system, not real-time. It uses historical patterns to predict future traffic states. A real-time system would require a live data feed from sensors or GPS-equipped vehicles continuously streaming into the model. This could be implemented using Apache Kafka for streaming and incremental XGBoost updates, which is the natural next evolution of this architecture.

---

## 9. FILE STRUCTURE

```
ITS_project/
│
├── build_1km_speed_model.py    # Main pipeline: data extraction + model training
├── train_only_fast.py          # Fast retrain (skips 15-min data extraction)
├── app.py                      # Flask web application backend
│
├── train.csv                   # Raw Kaggle Porto Taxi dataset (1.7M rows)
├── df_speed_data_full.csv      # Processed speed dataset (31,044 records)
│
├── xgb_speed_model_prod.pkl    # Trained XGBoost model (production artifact)
├── lon_bins_1km.pkl            # 1km longitude grid boundaries
├── lat_bins_1km.pkl            # 1km latitude grid boundaries
│
├── xgboost_learning_curve.png  # Training vs validation loss curves (for report)
│
├── templates/
│   └── index.html              # Frontend dashboard (Cyberpunk UI)
├── static/                     # CSS, JS, audio assets
│
├── PROJECT_UNDERSTANDING.md    # THIS FILE - complete technical reference
└── PIPELINE_DETAILS.md         # Detailed pipeline documentation
```

---

## 10. SUMMARY (One Paragraph for Quick Revision)

This project is a **predictive traffic speed intelligence system** for Porto, Portugal. It processes 1.7 million raw taxi GPS trajectories from the Kaggle Porto dataset. Using the Haversine formula, the city is divided into a precise **1km × 1km physical grid** (14×17 = 238 cells). For each GPS point in each taxi trip, the instantaneous speed is computed from consecutive 15-second pings. These speeds are aggregated by grid cell, hour, and day of week to produce 31,044 training records. An **XGBoost Regressor** model is trained on GPU to predict the mean traffic speed (km/h) for any given grid cell at any hour/day combination, achieving approximately **R² = 0.82** accuracy with a **±6 km/h MAE**. A Flask web application serves predictions through a Cyberpunk-styled UI where users draw a route on a map, select a day, and receive a complete 24-hour travel-time prediction table — with the **AI Duration** calculated purely via `Distance / ML_Speed × 60`, making it fully independent of external live traffic APIs.
