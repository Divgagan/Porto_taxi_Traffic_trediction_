# Porto Taxi Traffic Prediction System

An intelligent transportation system that predicts traffic speed for any route in Porto, Portugal using historical taxi trajectory data and XGBoost machine learning models.

## 📋 Project Overview

This system solves **predictive traffic intelligence** - forecasting traffic speed for any hour of any day of the week, for any route in Porto using:
- **1.7 million taxi trips** from the Kaggle Porto dataset
- **XGBoost GPU-accelerated models** trained on 80 million GPS speed readings
- **1km × 1km physical grid** covering the entire city
- **Flask web application** with Google Maps integration

### Key Features
- ✅ Real-time traffic speed predictions for any route
- ✅ Hourly and day-of-week pattern analysis
- ✅ Scientific 1km physical grid (not arbitrary binning)
- ✅ Interactive web dashboard with route visualization
- ✅ Pre-trained models ready for inference

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda
- Google Maps API Key (for web app)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Divgagan/Porto_taxi_Traffic_trediction_.git
cd Porto_taxi_Traffic_trediction_
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Flask Web App

1. **Set your Google Maps API Key**
```bash
# On Windows (PowerShell):
$env:GOOGLE_MAPS_API_KEY = "YOUR_API_KEY_HERE"

# On Windows (Command Prompt):
set GOOGLE_MAPS_API_KEY=YOUR_API_KEY_HERE

# On macOS/Linux:
export GOOGLE_MAPS_API_KEY="YOUR_API_KEY_HERE"
```

2. **Run the Flask app**
```bash
python app.py
```

3. **Open in browser**
```
http://localhost:5000
```

4. **How to use:**
   - Click on the map to set source point
   - Click on another location to set destination
   - The app will display predicted travel time for each hour of the day
   - Interactive table shows average speed for each hour

---

## 📊 System Architecture

```
├── [Raw Data: train.csv (1.7M trips)]
│
├── STEP 1: Physical Grid Construction
│   └── Create 1km × 1km cells using Haversine formula
│
├── STEP 2: Speed Extraction (80M GPS points)
│   └── Parse polylines → calculate instantaneous speed
│
├── STEP 3: Aggregation
│   └── Group by [hour, day_of_week, lon_bin, lat_bin]
│   └── Output: 31,044 training records
│
├── STEP 4: XGBoost GPU Training
│   └── Features: [hour, day_of_week, longitude_bin, latitude_bin]
│   └── Target: Average speed (km/h)
│
└── STEP 5: Production Models
    ├── xgb_speed_model_prod.pkl (trained XGBoost model)
    ├── lon_bins_1km.pkl (grid column boundaries)
    └── lat_bins_1km.pkl (grid row boundaries)
```

---

## 📁 Project Structure

```
.
├── app.py                              # Flask web application
├── main.ipynb                          # Main analysis notebook
├── preprocessing.ipynb                 # Data preprocessing workflow
├── taxi_preprocessing.ipynb            # Taxi-specific preprocessing
├── build_1km_speed_model.py           # Model training script
├── train_only_fast.py                 # Fast training script
│
├── xgb_speed_model_prod.pkl           # Pre-trained XGBoost model
├── xgb_grid_model_prod.pkl            # Alternative grid model
├── lon_bins_1km.pkl                   # Longitude grid boundaries
├── lat_bins_1km.pkl                   # Latitude grid boundaries
│
├── df_model_data_full.csv             # Processed model features (1.1 MB)
├── df_model_data_full_01.csv          # Alternative processed data (9.7 MB)
├── df_speed_data_full.csv             # Aggregated speed data (850 KB)
│
├── templates/
│   └── index.html                     # Flask web app frontend
│
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── PROJECT_UNDERSTANDING.md            # Detailed technical documentation
├── PIPELINE_DETAILS.md                # Pipeline implementation details
└── RESEARCH_PAPER_CONTEXT.md          # Research paper background
```

---

## 📦 Dependencies

The project requires:
```
pandas          # Data manipulation
numpy           # Numerical computing
joblib          # Model serialization
googlemaps      # Google Maps API
polyline        # Route encoding/decoding
flask           # Web framework
flask-caching   # Request caching
xgboost         # Gradient boosting models
pytz            # Timezone handling
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🔑 Google Maps API Setup

**To use the Flask web app, you need a Google Maps API Key:**

1. **Go to Google Cloud Console**
   - https://console.cloud.google.com/

2. **Create a new project or select existing**

3. **Enable these APIs:**
   - Maps JavaScript API
   - Directions API
   - Geocoding API

4. **Create API Key (Credentials > API Key)**

5. **Set in your environment:**
   ```bash
   export GOOGLE_MAPS_API_KEY="your_api_key_here"
   ```

6. **Or modify in app.py line 21:**
   ```python
   API_KEY = "your_api_key_here"
   ```

---

## 💻 Usage Examples

### 1. Running the Web App
```bash
python app.py
# Visit http://localhost:5000
```

### 2. Using Jupyter Notebooks

**Explore the data:**
```bash
jupyter notebook main.ipynb
```

**Run preprocessing:**
```bash
jupyter notebook preprocessing.ipynb
```

**Taxi-specific analysis:**
```bash
jupyter notebook taxi_preprocessing.ipynb
```

### 3. Training Models (requires train.csv)

The pre-trained models are included, but to retrain:

```bash
python build_1km_speed_model.py    # Full training
# or
python train_only_fast.py          # Fast training
```

⚠️ **Note:** Model training requires `train.csv` (1.7 GB) from the Kaggle Porto Taxi dataset, which is not included due to size limits.

---

## 📊 Data & Models

### Pre-trained Models Included
- **xgb_speed_model_prod.pkl** - Production XGBoost model for speed prediction
- **xgb_grid_model_prod.pkl** - Alternative grid-based model
- **lon_bins_1km.pkl** - Longitude bin boundaries for 1km grid cells
- **lat_bins_1km.pkl** - Latitude bin boundaries for 1km grid cells

### Grid Coverage
- **City:** Porto, Portugal
- **Bounding Box:** Lon [-8.73, -8.57] | Lat [41.10, 41.25]
- **Grid Cells:** 14 × 17 = 238 cells (1km × 1km each)
- **Training Records:** 31,044 aggregated records

### Data Files (CSV)
- `df_model_data_full.csv` - Complete model features with all columns
- `df_model_data_full_01.csv` - Alternative feature set with preprocessing
- `df_speed_data_full.csv` - Aggregated speed statistics by grid cell

---

## 🎯 Model Features

The XGBoost models use **4 input features:**
1. **Hour of Day** (0-23)
2. **Day of Week** (0-6, Monday=0)
3. **Longitude Bin** (grid column index)
4. **Latitude Bin** (grid row index)

**Output:** Average traffic speed (km/h) for the specified time and location

---

## 🧪 Troubleshooting

### "ModuleNotFoundError: No module named 'xgboost'"
```bash
pip install xgboost
```

### "Google Maps API Error"
- Check your API key is set correctly
- Verify the APIs are enabled in Google Cloud Console
- Check that API key has no IP restrictions

### "CUDA device not found" (if using GPU)
```bash
# Force CPU mode in app.py (already done by default)
```

### "Port 5000 already in use"
```bash
python app.py --port 8000
```

### Models not loading
- Ensure `xgb_speed_model_prod.pkl` exists in the root directory
- Check that all `.pkl` files are in the correct location

---

## 📚 Documentation

- **PROJECT_UNDERSTANDING.md** - Complete technical overview and problem statement
- **PIPELINE_DETAILS.md** - Detailed pipeline implementation
- **RESEARCH_PAPER_CONTEXT.md** - Research background and context

---

## 🏗️ Architecture Diagram

See `Project_zip/research_figures/Architecture_diagram.png` for visual overview

---

## 🔄 Getting train.csv (For Model Retraining)

The original training dataset can be obtained from:
- **Kaggle:** Porto Taxi Trajectory Dataset
- **Link:** https://www.kaggle.com/c/ecml-14-taxi-trajectory-prediction-i

File: `train.csv` (1.7GB, 1.71M trips)

To retrain models with your own data:
```bash
# Place train.csv in the root directory
python build_1km_speed_model.py
```

---

## ✅ What You Can Do Right Now

✅ Run the Flask web app (pre-trained models included)
✅ Explore Jupyter notebooks for analysis
✅ View research figures and visualizations
✅ Understand the system architecture
✅ Make predictions for any route in Porto

❌ Retrain models (requires train.csv - 1.7 GB, not included)

---

## 📝 License

This project is part of an Intelligent Transportation Systems (ITS) academic research initiative.

---

## 👤 Author

**Divgagan**  
[GitHub Profile](https://github.com/Divgagan)

---

## 🙏 Acknowledgments

- **Dataset:** Kaggle Porto Taxi Trajectory Prediction Challenge
- **Tools:** XGBoost, Google Maps API, Flask, Pandas
- **Research Background:** See RESEARCH_PAPER_CONTEXT.md

---

## 📞 Support

For issues, questions, or improvements:
1. Check the **Troubleshooting** section
2. Review **PROJECT_UNDERSTANDING.md** for technical details
3. Open an issue on GitHub

---

**Last Updated:** June 2026
