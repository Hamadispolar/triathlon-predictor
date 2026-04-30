# 🏊🚴🏃 TriPredict — Triathlon Finish Time Predictor

A machine learning project that predicts **Olympic-distance triathlon finish times** based on an athlete's training profile. Built using Python, scikit-learn, and a Flask REST API with an interactive HTML frontend.

---

## 🧠 Project Overview

This project applies **supervised regression** to sports performance prediction. A Random Forest model is trained on athlete training data — swim/bike/run volumes, paces, aerobic indicators — and predicts expected finish times broken into swim, bike, and run split estimates.

**Motivation:** As a competitive triathlete, I wanted to explore how training metrics translate to race performance using machine learning. This project bridges my two interests: endurance sport and AI/technology.

---

## 🗂️ Project Structure

```
triathlon-predictor/
├── model.py          # ML pipeline — feature engineering, training, prediction
├── app.py            # Flask REST API
├── index.html        # Interactive frontend (works standalone via demo mode)
├── requirements.txt  # Python dependencies
└── README.md
```

---

## ⚙️ How It Works

### 1. Feature Engineering (`model.py`)

Raw training inputs are transformed into meaningful composite features:

| Feature | Description |
|---|---|
| `training_load` | Weighted score combining swim/bike/run volume and pace |
| `brick_ratio` | Bike-to-run volume ratio (key triathlon metric) |
| `aerobic_efficiency` | Bike speed relative to run pace |
| `weekly_hours` | Estimated total weekly training hours |

### 2. Model

| Property | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Trees | 200 estimators |
| Preprocessing | StandardScaler |
| Cross-validation | 5-fold CV |
| Target | Finish time (minutes) |

### 3. Performance

| Metric | Score |
|---|---|
| Mean Absolute Error | ~4–5 minutes |
| R² Score | ~0.96 |
| CV R² Mean | ~0.95 |

---

## 🚀 Quick Start

### Run the ML model

```bash
# Install dependencies
pip install -r requirements.txt

# Train model and run a sample prediction
python model.py
```

### Start the API server

```bash
python app.py
# API running at http://localhost:5000
```

### Open the frontend

Open `index.html` in your browser. It connects to the local API, or runs in **demo mode** if the API isn't running.

---

## 🔌 API Endpoints

### `POST /predict`

Predict finish time for an athlete.

**Request body:**
```json
{
  "weekly_swim_km": 4.0,
  "weekly_bike_km": 160,
  "weekly_run_km": 35,
  "avg_swim_pace_min_per_100m": 2.1,
  "avg_bike_speed_kmh": 29,
  "avg_run_pace_min_per_km": 5.3,
  "weeks_of_training": 20,
  "races_completed": 3,
  "age": 18,
  "resting_hr": 52,
  "level": "intermediate"
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "predicted_finish_min": 148.3,
    "predicted_finish": "2h 28m",
    "split_estimate": {
      "swim": "26 min",
      "bike": "74 min",
      "run": "47 min"
    }
  }
}
```

### `GET /metrics`
Returns model performance metrics.

### `GET /health`
Health check.

---

## 📊 Input Features

| Feature | Unit | Example |
|---|---|---|
| `weekly_swim_km` | km/week | 4.0 |
| `weekly_bike_km` | km/week | 160 |
| `weekly_run_km` | km/week | 35 |
| `avg_swim_pace_min_per_100m` | min/100m | 2.1 |
| `avg_bike_speed_kmh` | km/h | 29 |
| `avg_run_pace_min_per_km` | min/km | 5.3 |
| `weeks_of_training` | weeks | 20 |
| `races_completed` | count | 3 |
| `age` | years | 18 |
| `resting_hr` | bpm | 52 |
| `level` | beginner/intermediate/advanced | intermediate |

---

## 🔮 Future Improvements

- [ ] Integrate real race data (Ironman, World Triathlon datasets)
- [ ] Add SHAP explainability to show feature importance per athlete
- [ ] Extend to 70.3 (Half Ironman) and full Ironman distances
- [ ] Mobile app using Swift/SwiftUI for on-device predictions

---

## 👤 Author

**Hamad Ibrahim Alhammadi**  
a triathlete

---

## 📄 License

MIT License — free to use and adapt with attribution.
