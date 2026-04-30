"""
Triathlon Finish Time Predictor
================================
Predicts triathlon race finish times based on athlete training data
using a Random Forest regression model.

Author: Hamad Ibrahim Alhammadi
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import json
import os

# ──────────────────────────────────────────────
#  Feature Engineering
# ──────────────────────────────────────────────

def compute_training_load(weekly_swim_km, weekly_bike_km, weekly_run_km,
                           avg_swim_pace, avg_bike_speed, avg_run_pace):
    """
    Compute a composite training load score using sport-specific weighting.
    Swim contributes more per km due to its technical demand.
    """
    swim_load = weekly_swim_km * 1.8 * (1 / max(avg_swim_pace, 1))
    bike_load = weekly_bike_km * 0.5 * (avg_bike_speed / 30)
    run_load  = weekly_run_km  * 1.0 * (1 / max(avg_run_pace, 1))
    return swim_load + bike_load + run_load


def build_features(df):
    """
    Derive composite features from raw training metrics.
    """
    df = df.copy()

    # Training load score
    df['training_load'] = df.apply(
        lambda r: compute_training_load(
            r['weekly_swim_km'], r['weekly_bike_km'], r['weekly_run_km'],
            r['avg_swim_pace_min_per_100m'],
            r['avg_bike_speed_kmh'],
            r['avg_run_pace_min_per_km']
        ), axis=1
    )

    # Brick ratio: cycling-to-running ratio (key triathlon metric)
    df['brick_ratio'] = df['weekly_bike_km'] / (df['weekly_run_km'] + 1e-6)

    # Aerobic efficiency index
    df['aerobic_efficiency'] = (df['avg_bike_speed_kmh'] / 
                                 df['avg_run_pace_min_per_km'].clip(lower=0.1))

    # Weekly training volume (hours approximation)
    df['weekly_hours'] = (
        (df['weekly_swim_km'] / 3.0) +         # ~3 km/h swimming
        (df['weekly_bike_km'] / df['avg_bike_speed_kmh'].clip(lower=1)) +
        (df['weekly_run_km'] * df['avg_run_pace_min_per_km'] / 60)
    )

    return df


# ──────────────────────────────────────────────
#  Synthetic Data Generation
# ──────────────────────────────────────────────

def generate_dataset(n=600, seed=42):
    """
    Generate realistic synthetic triathlon training data.
    Finish time is a function of training metrics + noise.
    """
    rng = np.random.default_rng(seed)

    # Athlete profiles: beginner / intermediate / advanced
    levels = rng.choice(['beginner', 'intermediate', 'advanced'],
                         size=n, p=[0.35, 0.45, 0.20])

    base_params = {
        'beginner':     dict(swim=(2.0, 0.8), bike=(80, 20),  run=(20, 8),
                             swim_pace=(2.8, 0.4), bike_spd=(22, 3), run_pace=(6.5, 0.8)),
        'intermediate': dict(swim=(4.0, 1.0), bike=(150, 30), run=(35, 10),
                             swim_pace=(2.2, 0.3), bike_spd=(28, 3), run_pace=(5.5, 0.6)),
        'advanced':     dict(swim=(6.5, 1.0), bike=(220, 35), run=(55, 12),
                             swim_pace=(1.7, 0.2), bike_spd=(35, 3), run_pace=(4.5, 0.5)),
    }

    rows = []
    for lvl in levels:
        p = base_params[lvl]
        row = {
            'weekly_swim_km':            max(0.5, rng.normal(*p['swim'])),
            'weekly_bike_km':            max(20,  rng.normal(*p['bike'])),
            'weekly_run_km':             max(5,   rng.normal(*p['run'])),
            'avg_swim_pace_min_per_100m':max(1.2, rng.normal(*p['swim_pace'])),
            'avg_bike_speed_kmh':        max(15,  rng.normal(*p['bike_spd'])),
            'avg_run_pace_min_per_km':   max(3.5, rng.normal(*p['run_pace'])),
            'weeks_of_training':         int(rng.integers(8, 52)),
            'races_completed':           int(rng.integers(0, 15)),
            'age':                       int(rng.integers(16, 55)),
            'resting_hr':                int(rng.integers(42, 80)),
            'level':                     lvl,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = build_features(df)

    # Encode level
    df['level_enc'] = df['level'].map({'beginner': 0, 'intermediate': 1, 'advanced': 2})

    # Generate finish time (Olympic distance: ~2h for advanced, ~3.5h for beginner)
    base_time = {'beginner': 210, 'intermediate': 160, 'advanced': 125}
    df['finish_time_min'] = (
        df['level'].map(base_time)
        - df['training_load'] * 0.8
        - df['weekly_hours'] * 1.2
        + df['avg_swim_pace_min_per_100m'] * 8
        - df['avg_bike_speed_kmh'] * 0.9
        + df['avg_run_pace_min_per_km'] * 6
        - df['races_completed'] * 1.5
        + rng.normal(0, 6, n)
    ).clip(lower=90, upper=360)

    return df


# ──────────────────────────────────────────────
#  Model Training
# ──────────────────────────────────────────────

FEATURE_COLS = [
    'weekly_swim_km', 'weekly_bike_km', 'weekly_run_km',
    'avg_swim_pace_min_per_100m', 'avg_bike_speed_kmh', 'avg_run_pace_min_per_km',
    'weeks_of_training', 'races_completed', 'age', 'resting_hr',
    'training_load', 'brick_ratio', 'aerobic_efficiency', 'weekly_hours',
    'level_enc',
]

def train(df=None, save_path='model.pkl'):
    if df is None:
        df = generate_dataset()

    X = df[FEATURE_COLS]
    y = df['finish_time_min']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    cv  = cross_val_score(pipeline, X, y, cv=5, scoring='r2')

    metrics = {
        'mae_minutes': round(mae, 2),
        'r2_score':    round(r2, 4),
        'cv_r2_mean':  round(cv.mean(), 4),
        'cv_r2_std':   round(cv.std(), 4),
        'n_samples':   len(df),
    }

    joblib.dump(pipeline, save_path)
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Model Training Complete ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nModel saved → {save_path}")
    return pipeline, metrics


# ──────────────────────────────────────────────
#  Prediction Interface
# ──────────────────────────────────────────────

def predict(athlete_data: dict, model_path='model.pkl') -> dict:
    """
    Predict finish time for a single athlete.

    athlete_data keys:
        weekly_swim_km, weekly_bike_km, weekly_run_km,
        avg_swim_pace_min_per_100m, avg_bike_speed_kmh, avg_run_pace_min_per_km,
        weeks_of_training, races_completed, age, resting_hr, level
    """
    model = joblib.load(model_path)

    df = pd.DataFrame([athlete_data])
    df = build_features(df)
    df['level_enc'] = df['level'].map({'beginner': 0, 'intermediate': 1, 'advanced': 2})

    pred_min = model.predict(df[FEATURE_COLS])[0]

    hours   = int(pred_min // 60)
    minutes = int(pred_min % 60)

    # Breakdown estimate (rough proportions: swim 18%, bike 50%, run 32%)
    return {
        'predicted_finish_min': round(pred_min, 1),
        'predicted_finish':     f"{hours}h {minutes:02d}m",
        'split_estimate': {
            'swim':  f"{int(pred_min * 0.18)} min",
            'bike':  f"{int(pred_min * 0.50)} min",
            'run':   f"{int(pred_min * 0.32)} min",
        }
    }


if __name__ == '__main__':
    df = generate_dataset(n=600)
    pipeline, metrics = train(df)

    # Example prediction
    sample = {
        'weekly_swim_km': 4.0,
        'weekly_bike_km': 160,
        'weekly_run_km': 35,
        'avg_swim_pace_min_per_100m': 2.1,
        'avg_bike_speed_kmh': 29,
        'avg_run_pace_min_per_km': 5.3,
        'weeks_of_training': 20,
        'races_completed': 3,
        'age': 18,
        'resting_hr': 52,
        'level': 'intermediate',
    }

    result = predict(sample)
    print("\n=== Sample Prediction ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
