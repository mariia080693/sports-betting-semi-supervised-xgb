# Sports Betting User Survey Prediction

## Overview
Semi-supervised XGBoost model to predict user survey responses (Chocolate/Strawberry/Vanilla) from sports betting user behavior data.

## Files
```
├── data_curation.py          # Data preprocessing
├── model.py                  # XGBoost training with self-training
├── data.csv                  # Processed dataset
├── requirements.txt          # Python dependencies
├── trained_model.json        # Saved XGBoost model (generated)
├── label_encoder.pkl         # Saved label encoder (generated)
└── Modeling Task_v3b.xlsx    # Original data
```

## Process Flow
```
Raw Data (Excel) 
    ↓
Data Curation and  Feature Engineering
    ↓
Labeled/Unlabeled Data Split
    ↓
Train/Validation/Test Split (70/10/20)
    ↓
Self-Training Loop (up to 20 iterations):
    ├── XGBoost Training with Class Weights
    ├── Validation Performance Monitoring  
    ├── Pseudo-Label Generation (Confidence Thresholds)
    ├── High-Confidence Sample Selection
    └── Training Set Augmentation
    ↓
Final Model Evaluation on Test Set

```

## Key Features
- **Semi-supervised learning**: Uses unlabeled data via confidence-based pseudo-labeling
- **Class imbalance handling**: Balanced weights + class-specific confidence thresholds
- **Feature engineering**: Date-based features, categorical standardization

## Data
**Features**: Age, State, RegistrationDevice, FirstBetDevice, AcquisitionSource, MainBetSport, FirstWeekTurnover, DaysSinceReg, DaysToFirstBet

**Target**: SurveyAnswer (3 classes: Chocolate, Strawberry, Vanilla)

## Model Configuration
- **Algorithm**: XGBoost with categorical support
- **Confidence thresholds**: {0: 0.8, 1: 0.6, 2: 0.6}
- **Class weights**: Balanced (recalculated each iteration)
- **Early stopping**: 20 rounds on validation loss

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run `data_curation.py` to preprocess data
3. Run `model.py` to train and evaluate model
   - Automatically saves `trained_model.json` (XGBoost model)
   - Automatically saves `label_encoder.pkl` (for class name conversion)

## Model Loading (for future predictions)
```python
import xgboost as xgb
import pickle

# Load model and encoder
model = xgb.XGBClassifier()
model.load_model("trained_model.json")

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Make predictions
predictions = model.predict(new_data)
class_names = label_encoder.inverse_transform(predictions)
```

## Dependencies
See `requirements.txt` for package versions