# Machine Learning Integration Documentation

## Overview

This implementation enhances the existing RC (Resistor-Capacitor) thermal model with machine learning capabilities while keeping the original model in place. The system now provides improved temperature predictions, comfort band violation detection, and enhanced runtime estimates.

## Key Features

### 1. ML Enhancement Layer (`ml_climate_enhancer.py`)

- **Ensemble Learning**: Uses RandomForest, XGBoost, LightGBM, and GradientBoosting models
- **Stable Blending**: 70% RC model + 30% ML prediction for reliability
- **Auto-Training**: Models train automatically every hour with 50+ samples
- **Feature Engineering**: 15 features including temperature, humidity, solar, HVAC state, and time patterns
- **Confidence Scoring**: Provides prediction confidence based on model agreement

### 2. Enhanced Predictions

- **ML-Enhanced Temperature Forecasts**: Combines RC physics with ML pattern recognition
- **Comfort Band Violations**: Data-verified predictions outside comfort zone
- **Runtime Estimates**: Start/end times with estimated HVAC runtime in minutes and hours
- **Confidence Levels**: Prediction reliability scoring

### 3. Dashboard Improvements

- **13-Hour Timeline**: Extended view with "now" positioned 1 hour from left
- **ML Prediction Overlay**: Orange line showing ML-enhanced predictions
- **AccuWeather Integration**: Green dashed line showing weather service forecasts
- **Runtime Information**: Detailed start/end times and duration estimates
- **ML Status Monitoring**: Training status and model performance metrics

### 4. Data Verification System

- **Temperature Trend Analysis**: Validates predictions against actual temperature changes
- **HVAC State Consideration**: Accounts for current heating/cooling status
- **Violation Confidence**: Only high-confidence violations trigger recommendations
- **Historical Tracking**: Maintains violation history for pattern analysis

## Installation

1. Install required packages:
```bash
pip install scikit-learn numpy pandas xgboost lightgbm
```

2. Place files in AppDaemon apps directory:
- `home_forecast.py` (enhanced)
- `ml_climate_enhancer.py` (new)

3. Update dashboard with new YAML configuration

## Configuration

The ML system automatically:
- Creates training samples from sensor data
- Trains models every hour with sufficient data
- Enhances RC predictions with ML insights
- Validates comfort band violations
- Provides runtime estimates

No additional configuration required beyond existing RC model setup.

## New Sensors

### ML-Specific Sensors
- `sensor.home_model_ml_status`: Training status and model metrics
- `sensor.home_model_ml_enhanced_prediction`: ML-enhanced temperature prediction
- `sensor.home_model_comfort_violation`: Comfort band violation alerts

### Enhanced Existing Sensors
- Start/end time sensors now include runtime estimates
- All sensors include confidence and method metadata

## Performance

### Model Training
- **Training Frequency**: Every hour (configurable)
- **Minimum Samples**: 50 samples required for training
- **Model Storage**: Persistent model storage in `/tmp/ml_climate_models`
- **Memory Usage**: Limited to 1000 historical samples

### Prediction Enhancement
- **Blend Ratio**: 70% RC + 30% ML for stability
- **Feature Count**: 15 engineered features per prediction
- **Response Time**: < 100ms for predictions
- **Confidence Range**: 0.1-0.9 based on model agreement

## Comfort Band Validation

The system validates predictions outside the comfort band:

1. **Overheating Detection**: Temperature rising with no cooling active
2. **Overcooling Detection**: Temperature falling with no heating active
3. **Confidence Scoring**: Based on current temperature proximity to limits
4. **Recommendations**: Start cooling/heating based on validated violations

## Dashboard Layout

The enhanced dashboard provides:
- **Timeline**: 13-hour view with "now" at 1-hour mark from left
- **Multiple Forecasts**: RC, ML-enhanced, and AccuWeather overlays
- **Runtime Details**: Start times, end times, and duration estimates
- **Status Monitoring**: ML training status and performance metrics
- **Comfort Tracking**: Real-time comfort and efficiency scores

## Troubleshooting

### ML Models Not Training
- Check that at least 50 samples are collected
- Verify ML package installation
- Check logs for training errors

### Predictions Not Enhanced
- Ensure ML models are trained (check `sensor.home_model_ml_status`)
- Verify feature data availability
- Check for import errors in logs

### Dashboard Issues
- Ensure ApexCharts custom card is installed
- Verify sensor entities are available
- Check for YAML syntax errors

## Future Enhancements

- Seasonal pattern learning
- Weather forecast integration
- Energy cost optimization
- Occupancy-based predictions
- Advanced comfort metrics