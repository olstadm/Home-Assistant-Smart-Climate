# ML Climate Enhancer - Enhance RC model predictions with machine learning
# Uses ensemble of models to improve temperature predictions while keeping existing RC model
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List, Optional, Dict, Any
import pickle
import os
import json
import logging

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import xgboost as xgb
    import lightgbm as lgb
    ML_AVAILABLE = True
    logger.info("ML packages loaded successfully")
except ImportError as e:
    ML_AVAILABLE = False
    logger.warning(f"ML packages not available: {e}")


class MLClimateEnhancer:
    """
    Machine Learning enhancer for the existing RC climate model.
    Keeps RC model in place but uses ML to improve predictions and validate comfort band violations.
    """
    
    def __init__(self, storage_path: str = "/data/models"):
        self.storage_path = storage_path
        self.models = {}
        self.scalers = {}
        self.feature_history = []
        self.prediction_history = []
        self.model_names = ['rf', 'xgb', 'lgb', 'gb'] if ML_AVAILABLE else []
        self.ensemble_weights = None
        self.comfort_violations = []
        
        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize models if ML is available
        if ML_AVAILABLE:
            self._initialize_models()
            self._load_models()
    
    def _initialize_models(self):
        """Initialize ML models with appropriate hyperparameters"""
        self.models = {
            'rf': RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'xgb': xgb.XGBRegressor(
                n_estimators=50,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=50,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=50,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        }
        
        # Initialize scalers for each model
        for name in self.model_names:
            self.scalers[name] = StandardScaler()
    
    def _save_models(self):
        """Save trained models and scalers to disk"""
        if not ML_AVAILABLE:
            return
            
        try:
            for name in self.model_names:
                if hasattr(self.models[name], 'feature_importances_'):
                    # Save model
                    model_path = os.path.join(self.storage_path, f'{name}_model.pkl')
                    with open(model_path, 'wb') as f:
                        pickle.dump(self.models[name], f)
                    
                    # Save scaler
                    scaler_path = os.path.join(self.storage_path, f'{name}_scaler.pkl')
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(self.scalers[name], f)
            
            # Save ensemble weights if available
            if self.ensemble_weights is not None:
                weights_path = os.path.join(self.storage_path, 'ensemble_weights.json')
                with open(weights_path, 'w') as f:
                    json.dump(self.ensemble_weights.tolist(), f)
            
            logger.info("ML models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_models(self):
        """Load trained models and scalers from disk"""
        if not ML_AVAILABLE:
            return
            
        try:
            for name in self.model_names:
                model_path = os.path.join(self.storage_path, f'{name}_model.pkl')
                scaler_path = os.path.join(self.storage_path, f'{name}_scaler.pkl')
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    with open(model_path, 'rb') as f:
                        self.models[name] = pickle.load(f)
                    with open(scaler_path, 'rb') as f:
                        self.scalers[name] = pickle.load(f)
            
            # Load ensemble weights if available
            weights_path = os.path.join(self.storage_path, 'ensemble_weights.json')
            if os.path.exists(weights_path):
                with open(weights_path, 'r') as f:
                    weights = json.load(f)
                    self.ensemble_weights = np.array(weights)
            
            logger.info("ML models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def create_features(self, current_temp: float, outdoor_temp: float, 
                       humidity_in: float, humidity_out: float, 
                       solar: float, hvac_action: str, 
                       time_features: Dict[str, float]) -> np.ndarray:
        """Create feature vector for ML models"""
        
        # Temperature features
        temp_diff = outdoor_temp - current_temp
        temp_ratio = current_temp / max(outdoor_temp, 1.0)
        
        # Humidity features
        humidity_diff = humidity_out - humidity_in
        humidity_ratio = humidity_in / max(humidity_out, 1.0)
        
        # HVAC action encoding
        hvac_heating = 1.0 if hvac_action == 'heating' else 0.0
        hvac_cooling = 1.0 if hvac_action == 'cooling' else 0.0
        hvac_idle = 1.0 if hvac_action == 'idle' else 0.0
        
        # Combine all features
        features = np.array([
            current_temp,
            outdoor_temp,
            temp_diff,
            temp_ratio,
            humidity_in,
            humidity_out,
            humidity_diff,
            humidity_ratio,
            solar,
            hvac_heating,
            hvac_cooling,
            hvac_idle,
            time_features.get('hour_sin', 0.0),
            time_features.get('hour_cos', 0.0),
            time_features.get('day_of_year_sin', 0.0)
        ])
        
        return features.reshape(1, -1)
    
    def add_sample(self, features: np.ndarray, actual_temp_change: float, 
                  rc_prediction: float, timestamp: datetime):
        """Add a new training sample to the dataset"""
        if not ML_AVAILABLE:
            return
            
        # Store feature and prediction history
        self.feature_history.append(features.flatten())
        self.prediction_history.append({
            'features': features.flatten(),
            'actual_change': actual_temp_change,
            'rc_prediction': rc_prediction,
            'timestamp': timestamp,
            'ml_error': None  # Will be filled after ML prediction
        })
        
        # Limit history to last 1000 samples to prevent memory issues
        if len(self.feature_history) > 1000:
            self.feature_history = self.feature_history[-1000:]
            self.prediction_history = self.prediction_history[-1000:]
    
    def train_models(self, min_samples: int = 50) -> Dict[str, float]:
        """Train ML models on collected data"""
        if not ML_AVAILABLE or len(self.prediction_history) < min_samples:
            logger.info(f"Not enough samples for training: {len(self.prediction_history)}")
            return {}
        
        try:
            # Prepare training data
            X = np.array([sample['features'] for sample in self.prediction_history])
            y = np.array([sample['actual_change'] for sample in self.prediction_history])
            
            # Train each model
            scores = {}
            predictions = {}
            
            for name in self.model_names:
                try:
                    # Scale features
                    X_scaled = self.scalers[name].fit_transform(X)
                    
                    # Train model
                    self.models[name].fit(X_scaled, y)
                    
                    # Get predictions for ensemble weight calculation
                    y_pred = self.models[name].predict(X_scaled)
                    predictions[name] = y_pred
                    
                    # Calculate score
                    mse = mean_squared_error(y, y_pred)
                    scores[name] = 1.0 / (1.0 + mse)  # Higher is better
                    
                    logger.info(f"Model {name} trained with MSE: {mse:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training model {name}: {e}")
                    scores[name] = 0.0
            
            # Calculate ensemble weights based on performance
            if scores:
                total_score = sum(scores.values())
                if total_score > 0:
                    self.ensemble_weights = np.array([
                        scores.get(name, 0.0) / total_score 
                        for name in self.model_names
                    ])
                else:
                    self.ensemble_weights = np.ones(len(self.model_names)) / len(self.model_names)
            
            # Save models
            self._save_models()
            
            logger.info(f"ML models trained successfully with {len(self.prediction_history)} samples")
            return scores
            
        except Exception as e:
            logger.error(f"Error training ML models: {e}")
            return {}
    
    def predict_enhancement(self, features: np.ndarray, rc_prediction: float) -> Tuple[float, Dict[str, Any]]:
        """Predict temperature change enhancement using trained ML models"""
        if not ML_AVAILABLE:
            return rc_prediction, {'method': 'rc_only', 'confidence': 0.5}
        
        # Check if models are trained
        trained_models = [name for name in self.model_names 
                         if hasattr(self.models[name], 'feature_importances_')]
        
        if not trained_models or self.ensemble_weights is None:
            return rc_prediction, {'method': 'rc_only', 'confidence': 0.5}
        
        try:
            ml_predictions = []
            
            for i, name in enumerate(self.model_names):
                if name in trained_models:
                    # Scale features
                    X_scaled = self.scalers[name].transform(features)
                    
                    # Get prediction
                    ml_pred = self.models[name].predict(X_scaled)[0]
                    ml_predictions.append(ml_pred)
                else:
                    ml_predictions.append(rc_prediction)
            
            # Ensemble prediction
            ml_predictions = np.array(ml_predictions)
            ensemble_prediction = np.dot(ml_predictions, self.ensemble_weights)
            
            # Blend with RC prediction (70% RC, 30% ML for stability)
            blend_ratio = 0.3
            enhanced_prediction = (1 - blend_ratio) * rc_prediction + blend_ratio * ensemble_prediction
            
            # Calculate confidence based on model agreement
            if len(ml_predictions) > 1:
                std_dev = np.std(ml_predictions)
                confidence = max(0.1, min(0.9, 1.0 / (1.0 + std_dev)))
            else:
                confidence = 0.7
            
            metadata = {
                'method': 'ml_enhanced',
                'confidence': confidence,
                'rc_prediction': rc_prediction,
                'ml_prediction': ensemble_prediction,
                'blend_ratio': blend_ratio,
                'models_used': trained_models
            }
            
            return enhanced_prediction, metadata
            
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return rc_prediction, {'method': 'rc_fallback', 'confidence': 0.5}
    
    def validate_comfort_band_violation(self, current_temp: float, predicted_temp: float,
                                      comfort_min: float, comfort_max: float,
                                      hvac_action: str) -> Dict[str, Any]:
        """Validate predictions for comfort band violations"""
        violation = {
            'violation_detected': False,
            'violation_type': None,
            'confidence': 0.0,
            'recommendation': None,
            'severity': 'none'
        }
        
        # Check for overheating (temperature rising, no cooling active)
        if predicted_temp > comfort_max and hvac_action != 'cooling':
            violation['violation_detected'] = True
            violation['violation_type'] = 'overheating'
            violation['recommendation'] = 'start_cooling'
            
            # Calculate severity based on how far above comfort max
            excess = predicted_temp - comfort_max
            if excess > 5.0:
                violation['severity'] = 'high'
            elif excess > 2.0:
                violation['severity'] = 'medium'
            else:
                violation['severity'] = 'low'
            
            # Confidence based on current temperature proximity to limit
            temp_proximity = (comfort_max - current_temp) / max(comfort_max - comfort_min, 1.0)
            violation['confidence'] = max(0.1, min(0.9, 1.0 - temp_proximity))
        
        # Check for overcooling (temperature falling, no heating active)
        elif predicted_temp < comfort_min and hvac_action != 'heating':
            violation['violation_detected'] = True
            violation['violation_type'] = 'overcooling'
            violation['recommendation'] = 'start_heating'
            
            # Calculate severity based on how far below comfort min
            deficit = comfort_min - predicted_temp
            if deficit > 5.0:
                violation['severity'] = 'high'
            elif deficit > 2.0:
                violation['severity'] = 'medium'
            else:
                violation['severity'] = 'low'
            
            # Confidence based on current temperature proximity to limit
            temp_proximity = (current_temp - comfort_min) / max(comfort_max - comfort_min, 1.0)
            violation['confidence'] = max(0.1, min(0.9, 1.0 - temp_proximity))
        
        return violation
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of ML models"""
        status = {
            'ml_available': ML_AVAILABLE,
            'models_trained': 0,
            'training_samples': len(self.prediction_history),
            'model_scores': {},
            'ensemble_weights': self.ensemble_weights.tolist() if self.ensemble_weights is not None else None
        }
        
        if ML_AVAILABLE:
            for name in self.model_names:
                if hasattr(self.models[name], 'feature_importances_'):
                    status['models_trained'] += 1
        
        return status