# ML Climate Enhancer - Enhance RC model predictions with machine learning
# Uses ensemble of models to improve temperature predictions while keeping existing RC model
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List, Optional, Dict, Any
import pickle
import os
import json

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import xgboost as xgb
    import lightgbm as lgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class MLClimateEnhancer:
    """
    Machine Learning enhancer for the existing RC climate model.
    Keeps RC model in place but uses ML to improve predictions and validate comfort band violations.
    """
    
    def __init__(self, storage_path: str = "/tmp/ml_climate_models"):
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
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'xgb': xgb.XGBRegressor(
                n_estimators=50,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=50,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=-1
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
            
        for name in self.model_names:
            if hasattr(self.models[name], 'feature_importances_'):
                model_path = os.path.join(self.storage_path, f'{name}_model.pkl')
                scaler_path = os.path.join(self.storage_path, f'{name}_scaler.pkl')
                
                with open(model_path, 'wb') as f:
                    pickle.dump(self.models[name], f)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scalers[name], f)
        
        # Save ensemble weights
        if self.ensemble_weights is not None:
            weights_path = os.path.join(self.storage_path, 'ensemble_weights.json')
            with open(weights_path, 'w') as f:
                json.dump(self.ensemble_weights.tolist(), f)
    
    def _load_models(self):
        """Load trained models and scalers from disk"""
        if not ML_AVAILABLE:
            return
            
        for name in self.model_names:
            model_path = os.path.join(self.storage_path, f'{name}_model.pkl')
            scaler_path = os.path.join(self.storage_path, f'{name}_scaler.pkl')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    with open(model_path, 'rb') as f:
                        self.models[name] = pickle.load(f)
                    with open(scaler_path, 'rb') as f:
                        self.scalers[name] = pickle.load(f)
                except Exception:
                    # If loading fails, reinitialize
                    self._initialize_models()
        
        # Load ensemble weights
        weights_path = os.path.join(self.storage_path, 'ensemble_weights.json')
        if os.path.exists(weights_path):
            try:
                with open(weights_path, 'r') as f:
                    self.ensemble_weights = np.array(json.load(f))
            except Exception:
                pass
    
    def create_features(self, current_temp: float, outdoor_temp: float, 
                       humidity_in: float, humidity_out: float, 
                       solar: float, hvac_action: str, 
                       time_features: Dict[str, float]) -> np.ndarray:
        """Create feature vector for ML models"""
        features = [
            current_temp,
            outdoor_temp,
            outdoor_temp - current_temp,  # Temperature difference
            humidity_in or 50.0,
            humidity_out or 50.0,
            solar,
            1.0 if hvac_action == "heating" else 0.0,
            1.0 if hvac_action == "cooling" else 0.0,
            1.0 if hvac_action == "idle" else 0.0,
            time_features.get('hour', 12.0) / 24.0,  # Hour of day normalized
            time_features.get('day_of_week', 3.0) / 7.0,  # Day of week normalized
            time_features.get('month', 6.0) / 12.0,  # Month normalized
        ]
        
        # Add moving averages if we have history
        if len(self.feature_history) > 0:
            recent_temps = [f[0] for f in self.feature_history[-10:]]  # Last 10 temps
            features.extend([
                np.mean(recent_temps) if recent_temps else current_temp,
                np.std(recent_temps) if len(recent_temps) > 1 else 0.0,
                np.max(recent_temps) - np.min(recent_temps) if recent_temps else 0.0
            ])
        else:
            features.extend([current_temp, 0.0, 0.0])
        
        return np.array(features).reshape(1, -1)
    
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
            return {}
        
        # Prepare training data
        X = np.array([p['features'] for p in self.prediction_history])
        y = np.array([p['actual_change'] for p in self.prediction_history])
        
        # Split into train/test (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model_scores = {}
        model_predictions = {}
        
        # Train each model
        for name in self.model_names:
            try:
                # Scale features
                X_train_scaled = self.scalers[name].fit_transform(X_train)
                X_test_scaled = self.scalers[name].transform(X_test)
                
                # Train model
                self.models[name].fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = self.models[name].predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                model_scores[name] = {'mse': mse, 'mae': mae}
                model_predictions[name] = y_pred
                
            except Exception as e:
                print(f"Error training model {name}: {e}")
                continue
        
        # Calculate ensemble weights based on inverse MSE
        if model_scores and len(model_scores) > 1:
            mse_values = [model_scores[name]['mse'] for name in self.model_names if name in model_scores]
            if all(mse > 0 for mse in mse_values):
                inv_mse = [1.0 / mse for mse in mse_values]
                total = sum(inv_mse)
                self.ensemble_weights = np.array([w / total for w in inv_mse])
            else:
                # Equal weights if MSE is zero or negative
                self.ensemble_weights = np.ones(len(model_scores)) / len(model_scores)
        
        # Save models
        self._save_models()
        
        return model_scores
    
    def predict_enhancement(self, features: np.ndarray, rc_prediction: float) -> Tuple[float, Dict[str, Any]]:
        """
        Enhance RC model prediction with ML ensemble
        Returns: (enhanced_prediction, metadata)
        """
        if not ML_AVAILABLE or not self.models:
            return rc_prediction, {'method': 'rc_only', 'confidence': 0.5}
        
        try:
            predictions = {}
            valid_models = []
            
            # Get prediction from each trained model
            for name in self.model_names:
                if hasattr(self.models[name], 'predict'):
                    try:
                        X_scaled = self.scalers[name].transform(features)
                        pred = self.models[name].predict(X_scaled)[0]
                        predictions[name] = pred
                        valid_models.append(name)
                    except Exception:
                        continue
            
            if not predictions:
                return rc_prediction, {'method': 'rc_only', 'confidence': 0.5}
            
            # Ensemble prediction using weights or simple average
            if self.ensemble_weights is not None and len(valid_models) == len(self.ensemble_weights):
                ml_prediction = sum(predictions[name] * weight 
                                  for name, weight in zip(valid_models, self.ensemble_weights))
            else:
                ml_prediction = np.mean(list(predictions.values()))
            
            # Blend RC and ML predictions (70% RC, 30% ML for stability)
            alpha = 0.7
            enhanced_prediction = alpha * rc_prediction + (1 - alpha) * ml_prediction
            
            # Calculate confidence based on prediction agreement
            pred_std = np.std(list(predictions.values())) if len(predictions) > 1 else 0.1
            confidence = max(0.1, min(0.9, 1.0 / (1.0 + pred_std)))
            
            metadata = {
                'method': 'ml_enhanced',
                'rc_prediction': rc_prediction,
                'ml_prediction': ml_prediction,
                'individual_predictions': predictions,
                'confidence': confidence,
                'prediction_std': pred_std
            }
            
            return enhanced_prediction, metadata
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            return rc_prediction, {'method': 'rc_fallback', 'confidence': 0.5, 'error': str(e)}
    
    def validate_comfort_band_violation(self, current_temp: float, predicted_temp: float,
                                      comfort_min: float, comfort_max: float,
                                      hvac_action: str) -> Dict[str, Any]:
        """
        Validate comfort band violations with data verification
        Returns validation result with confidence level
        """
        violation_info = {
            'violation_detected': False,
            'violation_type': None,
            'confidence': 0.0,
            'validation_method': 'data_verified',
            'recommendation': 'none'
        }
        
        # Check if prediction goes outside comfort band
        if predicted_temp > comfort_max:
            violation_info['violation_detected'] = True
            violation_info['violation_type'] = 'overheating'
        elif predicted_temp < comfort_min:
            violation_info['violation_detected'] = True
            violation_info['violation_type'] = 'overcooling'
        
        if not violation_info['violation_detected']:
            return violation_info
        
        # Data verification: check if current conditions support the violation
        temp_trend = predicted_temp - current_temp
        violation_supported = False
        
        if violation_info['violation_type'] == 'overheating':
            # Verify: temperature rising and no cooling active
            if temp_trend > 0 and hvac_action != 'cooling':
                violation_supported = True
                if current_temp > comfort_max - 1.0:  # Already close to limit
                    violation_info['confidence'] = 0.9
                else:
                    violation_info['confidence'] = 0.7
                violation_info['recommendation'] = 'start_cooling'
        
        elif violation_info['violation_type'] == 'overcooling':
            # Verify: temperature falling and no heating active
            if temp_trend < 0 and hvac_action != 'heating':
                violation_supported = True
                if current_temp < comfort_min + 1.0:  # Already close to limit
                    violation_info['confidence'] = 0.9
                else:
                    violation_info['confidence'] = 0.7
                violation_info['recommendation'] = 'start_heating'
        
        if not violation_supported:
            violation_info['violation_detected'] = False
            violation_info['confidence'] = 0.1
            violation_info['recommendation'] = 'monitor'
        
        # Store violation for historical analysis
        self.comfort_violations.append({
            'timestamp': datetime.now(),
            'current_temp': current_temp,
            'predicted_temp': predicted_temp,
            'violation_info': violation_info.copy()
        })
        
        # Keep only last 100 violations
        if len(self.comfort_violations) > 100:
            self.comfort_violations = self.comfort_violations[-100:]
        
        return violation_info
    
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