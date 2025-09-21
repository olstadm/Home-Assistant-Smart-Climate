"""Smart Climate Forecasting Service - RC Model with ML Enhancement."""
import asyncio
import json
import logging
import math
import os
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union

from .ha_client import HomeAssistantClient
from .ml_climate_enhancer import MLClimateEnhancer

logger = logging.getLogger(__name__)


def clip(x, lo, hi):
    """Clip value between bounds."""
    return lo if x < lo else hi if x > hi else x


# --- Psychrometrics (safe) ---
ATM_PA = 101325.0


def _f_to_c(f):
    """Convert Fahrenheit to Celsius."""
    try:
        f = float(f)
    except (ValueError, TypeError):
        f = 68.0  # 20C
    return (f - 32.0) * (5.0 / 9.0)


def _psat_pa_tc(tc):
    """Calculate saturation vapor pressure."""
    tc = max(-50.0, min(60.0, float(tc)))
    return 610.94 * math.exp(17.625 * tc / (tc + 243.04))


def _enthalpy_kj_per_kg_dryair(temp_f, rh):
    """Calculate enthalpy from temperature and humidity."""
    tc = _f_to_c(temp_f)
    tc = max(-50.0, min(60.0, tc))
    try:
        rhf = float(rh)
    except (ValueError, TypeError):
        rhf = 50.0
    rhf = max(1.0, min(99.0, rhf)) / 100.0

    ps = _psat_pa_tc(tc)
    pv = rhf * ps
    denom = max(100.0, (ATM_PA - pv))
    w = 0.62198 * pv / denom
    return 1.006 * tc + w * (2501.0 + 1.86 * tc)


class ClimateForecaster:
    """Smart Climate Forecasting Service."""
    
    # Parameter Limits
    MIN_TAU_H, MAX_TAU_H = 0.5, 72.0
    MIN_KH, MAX_KH = 0.0, 2.0
    MIN_KC, MAX_KC = -2.0, 0.0
    MIN_B, MAX_B = -0.2, 0.2
    MIN_KE, MAX_KE = -0.02, 0.02
    MIN_KS, MAX_KS = -0.002, 0.002
    
    def __init__(self, ha_client: HomeAssistantClient, config: Dict[str, Any]):
        """Initialize the climate forecaster.
        
        Args:
            ha_client: Home Assistant client
            config: Configuration dictionary
        """
        self.ha_client = ha_client
        self.config = config
        
        # Initialize ML enhancer
        models_dir = os.environ.get('MODELS_DIR', '/data/models')
        self.ml_enhancer = MLClimateEnhancer(storage_path=models_dir)
        
        # RC Model parameters
        self.theta = [0.001, 0.5, -0.5, 0.0, 0.0, 0.0]  # [a, kH, kC, b, kE, kS]
        self.samples = 0
        
        # Historical data for learning
        self._past_preds = []
        self._aw_cache = []
        self._aw_cache_ts = None
        
        # Sensor entity IDs based on config
        self.indoor_temp_sensor = config.get('indoor_temperature_sensor', '')
        self.outdoor_temp_sensor = config.get('outdoor_temperature_sensor', '')
        self.climate_entity = config.get('climate_entity', '')
        self.indoor_humidity_sensor = config.get('indoor_humidity_sensor', '')
        self.outdoor_humidity_sensor = config.get('outdoor_humidity_sensor', '')
        
        # Configuration parameters
        self.learning_enabled = config.get('learning_enabled', True)
        self.forecast_hours = config.get('forecast_hours', 12)
        self.comfort_max_temp = config.get('comfort_max_temp', 80.0)
        self.comfort_min_temp = config.get('comfort_min_temp', 62.0)
        self.update_interval = config.get('update_interval_minutes', 5)
        
        # AccuWeather configuration
        self.accuweather_api_key = config.get('accuweather_api_key', '')
        self.accuweather_location_key = config.get('accuweather_location_key', '')
        
        # ML configuration
        self.ml_training_enabled = config.get('ml_training_enabled', True)
        self.ml_training_interval = config.get('ml_training_interval_hours', 1)
        
        # Advanced RC model parameters from configuration
        self.tau_hours = config.get('tau_hours', 2.0)
        self.forgetting_factor = config.get('forgetting_factor', 0.99)
        self.bias = config.get('bias', 0.0)
        self.comfort_cap = config.get('comfort_cap', self.comfort_max_temp)
        self.heat_min_f = config.get('heat_min_f', self.comfort_min_temp)
        self.k_heat = config.get('k_heat', 0.5)
        self.k_cool = config.get('k_cool', -0.5)
        self.recommendation_cooldown = config.get('recommendation_cooldown', 30.0)
        
        # Update RC model parameters with configured values
        self.theta[1] = self.k_heat  # Heat gain coefficient
        self.theta[2] = self.k_cool  # Cool gain coefficient
        self.theta[3] = self.bias    # Temperature bias
        
        logger.info("Climate forecaster initialized")
        logger.info(f"Indoor sensor: {self.indoor_temp_sensor}")
        logger.info(f"Outdoor sensor: {self.outdoor_temp_sensor}")
        logger.info(f"Climate entity: {self.climate_entity}")
        logger.info(f"ML training enabled: {self.ml_training_enabled}")
        logger.info(f"Advanced parameters: tau_hours={self.tau_hours:.1f}, forgetting_factor={self.forgetting_factor:.3f}")
        logger.info(f"RC model coefficients: k_heat={self.k_heat:.2f}, k_cool={self.k_cool:.2f}, bias={self.bias:.3f}")
    
    async def run_forecast_loop(self):
        """Main forecasting loop."""
        logger.info("Starting forecast loop")
        
        # Schedule tasks
        ml_training_task = None
        if self.ml_training_enabled:
            ml_training_task = asyncio.create_task(self._ml_training_loop())
        
        accuweather_task = asyncio.create_task(self._accuweather_loop())
        forecast_task = asyncio.create_task(self._forecast_loop())
        
        try:
            # Run all tasks concurrently
            tasks = [forecast_task, accuweather_task]
            if ml_training_task:
                tasks.append(ml_training_task)
            
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in forecast loop: {e}")
            raise
    
    async def _forecast_loop(self):
        """Main forecasting update loop."""
        while True:
            try:
                await self._update_forecast()
                await asyncio.sleep(self.update_interval * 60)  # Convert minutes to seconds
            except Exception as e:
                logger.error(f"Error in forecast update: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _ml_training_loop(self):
        """ML model training loop."""
        while True:
            try:
                await self._train_ml_models()
                await asyncio.sleep(self.ml_training_interval * 3600)  # Convert hours to seconds
            except Exception as e:
                logger.error(f"Error in ML training: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def _accuweather_loop(self):
        """AccuWeather data refresh loop."""
        while True:
            try:
                if self.accuweather_api_key and self.accuweather_location_key:
                    await self._refresh_accuweather_forecast()
                await asyncio.sleep(3600)  # Update hourly
            except Exception as e:
                logger.error(f"Error in AccuWeather update: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def _update_forecast(self):
        """Update climate forecast."""
        try:
            # Get current sensor values
            indoor_temp = await self._get_sensor_value(self.indoor_temp_sensor, 70.0)
            outdoor_temp = await self._get_sensor_value(self.outdoor_temp_sensor, 70.0)
            
            if not indoor_temp or not outdoor_temp:
                logger.warning("Missing temperature sensor data")
                return
            
            # Get humidity values (optional)
            indoor_humidity = 50.0
            outdoor_humidity = 50.0
            if self.indoor_humidity_sensor:
                indoor_humidity = await self._get_sensor_value(self.indoor_humidity_sensor, 50.0)
            if self.outdoor_humidity_sensor:
                outdoor_humidity = await self._get_sensor_value(self.outdoor_humidity_sensor, 50.0)
            
            # Get HVAC action
            hvac_action = await self._get_hvac_action()
            
            # Get solar irradiance (mock value for now)
            solar_irradiance = 0.0  # TODO: Add solar sensor support
            
            # Check for learning update
            if self.learning_enabled and len(self._past_preds) > 0:
                await self._update_rc_model(indoor_temp)
            
            # Generate forecast
            forecast_data = await self._generate_forecast(
                indoor_temp, outdoor_temp, indoor_humidity, outdoor_humidity, 
                solar_irradiance, hvac_action
            )
            
            # Publish forecast to Home Assistant
            await self._publish_forecast(forecast_data)
            
        except Exception as e:
            logger.error(f"Error updating forecast: {e}")
    
    async def _get_sensor_value(self, entity_id: str, default: float) -> float:
        """Get sensor value from Home Assistant."""
        if not entity_id:
            return default
            
        state_data = await self.ha_client.get_state(entity_id)
        return self.ha_client.get_float_value(state_data, default)
    
    async def _get_hvac_action(self) -> str:
        """Get current HVAC action."""
        if not self.climate_entity:
            return "idle"
            
        state_data = await self.ha_client.get_state(self.climate_entity)
        if not state_data:
            return "idle"
            
        attributes = state_data.get('attributes', {})
        hvac_action = attributes.get('hvac_action', 'idle')
        
        # Map Home Assistant actions to our model
        action_map = {
            'heating': 'heating',
            'cooling': 'cooling',
            'idle': 'idle',
            'off': 'idle'
        }
        
        return action_map.get(hvac_action, 'idle')
    
    async def _update_rc_model(self, current_temp: float):
        """Update RC model parameters using recursive least squares."""
        try:
            if not self._past_preds:
                return
                
            # Get the most recent prediction for comparison
            predicted_temp = self._past_preds[-1]
            actual_change = current_temp - predicted_temp
            
            # Simple parameter update (simplified from original AppDaemon version)
            error = abs(actual_change)
            
            if error > 0.1:  # Only update if error is significant
                # Update parameters with simple gradient descent
                learning_rate = 0.01
                self.theta[0] = clip(self.theta[0] * (1 + learning_rate * error), 
                                   1.0/(self.MAX_TAU_H*60.0), 1.0/(self.MIN_TAU_H*60.0))
                
                self.samples += 1
                
                if self.samples % 100 == 0:
                    tau_hours = 1.0 / (self.theta[0] * 60.0)
                    logger.info(f"RC model update #{self.samples}: tau={tau_hours:.1f}h, error={error:.3f}")
                    
        except Exception as e:
            logger.error(f"Error updating RC model: {e}")
    
    async def _generate_forecast(self, indoor_temp: float, outdoor_temp: float,
                               indoor_humidity: float, outdoor_humidity: float,
                               solar_irradiance: float, hvac_action: str) -> Dict[str, Any]:
        """Generate temperature forecast."""
        try:
            # Build forecast horizon
            forecast_times = []
            forecast_temps_idle = []
            forecast_temps_controlled = []
            
            current_time = datetime.now(timezone.utc)
            
            # Simple RC model simulation
            temp = indoor_temp
            
            for hour in range(self.forecast_hours):
                forecast_time = current_time + timedelta(hours=hour)
                forecast_times.append(forecast_time.isoformat())
                
                # Simple temperature prediction using RC model
                # dT/dt = a*(Tout - Tin) + kH*Ih + kC*Ic + b + kE*(hout - hin) + kS*Solar
                a, kH, kC, b, kE, kS = self.theta
                
                # Calculate enthalpy difference
                h_out = _enthalpy_kj_per_kg_dryair(outdoor_temp, outdoor_humidity)
                h_in = _enthalpy_kj_per_kg_dryair(temp, indoor_humidity)
                
                # Calculate temperature change for idle mode
                dt_idle = a * (outdoor_temp - temp) + b + kE * (h_out - h_in) + kS * solar_irradiance
                temp_idle = temp + dt_idle
                
                # For controlled mode, assume ideal temperature control
                temp_controlled = max(self.comfort_min_temp, 
                                    min(self.comfort_max_temp, temp_idle))
                
                forecast_temps_idle.append(round(temp_idle, 1))
                forecast_temps_controlled.append(round(temp_controlled, 1))
                
                temp = temp_idle  # Use idle temp for next iteration
            
            # ML Enhancement
            ml_enhanced_prediction = forecast_temps_idle[0] if forecast_temps_idle else indoor_temp
            ml_metadata = {'method': 'rc_only'}
            
            if self.ml_enhancer:
                try:
                    # Create features for ML
                    now = datetime.now()
                    time_features = {
                        'hour_sin': math.sin(2 * math.pi * now.hour / 24),
                        'hour_cos': math.cos(2 * math.pi * now.hour / 24),
                        'day_of_year_sin': math.sin(2 * math.pi * now.timetuple().tm_yday / 365)
                    }
                    
                    features = self.ml_enhancer.create_features(
                        indoor_temp, outdoor_temp, indoor_humidity, outdoor_humidity,
                        solar_irradiance, hvac_action, time_features
                    )
                    
                    # Get ML-enhanced prediction
                    ml_enhanced_prediction, ml_metadata = self.ml_enhancer.predict_enhancement(
                        features, forecast_temps_idle[0] if forecast_temps_idle else indoor_temp
                    )
                    
                    # Add training sample
                    if len(self._past_preds) > 0:
                        actual_change = indoor_temp - self._past_preds[-1]
                        rc_prediction = forecast_temps_idle[0] if forecast_temps_idle else indoor_temp
                        self.ml_enhancer.add_sample(features, actual_change, rc_prediction, now)
                        
                except Exception as e:
                    logger.error(f"ML enhancement error: {e}")
            
            # Comfort band validation
            comfort_violation = {}
            if self.ml_enhancer:
                comfort_violation = self.ml_enhancer.validate_comfort_band_violation(
                    indoor_temp, ml_enhanced_prediction, 
                    self.comfort_min_temp, self.comfort_max_temp, hvac_action
                )
            
            # Store prediction for next iteration
            self._past_preds.append(ml_enhanced_prediction)
            if len(self._past_preds) > 5:
                self._past_preds.pop(0)
            
            # Calculate comfort and efficiency scores
            comfort_score = self._calculate_comfort_score(indoor_temp, indoor_humidity)
            efficiency_score = self._calculate_efficiency_score(
                abs(outdoor_temp - indoor_temp), hvac_action, solar_irradiance
            )
            
            return {
                'timestamp': current_time.isoformat(),
                'current_indoor_temp': indoor_temp,
                'current_outdoor_temp': outdoor_temp,
                'hvac_action': hvac_action,
                'forecast_times': forecast_times,
                'forecast_idle': forecast_temps_idle,
                'forecast_controlled': forecast_temps_controlled,
                'ml_enhanced_prediction': ml_enhanced_prediction,
                'ml_metadata': ml_metadata,
                'comfort_violation': comfort_violation,
                'comfort_score': comfort_score,
                'efficiency_score': efficiency_score,
                'rc_parameters': {
                    'tau_hours': 1.0 / (self.theta[0] * 60.0),
                    'k_heat': self.theta[1],
                    'k_cool': self.theta[2],
                    'bias': self.theta[3],
                    'k_enthalpy': self.theta[4],
                    'k_solar': self.theta[5]
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return {}
    
    def _calculate_comfort_score(self, temp: float, humidity: float) -> float:
        """Calculate comfort score (0-100) based on temperature and humidity."""
        try:
            # Optimal ranges: 68-75°F temperature, 40-60% humidity
            temp_score = 100.0
            if temp < 68.0:
                temp_score = max(0.0, 100.0 - 2.0 * (68.0 - temp))
            elif temp > 75.0:
                temp_score = max(0.0, 100.0 - 2.0 * (temp - 75.0))
            
            humidity_score = 100.0
            if humidity < 40.0:
                humidity_score = max(0.0, 100.0 - 2.0 * (40.0 - humidity))
            elif humidity > 60.0:
                humidity_score = max(0.0, 100.0 - 2.0 * (humidity - 60.0))
            
            return round((temp_score + humidity_score) / 2.0, 1)
        except:
            return 50.0
    
    def _calculate_efficiency_score(self, temp_diff: float, hvac_action: str, 
                                  solar_irradiance: float) -> float:
        """Calculate energy efficiency score based on conditions."""
        try:
            # Base score starts at 100
            score = 100.0
            
            # Penalize large temperature differentials
            if temp_diff > 20.0:
                score -= min(50.0, (temp_diff - 20.0) * 2.0)
            
            # Bonus for using solar energy effectively
            if solar_irradiance > 200.0 and hvac_action == 'idle':
                score += 10.0
            
            # Penalize running HVAC with doors/windows open (approximated by large temp diff)
            if hvac_action != 'idle' and temp_diff > 30.0:
                score -= 20.0
            
            return round(max(0.0, min(100.0, score)), 1)
        except:
            return 50.0
    
    async def _train_ml_models(self):
        """Train ML models periodically."""
        if not self.ml_training_enabled or not self.ml_enhancer:
            return
            
        try:
            logger.info("Starting ML model training...")
            scores = self.ml_enhancer.train_models()
            
            if scores:
                logger.info(f"ML models trained successfully. Scores: {scores}")
                
                # Publish ML status to Home Assistant
                status = self.ml_enhancer.get_model_status()
                await self.ha_client.set_state(
                    "sensor.smart_climate_ml_status",
                    "trained",
                    {
                        "models_trained": status.get('models_trained', 0),
                        "training_samples": status.get('training_samples', 0),
                        "scores": scores,
                        "ensemble_weights": status.get('ensemble_weights'),
                        "friendly_name": "Smart Climate ML Status"
                    }
                )
            else:
                logger.info("ML training skipped - insufficient data")
                
        except Exception as e:
            logger.error(f"ML training error: {e}")
    
    async def _refresh_accuweather_forecast(self):
        """Refresh AccuWeather forecast data."""
        if not self.accuweather_api_key or not self.accuweather_location_key:
            return
            
        try:
            url = f"https://dataservice.accuweather.com/forecasts/v1/hourly/12hour/{self.accuweather_location_key}"
            params = {
                'apikey': self.accuweather_api_key,
                'details': 'true'
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data:
                    self._aw_cache = []
                    for item in data:
                        temp_f = item.get('Temperature', {}).get('Value')
                        dt = item.get('DateTime')
                        rh = item.get('RelativeHumidity', 50)
                        solar = item.get('SolarIrradiance', {}).get('Value', 0.0)
                        
                        if temp_f and dt:
                            self._aw_cache.append({
                                'DateTime': dt,
                                'TempF': float(temp_f),
                                'RH': rh,
                                'Solar': float(solar)
                            })
                    
                    self._aw_cache_ts = datetime.now(timezone.utc)
                    logger.info(f"AccuWeather forecast updated: {len(self._aw_cache)} hours")
            else:
                logger.warning(f"AccuWeather API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"AccuWeather error: {e}")
    
    async def _publish_forecast(self, forecast_data: Dict[str, Any]):
        """Publish forecast data to Home Assistant sensors."""
        if not forecast_data:
            return
            
        try:
            # Publish main forecast sensor
            forecast_series = []
            if 'forecast_times' in forecast_data and 'forecast_idle' in forecast_data:
                for i, (time_str, temp) in enumerate(zip(forecast_data['forecast_times'], forecast_data['forecast_idle'])):
                    forecast_series.append({'t': time_str, 'y': temp})
            
            await self.ha_client.set_state(
                "sensor.smart_climate_indoor_forecast",
                forecast_data.get('ml_enhanced_prediction', 0),
                {
                    "series_idle": forecast_series,
                    "series_controlled": [
                        {'t': t, 'y': temp} 
                        for t, temp in zip(forecast_data.get('forecast_times', []), 
                                         forecast_data.get('forecast_controlled', []))
                    ],
                    "current_temp": forecast_data.get('current_indoor_temp', 0),
                    "hvac_action": forecast_data.get('hvac_action', 'idle'),
                    "forecast_method": forecast_data.get('ml_metadata', {}).get('method', 'rc_only'),
                    "friendly_name": "Smart Climate Indoor Forecast",
                    "unit_of_measurement": "°F"
                }
            )
            
            # Publish ML enhanced prediction
            ml_metadata = forecast_data.get('ml_metadata', {})
            await self.ha_client.set_state(
                "sensor.smart_climate_ml_enhanced_prediction",
                forecast_data.get('ml_enhanced_prediction', 0),
                {
                    "confidence": ml_metadata.get('confidence', 0.5),
                    "method": ml_metadata.get('method', 'rc_only'),
                    "rc_prediction": ml_metadata.get('rc_prediction', 0),
                    "ml_prediction": ml_metadata.get('ml_prediction', 0),
                    "friendly_name": "Smart Climate ML Enhanced Prediction",
                    "unit_of_measurement": "°F"
                }
            )
            
            # Publish comfort violation sensor
            violation = forecast_data.get('comfort_violation', {})
            await self.ha_client.set_state(
                "sensor.smart_climate_comfort_violation",
                "detected" if violation.get('violation_detected', False) else "none",
                {
                    "violation_type": violation.get('violation_type', 'none'),
                    "severity": violation.get('severity', 'none'),
                    "confidence": violation.get('confidence', 0.0),
                    "recommendation": violation.get('recommendation', 'none'),
                    "friendly_name": "Smart Climate Comfort Violation"
                }
            )
            
            # Publish comfort and efficiency scores
            await self.ha_client.set_state(
                "sensor.smart_climate_comfort_score",
                forecast_data.get('comfort_score', 50),
                {
                    "friendly_name": "Smart Climate Comfort Score",
                    "unit_of_measurement": "%"
                }
            )
            
            await self.ha_client.set_state(
                "sensor.smart_climate_efficiency_score",
                forecast_data.get('efficiency_score', 50),
                {
                    "friendly_name": "Smart Climate Energy Efficiency",
                    "unit_of_measurement": "%"
                }
            )
            
            # Publish RC model parameters
            await self.ha_client.set_state(
                "sensor.smart_climate_tau_hours",
                self.tau_hours,
                {
                    "friendly_name": "Smart Climate Tau Hours",
                    "unit_of_measurement": "h"
                }
            )
            
            await self.ha_client.set_state(
                "sensor.smart_climate_k_heat",
                self.k_heat,
                {
                    "friendly_name": "Smart Climate K Heat"
                }
            )
            
            await self.ha_client.set_state(
                "sensor.smart_climate_k_cool",
                self.k_cool,
                {
                    "friendly_name": "Smart Climate K Cool"
                }
            )
            
            await self.ha_client.set_state(
                "sensor.smart_climate_bias",
                self.bias,
                {
                    "friendly_name": "Smart Climate Model Bias",
                    "unit_of_measurement": "°F"
                }
            )
            
            await self.ha_client.set_state(
                "sensor.smart_climate_forgetting_factor",
                self.forgetting_factor,
                {
                    "friendly_name": "Smart Climate Forgetting Factor"
                }
            )
            
            await self.ha_client.set_state(
                "sensor.smart_climate_comfort_cap",
                self.comfort_cap,
                {
                    "friendly_name": "Smart Climate Comfort Cap",
                    "unit_of_measurement": "°F"
                }
            )
            
            await self.ha_client.set_state(
                "sensor.smart_climate_heat_min_f",
                self.heat_min_f,
                {
                    "friendly_name": "Smart Climate Heat Minimum",
                    "unit_of_measurement": "°F"
                }
            )
            
            await self.ha_client.set_state(
                "sensor.smart_climate_recommendation_cooldown",
                self.recommendation_cooldown,
                {
                    "friendly_name": "Smart Climate Recommendation Cooldown",
                    "unit_of_measurement": "min"
                }
            )
            
            logger.debug("Forecast data published to Home Assistant")
            
        except Exception as e:
            logger.error(f"Error publishing forecast: {e}")