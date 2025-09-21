"""Smart Climate Forecasting Add-on Main Entry Point."""
import asyncio
import json
import logging
import os
import signal
import sys
from typing import Dict, Any, List

from .ha_client import HomeAssistantClient
from .climate_forecaster import ClimateForecaster


def setup_logging():
    """Set up logging configuration."""
    log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    debug_mode = os.environ.get('DEBUG_MODE', 'false').lower() == 'true'
    
    if debug_mode:
        log_level = 'DEBUG'
    
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set specific logger levels
    if debug_mode:
        logging.getLogger('aiohttp').setLevel(logging.INFO)
        logging.getLogger('asyncio').setLevel(logging.INFO)
    else:
        logging.getLogger('aiohttp').setLevel(logging.WARNING)
        logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    if debug_mode:
        logger.info("Debug mode enabled - verbose logging activated")
        logger.debug("Available environment variables:")
        for key, value in sorted(os.environ.items()):
            if 'TOKEN' in key or 'KEY' in key:
                logger.debug(f"  {key}=<hidden>")
            else:
                logger.debug(f"  {key}={value}")
    
    return debug_mode


def load_config() -> Dict[str, Any]:
    """Load configuration from environment and options file."""
    logger = logging.getLogger(__name__)
    config = {}
    
    logger.info("Loading configuration from environment variables")
    
    # Load from environment variables (set by run.sh)
    config['indoor_temperature_sensor'] = os.environ.get('INDOOR_TEMP_SENSOR', '')
    config['outdoor_temperature_sensor'] = os.environ.get('OUTDOOR_TEMP_SENSOR', '')
    config['climate_entity'] = os.environ.get('CLIMATE_ENTITY', '')
    config['indoor_humidity_sensor'] = os.environ.get('INDOOR_HUMIDITY_SENSOR', '')
    config['outdoor_humidity_sensor'] = os.environ.get('OUTDOOR_HUMIDITY_SENSOR', '')
    config['learning_enabled'] = os.environ.get('LEARNING_ENABLED', 'true').lower() == 'true'
    config['forecast_hours'] = int(os.environ.get('FORECAST_HOURS', '12'))
    config['update_interval_minutes'] = int(os.environ.get('UPDATE_INTERVAL', '5'))
    config['comfort_max_temp'] = float(os.environ.get('COMFORT_MAX_TEMP', '80.0'))
    config['comfort_min_temp'] = float(os.environ.get('COMFORT_MIN_TEMP', '62.0'))
    config['accuweather_api_key'] = os.environ.get('ACCUWEATHER_API_KEY', '')
    config['accuweather_location_key'] = os.environ.get('ACCUWEATHER_LOCATION_KEY', '')
    config['ml_training_enabled'] = os.environ.get('ML_TRAINING_ENABLED', 'true').lower() == 'true'
    config['ml_training_interval_hours'] = int(os.environ.get('ML_TRAINING_INTERVAL', '1'))
    
    # Add helper entity mappings for dynamic reading
    config['helper_entities'] = {
        'home_forecast': os.environ.get('HELPER_HOME_FORECAST', ''),
        'accuweather_location_key': os.environ.get('HELPER_ACCUWEATHER_LOCATION_KEY', ''),
        'accuweather_token': os.environ.get('HELPER_ACCUWEATHER_TOKEN', ''),
        'forecast_hours': os.environ.get('HELPER_HOME_MODEL_FORECAST_HOURS', ''),
        'climate_entity': os.environ.get('HELPER_HOME_MODEL_CLIMATE_ENTITY', ''),
        'outdoor_sensor': os.environ.get('HELPER_HOME_MODEL_OUTDOOR_SENSOR', ''),
        'indoor_sensor': os.environ.get('HELPER_HOME_MODEL_INDOOR_SENSOR', ''),
        'outdoor_humidity_entity': os.environ.get('HELPER_HOME_MODEL_OUTDOOR_HUMIDITY_ENTITY', ''),
        'indoor_humidity_entity': os.environ.get('HELPER_HOME_MODEL_INDOOR_HUMIDITY_ENTITY', ''),
        'tau_hours': os.environ.get('HELPER_HOME_MODEL_TAU_HOURS', ''),
        'update_minutes': os.environ.get('HELPER_HOME_MODEL_UPDATE_MINUTES', ''),
        'forgetting_factor': os.environ.get('HELPER_HOME_MODEL_FORGETTING_FACTOR', ''),
        'bias': os.environ.get('HELPER_HOME_MODEL_BIAS', ''),
        'comfort_cap': os.environ.get('HELPER_HOME_MODEL_COMFORT_CAP', ''),
        'heat_min_f': os.environ.get('HELPER_HOME_MODEL_HEAT_MIN_F', ''),
        'k_heat': os.environ.get('HELPER_HOME_MODEL_K_HEAT', ''),
        'k_cool': os.environ.get('HELPER_HOME_MODEL_K_COOL', ''),
        'learning_enabled': os.environ.get('HELPER_HOME_MODEL_LEARNING_ENABLED', ''),
        'recommendation_cooldown': os.environ.get('HELPER_HOME_MODEL_RECOMMENDATION_COOLDOWN', ''),
        'storage': os.environ.get('HELPER_HOME_MODEL_STORAGE', '')
    }
    
    debug_mode = os.environ.get('DEBUG_MODE', 'false').lower() == 'true'
    if debug_mode:
        logger.debug("Configuration loaded:")
        for key, value in config.items():
            if key == 'helper_entities':
                logger.debug(f"  {key}: {len(value)} helper entities mapped")
                for helper_key, helper_value in value.items():
                    logger.debug(f"    {helper_key}: {helper_value}")
            elif 'key' in key.lower() or 'token' in key.lower():
                logger.debug(f"  {key}: <hidden>" if value else f"  {key}: <empty>")
            else:
                logger.debug(f"  {key}: {value}")
    
    return config


async def load_helper_config(ha_client: HomeAssistantClient, config: Dict[str, Any]) -> Dict[str, Any]:
    """Load configuration from helper entities, falling back to config values."""
    logger = logging.getLogger(__name__)
    enhanced_config = config.copy()
    helper_entities = config.get('helper_entities', {})
    
    logger.info("Checking helper entities for configuration overrides...")
    
    # Helper entity mappings to config keys
    helper_mappings = {
        'accuweather_location_key': ('accuweather_location_key', str),
        'accuweather_token': ('accuweather_api_key', str),
        'forecast_hours': ('forecast_hours', int),
        'climate_entity': ('climate_entity', str),
        'outdoor_sensor': ('outdoor_temperature_sensor', str),
        'indoor_sensor': ('indoor_temperature_sensor', str),
        'outdoor_humidity_entity': ('outdoor_humidity_sensor', str),
        'indoor_humidity_entity': ('indoor_humidity_sensor', str),
        'learning_enabled': ('learning_enabled', bool),
    }
    
    helper_read_count = 0
    for helper_key, (config_key, value_type) in helper_mappings.items():
        entity_id = helper_entities.get(helper_key, '')
        if entity_id:
            try:
                state = await ha_client.get_state(entity_id)
                if state and 'state' in state:
                    raw_value = state['state']
                    
                    # Convert to appropriate type
                    if value_type == bool:
                        if isinstance(raw_value, str):
                            converted_value = raw_value.lower() in ('true', 'on', '1', 'yes')
                        else:
                            converted_value = bool(raw_value)
                    elif value_type == int:
                        converted_value = int(float(raw_value))
                    elif value_type == float:
                        converted_value = float(raw_value)
                    else:
                        converted_value = str(raw_value)
                    
                    # Only override if the helper has a meaningful value
                    if (value_type == str and converted_value.strip()) or (value_type != str and converted_value is not None):
                        old_value = enhanced_config.get(config_key)
                        enhanced_config[config_key] = converted_value
                        logger.info(f"✓ Helper override: {config_key} = {converted_value} (from {entity_id})")
                        if old_value != converted_value:
                            logger.info(f"  Previous value: {old_value}")
                        helper_read_count += 1
                    else:
                        logger.debug(f"Helper {entity_id} has empty/invalid value: {raw_value}")
                else:
                    logger.warning(f"Helper entity {entity_id} has no state")
            except Exception as e:
                logger.warning(f"Failed to read helper entity {entity_id}: {e}")
    
    logger.info(f"Successfully read {helper_read_count} helper entity overrides")
    
    # Additional helper entities for advanced model parameters (stored in enhanced_config for later use)
    advanced_helpers = {
        'tau_hours': 'tau_hours',
        'update_minutes': 'update_minutes', 
        'forgetting_factor': 'forgetting_factor',
        'bias': 'bias',
        'comfort_cap': 'comfort_cap',
        'heat_min_f': 'heat_min_f',
        'k_heat': 'k_heat',
        'k_cool': 'k_cool',
        'recommendation_cooldown': 'recommendation_cooldown',
        'storage': 'storage_path'
    }
    
    enhanced_config['advanced_params'] = {}
    for helper_key, param_key in advanced_helpers.items():
        entity_id = helper_entities.get(helper_key, '')
        if entity_id:
            try:
                state = await ha_client.get_state(entity_id)
                if state and 'state' in state and state['state'] not in ('unknown', 'unavailable', ''):
                    value = float(state['state']) if param_key != 'storage_path' else str(state['state'])
                    enhanced_config['advanced_params'][param_key] = value
                    logger.info(f"✓ Advanced parameter: {param_key} = {value} (from {entity_id})")
            except Exception as e:
                logger.debug(f"Could not read advanced helper {entity_id}: {e}")
    
    return enhanced_config


class SmartClimateApp:
    """Main Smart Climate application."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.running = True
        self.forecaster = None
        self.ha_client = None
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    async def run(self):
        """Run the Smart Climate application."""
        # Set up signal handlers
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        try:
            # Load initial configuration
            config = load_config()
            self.logger.info("Initial configuration loaded from environment")
            
            # Set up Home Assistant client
            ha_url = os.environ.get('HOME_ASSISTANT_URL', 'http://supervisor/core')
            ha_token = os.environ.get('SUPERVISOR_TOKEN', '')
            
            if not ha_token:
                self.logger.error("Home Assistant token not available")
                return 1
            
            self.logger.info(f"Connecting to Home Assistant at {ha_url}")
            
            async with HomeAssistantClient(ha_url, ha_token) as ha_client:
                self.ha_client = ha_client
                
                # Test connection
                self.logger.info("Testing Home Assistant connection...")
                config_data = await ha_client.get_config()
                if config_data:
                    self.logger.info(f"✓ Connected to Home Assistant: {config_data.get('location_name', 'Unknown')}")
                else:
                    self.logger.error("Failed to connect to Home Assistant")
                    return 1
                
                # Load helper entity configuration
                self.logger.info("Loading configuration from helper entities...")
                try:
                    config = await load_helper_config(ha_client, config)
                    self.logger.info("✓ Helper entity configuration loaded successfully")
                except Exception as e:
                    self.logger.warning(f"Helper entity loading failed, using base config: {e}")
                
                # Validate final configuration
                validation_errors = await self._validate_configuration(ha_client, config)
                if validation_errors:
                    self.logger.error("Configuration validation failed:")
                    for error in validation_errors:
                        self.logger.error(f"  - {error}")
                    return 1
                
                self.logger.info("✓ Configuration validation passed")
                
                # Initialize forecaster
                self.logger.info("Initializing climate forecaster...")
                self.forecaster = ClimateForecaster(ha_client, config)
                self.logger.info("✓ Climate forecaster initialized")
                
                # Start the forecasting service
                self.logger.info("Starting Smart Climate Forecasting service...")
                
                # Create health check endpoint task
                health_task = asyncio.create_task(self._health_server())
                
                # Run forecast loop
                forecast_task = asyncio.create_task(self.forecaster.run_forecast_loop())
                
                # Wait for shutdown signal or task completion
                try:
                    await asyncio.gather(health_task, forecast_task)
                except asyncio.CancelledError:
                    self.logger.info("Tasks cancelled, shutting down...")
                
                return 0
                
        except Exception as e:
            self.logger.error(f"Application error: {e}", exc_info=True)
            return 1
    
    async def _validate_configuration(self, ha_client: HomeAssistantClient, config: Dict[str, Any]) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate required configuration
        required_fields = [
            ('indoor_temperature_sensor', 'Indoor temperature sensor'),
            ('outdoor_temperature_sensor', 'Outdoor temperature sensor'),
            ('climate_entity', 'Climate entity')
        ]
        
        for field, description in required_fields:
            if not config.get(field):
                errors.append(f"{description} not configured")
        
        # Validate entities exist in Home Assistant
        entities_to_check = [
            (config.get('indoor_temperature_sensor'), 'Indoor temperature sensor'),
            (config.get('outdoor_temperature_sensor'), 'Outdoor temperature sensor'),
            (config.get('climate_entity'), 'Climate entity')
        ]
        
        # Add optional entities if configured
        if config.get('indoor_humidity_sensor'):
            entities_to_check.append((config['indoor_humidity_sensor'], 'Indoor humidity sensor'))
        if config.get('outdoor_humidity_sensor'):
            entities_to_check.append((config['outdoor_humidity_sensor'], 'Outdoor humidity sensor'))
        
        for entity_id, description in entities_to_check:
            if entity_id:
                try:
                    state = await ha_client.get_state(entity_id)
                    if not state:
                        errors.append(f"{description} '{entity_id}' not found in Home Assistant")
                    elif state.get('state') in ('unknown', 'unavailable'):
                        self.logger.warning(f"{description} '{entity_id}' is currently {state.get('state')}")
                    else:
                        self.logger.info(f"✓ {description} '{entity_id}' is available (state: {state.get('state')})")
                except Exception as e:
                    errors.append(f"Failed to validate {description} '{entity_id}': {e}")
        
        # Validate numeric ranges
        if config.get('forecast_hours', 0) < 1 or config.get('forecast_hours', 0) > 24:
            errors.append(f"Forecast hours must be between 1 and 24, got {config.get('forecast_hours')}")
        
        if config.get('update_interval_minutes', 0) < 1 or config.get('update_interval_minutes', 0) > 60:
            errors.append(f"Update interval must be between 1 and 60 minutes, got {config.get('update_interval_minutes')}")
        
        # Validate temperature ranges
        comfort_min = config.get('comfort_min_temp', 62.0)
        comfort_max = config.get('comfort_max_temp', 80.0)
        if comfort_min >= comfort_max:
            errors.append(f"Comfort min temp ({comfort_min}) must be less than max temp ({comfort_max})")
        
        return errors
    
    async def _health_server(self):
        """Simple health check HTTP server."""
        from aiohttp import web
        
        async def health_check(request):
            """Health check endpoint."""
            return web.Response(text="OK", status=200)
        
        app = web.Application()
        app.router.add_get('/health', health_check)
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, '0.0.0.0', 8099)
        await site.start()
        
        self.logger.info("Health check server started on port 8099")
        
        # Keep running until shutdown
        while self.running:
            await asyncio.sleep(1)
        
        await runner.cleanup()


async def main():
    """Main entry point."""
    debug_mode = setup_logging()
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Smart Climate Forecasting Add-on v1.0.1 (debug: {debug_mode})")
    
    app = SmartClimateApp()
    exit_code = await app.run()
    
    logger.info("Smart Climate Add-on stopped")
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())