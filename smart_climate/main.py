"""Smart Climate Forecasting Add-on Main Entry Point."""
import asyncio
import json
import logging
import os
import signal
import sys
from typing import Dict, Any

from .ha_client import HomeAssistantClient
from .climate_forecaster import ClimateForecaster


def setup_logging():
    """Set up logging configuration."""
    log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set specific logger levels
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)


def load_config() -> Dict[str, Any]:
    """Load configuration from environment and options file."""
    config = {}
    
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
    
    return config


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
            # Load configuration
            config = load_config()
            self.logger.info("Configuration loaded")
            
            # Validate required configuration
            if not config['indoor_temperature_sensor']:
                self.logger.error("Indoor temperature sensor not configured")
                return 1
            
            if not config['outdoor_temperature_sensor']:
                self.logger.error("Outdoor temperature sensor not configured")
                return 1
            
            if not config['climate_entity']:
                self.logger.error("Climate entity not configured")
                return 1
            
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
                config_data = await ha_client.get_config()
                if config_data:
                    self.logger.info(f"Connected to Home Assistant: {config_data.get('location_name', 'Unknown')}")
                else:
                    self.logger.error("Failed to connect to Home Assistant")
                    return 1
                
                # Initialize forecaster
                self.forecaster = ClimateForecaster(ha_client, config)
                
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
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Smart Climate Forecasting Add-on v1.0.0")
    
    app = SmartClimateApp()
    exit_code = await app.run()
    
    logger.info("Smart Climate Add-on stopped")
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())