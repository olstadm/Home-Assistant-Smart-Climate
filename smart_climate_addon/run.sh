#!/usr/bin/with-contenv bashio

# Set log level
bashio::log.info "Starting Smart Climate Forecasting Add-on..."

# Get configuration
CONFIG_PATH="/data/options.json"
if bashio::fs.file_exists "${CONFIG_PATH}"; then
    bashio::log.info "Found configuration file"
else
    bashio::log.fatal "Configuration file not found!"
    exit 1
fi

# Extract configuration values
INDOOR_TEMP_SENSOR=$(bashio::config 'indoor_temperature_sensor')
OUTDOOR_TEMP_SENSOR=$(bashio::config 'outdoor_temperature_sensor')
CLIMATE_ENTITY=$(bashio::config 'climate_entity')
INDOOR_HUMIDITY_SENSOR=$(bashio::config 'indoor_humidity_sensor')
OUTDOOR_HUMIDITY_SENSOR=$(bashio::config 'outdoor_humidity_sensor')
LEARNING_ENABLED=$(bashio::config 'learning_enabled')
FORECAST_HOURS=$(bashio::config 'forecast_hours')
UPDATE_INTERVAL=$(bashio::config 'update_interval_minutes')
COMFORT_MAX_TEMP=$(bashio::config 'comfort_max_temp')
COMFORT_MIN_TEMP=$(bashio::config 'comfort_min_temp')
ACCUWEATHER_API_KEY=$(bashio::config 'accuweather_api_key')
ACCUWEATHER_LOCATION_KEY=$(bashio::config 'accuweather_location_key')
ML_TRAINING_ENABLED=$(bashio::config 'ml_training_enabled')
ML_TRAINING_INTERVAL=$(bashio::config 'ml_training_interval_hours')

# Get Home Assistant details
SUPERVISOR_TOKEN="${SUPERVISOR_TOKEN:-}"
HASSIO_TOKEN="${HASSIO_TOKEN:-}"

if [ -z "$SUPERVISOR_TOKEN" ] && [ -z "$HASSIO_TOKEN" ]; then
    bashio::log.fatal "No supervisor token available!"
    exit 1
fi

# Export environment variables for the Python application
export INDOOR_TEMP_SENSOR
export OUTDOOR_TEMP_SENSOR
export CLIMATE_ENTITY
export INDOOR_HUMIDITY_SENSOR
export OUTDOOR_HUMIDITY_SENSOR
export LEARNING_ENABLED
export FORECAST_HOURS
export UPDATE_INTERVAL
export COMFORT_MAX_TEMP
export COMFORT_MIN_TEMP
export ACCUWEATHER_API_KEY
export ACCUWEATHER_LOCATION_KEY
export ML_TRAINING_ENABLED
export ML_TRAINING_INTERVAL
export SUPERVISOR_TOKEN="${SUPERVISOR_TOKEN:-$HASSIO_TOKEN}"
export HOME_ASSISTANT_URL="http://supervisor/core"
export DATA_DIR="/data"
export MODELS_DIR="/data/models"

bashio::log.info "Configuration loaded successfully"
bashio::log.info "Indoor temperature sensor: ${INDOOR_TEMP_SENSOR}"
bashio::log.info "Outdoor temperature sensor: ${OUTDOOR_TEMP_SENSOR}"
bashio::log.info "Climate entity: ${CLIMATE_ENTITY}"
bashio::log.info "Learning enabled: ${LEARNING_ENABLED}"
bashio::log.info "ML training enabled: ${ML_TRAINING_ENABLED}"

# Create models directory if it doesn't exist
mkdir -p "${MODELS_DIR}"

# Start the Python application
bashio::log.info "Starting Smart Climate service..."
cd /app
exec python3 -m smart_climate.main