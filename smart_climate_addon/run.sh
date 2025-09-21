#!/usr/bin/with-contenv bashio

# Exit on any error and enable debugging
set -e

# Function to safely call bashio::log.info with error handling
log_info() {
    if command -v bashio::log.info >/dev/null 2>&1; then
        bashio::log.info "$@"
    else
        echo "[INFO] $@"
    fi
}

# Function to safely call bashio::log.fatal with error handling
log_fatal() {
    if command -v bashio::log.fatal >/dev/null 2>&1; then
        bashio::log.fatal "$@"
    else
        echo "[FATAL] $@"
    fi
}

# Function to safely call bashio::log.warning with error handling
log_warning() {
    if command -v bashio::log.warning >/dev/null 2>&1; then
        bashio::log.warning "$@"
    else
        echo "[WARNING] $@"
    fi
}

# Enable debugging if requested
DEBUG=$(bashio::config 'debug' 'false' 2>/dev/null || echo 'false')
if [[ "${DEBUG}" == "true" ]]; then
    set -x
    log_info "Debug mode enabled"
fi

# Set log level
log_info "Starting Smart Climate Forecasting Add-on..."

# Get configuration
CONFIG_PATH="/data/options.json"
if bashio::fs.file_exists "${CONFIG_PATH}"; then
    log_info "Found configuration file"
else
    log_fatal "Configuration file not found!"
    exit 1
fi

# Extract configuration values with error handling
log_info "Loading configuration values..."

# Validate required configuration exists first
if ! bashio::config.exists 'indoor_temperature_sensor'; then
    log_fatal "Required configuration 'indoor_temperature_sensor' is missing!"
    exit 1
fi

if ! bashio::config.exists 'outdoor_temperature_sensor'; then
    log_fatal "Required configuration 'outdoor_temperature_sensor' is missing!"
    exit 1
fi

if ! bashio::config.exists 'climate_entity'; then
    log_fatal "Required configuration 'climate_entity' is missing!"
    exit 1
fi

INDOOR_TEMP_SENSOR=$(bashio::config 'indoor_temperature_sensor')
OUTDOOR_TEMP_SENSOR=$(bashio::config 'outdoor_temperature_sensor')
CLIMATE_ENTITY=$(bashio::config 'climate_entity')
INDOOR_HUMIDITY_SENSOR=$(bashio::config 'indoor_humidity_sensor' '')
OUTDOOR_HUMIDITY_SENSOR=$(bashio::config 'outdoor_humidity_sensor' '')
LEARNING_ENABLED=$(bashio::config 'learning_enabled' 'true')
FORECAST_HOURS=$(bashio::config 'forecast_hours' '12')
UPDATE_INTERVAL=$(bashio::config 'update_interval_minutes' '5')
COMFORT_MAX_TEMP=$(bashio::config 'comfort_max_temp' '80.0')
COMFORT_MIN_TEMP=$(bashio::config 'comfort_min_temp' '62.0')
ACCUWEATHER_API_KEY=$(bashio::config 'accuweather_api_key' '')
ACCUWEATHER_LOCATION_KEY=$(bashio::config 'accuweather_location_key' '')
ML_TRAINING_ENABLED=$(bashio::config 'ml_training_enabled' 'true')
ML_TRAINING_INTERVAL=$(bashio::config 'ml_training_interval_hours' '1')
DEBUG=$(bashio::config 'debug' 'false')

# Get Home Assistant details with proper error handling
log_info "Checking for Home Assistant supervisor token..."

SUPERVISOR_TOKEN="${SUPERVISOR_TOKEN:-}"
HASSIO_TOKEN="${HASSIO_TOKEN:-}"

# Debug token availability
if [[ -n "${SUPERVISOR_TOKEN}" ]]; then
    log_info "SUPERVISOR_TOKEN is available"
elif [[ -n "${HASSIO_TOKEN}" ]]; then
    log_info "HASSIO_TOKEN is available"
else
    log_warning "No supervisor tokens found in environment"
fi

if [[ -z "$SUPERVISOR_TOKEN" ]] && [[ -z "$HASSIO_TOKEN" ]]; then
    log_fatal "No supervisor token available! Cannot communicate with Home Assistant."
    log_fatal "This add-on requires supervisor API access to function properly."
    exit 1
fi

# Export environment variables for the Python application
log_info "Exporting environment variables..."
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

log_info "Configuration loaded successfully"
log_info "Indoor temperature sensor: ${INDOOR_TEMP_SENSOR}"
log_info "Outdoor temperature sensor: ${OUTDOOR_TEMP_SENSOR}"
log_info "Climate entity: ${CLIMATE_ENTITY}"
log_info "Learning enabled: ${LEARNING_ENABLED}"
log_info "ML training enabled: ${ML_TRAINING_ENABLED}"
log_info "Data directory: ${DATA_DIR}"
log_info "Models directory: ${MODELS_DIR}"

# Create models directory if it doesn't exist
log_info "Creating models directory..."
mkdir -p "${MODELS_DIR}"

# Validate that the directory was created
if [[ ! -d "${MODELS_DIR}" ]]; then
    log_fatal "Failed to create models directory: ${MODELS_DIR}"
    exit 1
fi

log_info "Models directory ready: ${MODELS_DIR}"

# Start the Python application
log_info "Starting Smart Climate service..."
log_info "Working directory: $(pwd)"
log_info "Python path: ${PYTHONPATH:-/app}"

cd /app || {
    log_fatal "Failed to change to /app directory"
    exit 1
}

# Final debug output before starting the main application
log_info "About to start Python application with: python3 -m smart_climate.main"

exec python3 -m smart_climate.main