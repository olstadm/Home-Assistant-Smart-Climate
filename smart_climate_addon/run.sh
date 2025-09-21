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
    log_info "Debug mode enabled - verbose logging activated"
    log_info "Bash version: ${BASH_VERSION}"
    log_info "Shell options: $-"
    export DEBUG_MODE="true" 
else
    export DEBUG_MODE="false"
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
log_info "Validating required configuration parameters..."
required_configs=("indoor_temperature_sensor" "outdoor_temperature_sensor" "climate_entity")
for config_key in "${required_configs[@]}"; do
    if ! bashio::config.exists "$config_key"; then
        log_fatal "Required configuration '$config_key' is missing!"
        log_fatal "Please configure this parameter in the add-on settings"
        exit 1
    else
        log_info "✓ Required config '$config_key' found"
    fi
done

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
    log_info "✓ SUPERVISOR_TOKEN is available (length: ${#SUPERVISOR_TOKEN})"
elif [[ -n "${HASSIO_TOKEN}" ]]; then
    log_info "✓ HASSIO_TOKEN is available (length: ${#HASSIO_TOKEN})"
else
    log_warning "No supervisor tokens found in environment"
    log_info "Available environment variables:"
    env | grep -E "(SUPERVISOR|HASSIO|TOKEN)" | while read -r line; do
        var_name=$(echo "$line" | cut -d'=' -f1)
        log_info "  - $var_name"
    done
fi

if [[ -z "$SUPERVISOR_TOKEN" ]] && [[ -z "$HASSIO_TOKEN" ]]; then
    log_fatal "No supervisor token available! Cannot communicate with Home Assistant."
    log_fatal "This add-on requires supervisor API access to function properly."
    log_fatal "Please ensure you're running this as a Home Assistant add-on, not standalone."
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
export DEBUG_MODE
export SUPERVISOR_TOKEN="${SUPERVISOR_TOKEN:-$HASSIO_TOKEN}"
export HOME_ASSISTANT_URL="http://supervisor/core"
export DATA_DIR="/data"
export MODELS_DIR="/data/models"

# Export helper entity names for dynamic reading
log_info "Setting up helper entity mappings..."
export HELPER_HOME_FORECAST="input_text.home_forecast"
export HELPER_ACCUWEATHER_LOCATION_KEY="input_text.accuweather_location_key"
export HELPER_ACCUWEATHER_TOKEN="input_text.accuweather_token"
export HELPER_HOME_MODEL_FORECAST_HOURS="input_number.home_model_forecast_hours"
export HELPER_HOME_MODEL_CLIMATE_ENTITY="input_text.home_model_climate_entity"
export HELPER_HOME_MODEL_OUTDOOR_SENSOR="input_text.home_model_outdoor_sensor"
export HELPER_HOME_MODEL_INDOOR_SENSOR="input_text.home_model_indoor_sensor"
export HELPER_HOME_MODEL_OUTDOOR_HUMIDITY_ENTITY="input_text.home_model_outdoor_humidity_entity"
export HELPER_HOME_MODEL_INDOOR_HUMIDITY_ENTITY="input_text.home_model_indoor_humidity_entity"
export HELPER_HOME_MODEL_TAU_HOURS="input_number.home_model_tau_hours"
export HELPER_HOME_MODEL_UPDATE_MINUTES="input_number.home_model_update_minutes"
export HELPER_HOME_MODEL_FORGETTING_FACTOR="input_number.home_model_forgetting_factor"
export HELPER_HOME_MODEL_BIAS="input_number.home_model_bias"
export HELPER_HOME_MODEL_COMFORT_CAP="input_number.home_model_comfort_cap"
export HELPER_HOME_MODEL_HEAT_MIN_F="input_number.home_model_heat_min_f"
export HELPER_HOME_MODEL_K_HEAT="input_number.home_model_k_heat"
export HELPER_HOME_MODEL_K_COOL="input_number.home_model_k_cool"
export HELPER_HOME_MODEL_LEARNING_ENABLED="input_boolean.home_model_learning_enabled"
export HELPER_HOME_MODEL_RECOMMENDATION_COOLDOWN="input_number.home_model_recommendation_cooldown"
export HELPER_HOME_MODEL_STORAGE="input_text.home_model_storage"

log_info "Configuration loaded successfully"
log_info "Core Sensors:"
log_info "  - Indoor temperature sensor: ${INDOOR_TEMP_SENSOR}"
log_info "  - Outdoor temperature sensor: ${OUTDOOR_TEMP_SENSOR}"
log_info "  - Climate entity: ${CLIMATE_ENTITY}"
log_info "Optional Sensors:"
log_info "  - Indoor humidity sensor: ${INDOOR_HUMIDITY_SENSOR:-'not configured'}"
log_info "  - Outdoor humidity sensor: ${OUTDOOR_HUMIDITY_SENSOR:-'not configured'}"
log_info "Configuration:"
log_info "  - Learning enabled: ${LEARNING_ENABLED}"
log_info "  - ML training enabled: ${ML_TRAINING_ENABLED}"
log_info "  - Forecast hours: ${FORECAST_HOURS}"
log_info "  - Update interval: ${UPDATE_INTERVAL} minutes"
log_info "  - Debug mode: ${DEBUG_MODE}"
log_info "Storage:"
log_info "  - Data directory: ${DATA_DIR}"
log_info "  - Models directory: ${MODELS_DIR}"
if [[ "${DEBUG_MODE}" == "true" ]]; then
    log_info "Helper Entity Mappings (for dynamic reading):"
    env | grep "^HELPER_" | sort | while read -r line; do
        log_info "  - $line"
    done
fi

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