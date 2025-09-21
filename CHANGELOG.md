# Changelog

All notable changes to the Smart Climate Forecasting project will be documented in this file.

## [1.0.0] - 2024-09-21

### Added - Home Assistant Add-on Migration

#### 🎉 Major Release: Home Assistant Add-on
- **Complete migration from AppDaemon to standalone Home Assistant Add-on**
- **Debian-slim Docker container** for easy installation and maintenance
- **Zero-dependency installation** through Home Assistant Add-on Store
- **Web-based configuration** through Home Assistant UI

#### 🚀 New Features
- **Home Assistant Add-on structure** with `config.json`, `Dockerfile`, and `run.sh`
- **REST API integration** replacing AppDaemon dependency
- **Persistent ML model storage** in `/data/models` volume
- **Health check endpoint** for container monitoring
- **Configuration validation** with comprehensive error handling
- **Automatic service discovery** through Home Assistant API

#### 🔧 Technical Improvements
- **Async/await architecture** using `aiohttp` for better performance
- **Structured logging** with configurable log levels
- **Graceful shutdown handling** with proper signal management
- **Resource optimization** for container environments
- **Error resilience** with automatic retry mechanisms

#### 📊 Enhanced ML Pipeline
- **Persistent model storage** across container restarts
- **Ensemble model weighting** based on performance metrics
- **Feature engineering improvements** with time-based features
- **Training data management** with memory-efficient storage
- **Model validation** and performance monitoring

#### 🎛️ Configuration Management
- **Schema validation** for add-on options
- **Environment variable injection** from Home Assistant
- **Runtime configuration updates** without restart
- **Default value handling** for optional parameters
- **Configuration persistence** across updates

#### 🔌 Integration Improvements
- **Direct Home Assistant API** communication
- **Sensor entity management** with proper attributes
- **Service call capabilities** for HVAC control
- **State persistence** and recovery
- **Configuration validation** against Home Assistant entities

#### 📈 Monitoring and Observability
- **Health check endpoint** on port 8099
- **Structured sensor attributes** for better dashboard integration
- **Performance metrics** tracking
- **Training status indicators** 
- **Error reporting** through Home Assistant logs

### Changed
- **Architecture**: Migrated from AppDaemon app to standalone Docker service
- **Dependencies**: Removed AppDaemon, added aiohttp and container utilities
- **Configuration**: Moved from YAML files to Home Assistant add-on options
- **Entity naming**: Changed from `home_model_*` to `smart_climate_*` prefix
- **Installation**: Now available through Home Assistant Add-on Store
- **Documentation**: Updated for add-on installation and configuration

### Removed
- **AppDaemon dependency** - No longer required
- **Manual YAML configuration** - Replaced with UI configuration
- **Complex installation process** - Simplified to add-on installation
- **Entity helpers requirement** - Direct API integration instead

### Migration Guide
For users migrating from the AppDaemon version:

1. **Stop AppDaemon app**: Disable the old `home_forecast.py` app
2. **Install add-on**: Install Smart Climate Forecasting from add-on store
3. **Configure entities**: Set sensor and climate entities in add-on options
4. **Update dashboard**: Change entity names from `home_model_*` to `smart_climate_*`
5. **Verify operation**: Check new sensors are created and updating
6. **Remove old files**: Clean up AppDaemon apps directory

### Technical Details

#### Container Specifications
- **Base Image**: `ghcr.io/hassio-addons/debian-base:7.2.0`
- **Python Version**: 3.11
- **Architecture Support**: armhf, armv7, aarch64, amd64, i386
- **Memory Requirements**: ~500MB (including ML models)
- **Storage Requirements**: ~50MB for persistent data

#### API Endpoints
- **Health Check**: `GET /health` (port 8099)
- **Home Assistant API**: Full REST API integration
- **Supervisor API**: Add-on management and configuration

#### Generated Entities
All entities now use `smart_climate_` prefix:
- `sensor.smart_climate_indoor_forecast`
- `sensor.smart_climate_ml_enhanced_prediction`
- `sensor.smart_climate_comfort_violation`
- `sensor.smart_climate_comfort_score`
- `sensor.smart_climate_efficiency_score`
- `sensor.smart_climate_ml_status`
- `sensor.smart_climate_tau_hours`
- `sensor.smart_climate_k_heat`
- `sensor.smart_climate_k_cool`

## [0.9.0] - Previous AppDaemon Version

### Features (Legacy)
- AppDaemon-based implementation
- RC thermal modeling with ML enhancement
- XGBoost, LightGBM, Random Forest ensemble
- AccuWeather integration
- Comfort band validation
- Real-time parameter learning

### Note
This version is now deprecated in favor of the Home Assistant Add-on implementation.