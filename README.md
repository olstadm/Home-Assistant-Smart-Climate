# Smart Climate Forecasting Add-on

Advanced climate prediction and control using RC (Resistor-Capacitor) thermal modeling enhanced with machine learning for Home Assistant.

## Features

- **RC Thermal Modeling**: Physics-based temperature prediction using building thermal dynamics
- **Machine Learning Enhancement**: XGBoost, LightGBM, and Random Forest models improve predictions
- **Real-time Learning**: Model parameters adapt automatically to your home's characteristics
- **Comfort Band Validation**: Detects and alerts on potential comfort violations
- **AccuWeather Integration**: Optional weather forecast integration for improved accuracy
- **Energy Efficiency Scoring**: Track and optimize HVAC efficiency

## Installation

### Home Assistant Add-on Store (Recommended)

1. Navigate to **Supervisor** → **Add-on Store** in Home Assistant
2. Click the menu (⋮) → **Repositories**
3. Add repository URL: `https://github.com/olstadm/Home-Assistant-Smart-Climate`
4. Find "Smart Climate Forecasting" in the add-on store
5. Click **Install**

### Manual Installation

1. Copy the add-on files to your Home Assistant add-ons directory:
   ```
   /addon_configs/local/smart_climate/
   ```
2. Restart Home Assistant
3. Navigate to **Supervisor** → **Add-on Store** → **Local Add-ons**
4. Install "Smart Climate Forecasting"

## Configuration

### Entity Selection via UI

The add-on now supports **UI-based entity selection** - no need to manually type entity IDs! When configuring the add-on:

1. Go to **Supervisor** → **Add-on Store** → **Smart Climate Forecasting** → **Configuration**
2. Use the dropdown menus to select your entities:
   - **Indoor Temperature Sensor**: Select from available temperature sensors
   - **Outdoor Temperature Sensor**: Select from available temperature sensors  
   - **Climate Entity**: Select from available thermostats/climate devices
   - **Indoor/Outdoor Humidity Sensors**: Optional - select from humidity sensors

### Configuration Options

All configuration is now done through the **Home Assistant UI** with intelligent entity selectors that only show relevant entities for each field.

### Available Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `indoor_temperature_sensor` | entity selector | **Required** | Indoor temperature sensor (dropdown selection) |
| `outdoor_temperature_sensor` | entity selector | **Required** | Outdoor temperature sensor (dropdown selection) |
| `climate_entity` | entity selector | **Required** | Climate/thermostat entity (dropdown selection) |
| `indoor_humidity_sensor` | entity selector | optional | Indoor humidity sensor (dropdown selection) |
| `outdoor_humidity_sensor` | entity selector | optional | Outdoor humidity sensor (dropdown selection) |
| `learning_enabled` | bool | `true` | Enable adaptive RC model learning |
| `forecast_hours` | int | `12` | Number of hours to forecast (1-24) |
| `update_interval_minutes` | int | `5` | Update interval in minutes (1-60) |
| `comfort_max_temp` | float | `80.0` | Maximum comfort temperature (°F) |
| `comfort_min_temp` | float | `62.0` | Minimum comfort temperature (°F) |
| `accuweather_api_key` | string | optional | AccuWeather API key |
| `accuweather_location_key` | string | optional | AccuWeather location key |
| `ml_training_enabled` | bool | `true` | Enable ML model training |
| `ml_training_interval_hours` | int | `1` | ML model retraining interval (1-24 hours) |
| `debug` | bool | `false` | Enable debug logging and error diagnostics |

### Advanced RC Model Parameters

These parameters allow fine-tuning of the thermal RC model for your specific home:

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `tau_hours` | float | `2.0` | 0.5-72.0 | Thermal time constant (hours) - how quickly your home responds to temperature changes |
| `forgetting_factor` | float | `0.99` | 0.8-1.0 | Learning rate for model adaptation (closer to 1.0 = slower adaptation) |
| `bias` | float | `0.0` | -0.2-0.2 | Temperature bias correction (°F) |
| `comfort_cap` | float | `80.0` | 70.0-95.0 | Upper comfort temperature limit (°F) |
| `heat_min_f` | float | `62.0` | 50.0-80.0 | Minimum heating temperature (°F) |
| `k_heat` | float | `0.5` | 0.0-2.0 | Heat gain coefficient - how much heating affects temperature |
| `k_cool` | float | `-0.5` | -2.0-0.0 | Cooling gain coefficient - how much cooling affects temperature |
| `recommendation_cooldown` | float | `30.0` | 1.0-120.0 | Minimum time between recommendations (minutes) |

### Configuration Benefits

- **Complete Standalone Operation**: No dependencies on Home Assistant helper entities
- **UI-Based Configuration**: All parameters configurable through the Home Assistant add-on interface
- **Real-time Updates**: Parameter changes take effect without restarting the add-on
- **Intelligent Validation**: Parameter bounds checking prevents invalid configurations
- **Advanced Tuning**: Fine-tune the RC model for your home's specific thermal characteristics

## Generated Sensors

The add-on creates the following sensors in Home Assistant:

### Primary Sensors
- `sensor.smart_climate_indoor_forecast` - Main forecast with idle and controlled predictions
- `sensor.smart_climate_ml_enhanced_prediction` - ML-enhanced temperature prediction
- `sensor.smart_climate_comfort_violation` - Comfort band violation alerts
- `sensor.smart_climate_comfort_score` - Current comfort score (0-100%)
- `sensor.smart_climate_efficiency_score` - Energy efficiency score (0-100%)

### ML Status Sensors
- `sensor.smart_climate_ml_status` - ML training status and metrics

### RC Model Parameters
- `sensor.smart_climate_tau_hours` - Building time constant (hours)
- `sensor.smart_climate_k_heat` - Heating coefficient
- `sensor.smart_climate_k_cool` - Cooling coefficient
- `sensor.smart_climate_bias` - Model temperature bias (°F)
- `sensor.smart_climate_forgetting_factor` - Learning rate factor
- `sensor.smart_climate_comfort_cap` - Comfort cap temperature (°F)
- `sensor.smart_climate_heat_min_f` - Heating minimum temperature (°F)
- `sensor.smart_climate_recommendation_cooldown` - Recommendation cooldown (minutes)

## Dashboard Integration

Import the provided dashboard configuration (`addon_dashboard.yaml`) to visualize:

1. 13-hour temperature forecast with ML enhancement
2. Real-time comfort and efficiency scores
3. ML training status and model parameters
4. Comfort violation alerts

## How It Works

### RC Thermal Model

The system models your home as an RC circuit where:
- **R** (Resistance): Thermal resistance of building envelope
- **C** (Capacitance): Thermal mass of building materials
- **τ** (Tau): Time constant = R × C

The model equation:
```
dT/dt = a*(T_out - T_in) + k_H*I_H + k_C*I_C + b + k_E*(h_out - h_in) + k_S*Solar
```

### Machine Learning Enhancement

Four ML models work together:
1. **Random Forest** - Robust ensemble method
2. **XGBoost** - Gradient boosting with regularization
3. **LightGBM** - Fast gradient boosting
4. **Gradient Boosting** - Classical boosting approach

The final prediction blends 70% RC model + 30% ML ensemble for stability.

### Adaptive Learning

- RC parameters update automatically using recursive least squares
- ML models retrain hourly with new data
- Model performance continuously monitored and weighted

## AccuWeather Integration

To use AccuWeather weather forecasts:

1. Sign up for a free AccuWeather API account
2. Get your API key and location key
3. Configure in add-on options
4. Weather data will enhance predictions automatically

## Troubleshooting

### Add-on Won't Start
- Check Home Assistant logs for detailed error messages
- Verify required sensor entities exist and are providing data
- Ensure adequate system resources (ML models need ~500MB RAM)

### No ML Enhancement
- Check `sensor.smart_climate_ml_status` for training status
- ML requires 50+ data samples before training
- Verify Python ML packages installed correctly

### Poor Predictions
- Allow 24-48 hours for RC parameters to converge
- Ensure sensors are accurate and positioned correctly
- Check for HVAC system issues affecting thermal dynamics

### Dashboard Issues
- Install ApexCharts custom card: `https://github.com/RomRider/apexcharts-card`
- Verify all sensor entities are available
- Check for YAML syntax errors in dashboard configuration

## Performance

- **Memory Usage**: ~500MB (including ML models)
- **CPU Usage**: Low (< 5% on typical systems)
- **Training Time**: < 30 seconds for ML models
- **Prediction Time**: < 100ms
- **Storage**: ~50MB for persistent model data

## Migration from AppDaemon

If migrating from the AppDaemon version:

1. Stop the AppDaemon app
2. Install this add-on
3. Configure with same sensor entities
4. The add-on will create new sensor entities with `smart_climate_` prefix
5. Update dashboard to use new sensor entities
6. ML models will retrain automatically with historical data

## Troubleshooting

### Container Startup Issues

If the add-on fails to start with errors related to configuration or logging:

1. **Enable Debug Mode**: Set the `debug` option to `true` in the add-on configuration to get detailed startup logs
2. **Check Logs**: Navigate to **Supervisor** → **Add-on Store** → **Smart Climate Forecasting** → **Log** tab
3. **Verify Configuration**: Ensure all required sensors and climate entities are properly configured and available
4. **Supervisor Token**: The add-on requires Home Assistant supervisor API access - this should be automatic in most installations

### Common Issues

- **"Configuration file not found"**: Restart Home Assistant and the add-on
- **"No supervisor token available"**: Check that you're running the add-on in a proper Home Assistant environment
- **"Required configuration missing"**: Verify that indoor/outdoor temperature sensors and climate entity are set
- **Python import errors**: The add-on will automatically install required dependencies on first start

### Debug Information

When debug mode is enabled, the add-on will provide detailed information about:
- Configuration loading process
- Home Assistant API connectivity
- Sensor entity validation
- ML model initialization
- Error diagnostics

## Support

- **Issues**: [GitHub Issues](https://github.com/olstadm/Home-Assistant-Smart-Climate/issues)
- **Discussion**: [Home Assistant Community Forum](https://community.home-assistant.io/)
- **Documentation**: [Full Documentation](https://github.com/olstadm/Home-Assistant-Smart-Climate/blob/main/ML_INTEGRATION.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read the contributing guidelines and submit pull requests to the main repository.

---

**Note**: This add-on replaces the previous AppDaemon-based implementation and provides the same functionality with easier installation and configuration through the Home Assistant UI.