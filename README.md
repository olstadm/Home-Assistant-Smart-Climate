# Home Assistant Smart Climate Dashboard

## Overview

This repository contains an enhanced Home Assistant Smart Climate system with an improved dashboard that provides intelligent climate control recommendations and easy-to-understand visualizations.

## 🚀 Recent Enhancements

The dashboard has been completely redesigned to improve user experience and make climate intelligence more digestible. The enhanced Python script provides comprehensive sensors that power this intuitive interface.

### Key Features

- **Real-time Comfort Scoring**: Live comfort metrics combining temperature and humidity
- **Energy Efficiency Tracking**: Smart efficiency scoring based on HVAC usage and conditions  
- **Visual Comfort Band**: Interactive temperature chart with highlighted comfort zones
- **Proactive HVAC Timing**: Smart recommendations for when to start/stop heating and cooling
- **12-Hour Temperature Forecasting**: Predictive modeling for both idle and controlled scenarios

## 📊 Dashboard Features

### Enhanced User Interface
- **Prominent Status Display**: Current HVAC recommendations and comfort metrics at the top
- **Visual Temperature Forecast**: Interactive chart with comfort zone highlighting
- **Quick Access Insights**: Fast temperature checkpoints (+30min, +1hr, +3hr)
- **Smart Timing Alerts**: Proactive notifications about temperature limits
- **Organized Configuration**: Logical grouping of settings and model parameters

### Improved Information Architecture
1. **Status Overview** - Current recommendation and live comfort metrics
2. **Forecast Visualization** - Enhanced chart with comfort band highlighting
3. **Quick Insights** - Fast access to key timing information  
4. **Configuration** - Settings and model learning status

## 🧠 Smart Features

### Comfort Score (0-100%)
Combines temperature (70% weight) and humidity (30% weight) to provide real-time comfort assessment:
- Temperature comfort zone based on your heating/cooling limits
- Humidity comfort zone (30-60% RH optimal)
- Real-time scoring helps optimize your environment

### Energy Efficiency Score (0-100%)
Evaluates system efficiency based on:
- Temperature differential between indoor/outdoor
- Current HVAC action (idle, heating, cooling)
- Solar conditions for natural heating assistance
- Helps identify optimal operation times

### Predictive Intelligence
- **Time-to-Limit**: Predicts when temperature will reach comfort boundaries
- **Optimal Start Times**: Recommends when to begin heating/cooling
- **Energy-Aware Scheduling**: Considers weather patterns and solar gain
- **Accuracy Tracking**: Monitors and improves prediction quality over time

## 📁 Files

- `HomeWeatherPredicter (1).py` - Enhanced Python script with all the climate intelligence
- `dashboard.yaml` - Improved Home Assistant dashboard configuration
- `dashboard_improvements.md` - detailed explanation of UX/UI improvements
- `dashboard_preview_screenshot.png` - Visual preview of the enhanced dashboard
- `dashboard_preview.html` - Standalone preview of the dashboard layout

## 🔧 Installation

1. Copy `HomeWeatherPredicter (1).py` to your AppDaemon apps directory
2. Configure the input entities for your temperature sensors and climate control
3. Import `dashboard.yaml` into your Home Assistant dashboard
4. Requires `custom:apexcharts-card` for enhanced charting capabilities

## 📈 Benefits

- **Reduced Energy Costs**: Smart timing recommendations optimize HVAC usage
- **Improved Comfort**: Real-time scoring helps maintain optimal conditions
- **Proactive Planning**: Advance warnings about temperature changes
- **Better Understanding**: Clear visualization of your home's thermal behavior
- **Easy Monitoring**: Intuitive dashboard shows all key information at a glance

## 🎯 Comfort Band Visualization

The comfort band is now visualized multiple ways:
- **Chart Background**: Green-highlighted comfort zone (68-78°F default)
- **Limit Lines**: Clear cooling and heating boundaries
- **Real-time Status**: Live comfort score based on current conditions
- **Settings Panel**: Easy adjustment of comfort parameters

This enhanced system transforms complex climate data into actionable insights, helping you maintain optimal comfort while minimizing energy usage.