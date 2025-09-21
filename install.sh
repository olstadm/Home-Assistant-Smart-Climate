#!/bin/bash
# Smart Climate Add-on Installation Script
# This script helps install the Smart Climate Add-on to Home Assistant

set -e

ADDON_NAME="smart_climate"
ADDON_TITLE="Smart Climate Forecasting"

echo "🏠 Smart Climate Add-on Installation Helper"
echo "=========================================="

# Function to detect Home Assistant installation
detect_ha_installation() {
    if [ -d "/usr/share/hassio" ] || [ -d "/data/supervisor" ]; then
        echo "✅ Home Assistant OS/Supervised detected"
        return 0
    elif [ -d "/config" ] && [ -f "/config/configuration.yaml" ]; then
        echo "✅ Home Assistant Container detected"
        return 0
    else
        echo "❌ Home Assistant installation not detected"
        echo "   Please run this script on your Home Assistant system"
        return 1
    fi
}

# Function to find add-on directory
find_addon_directory() {
    local directories=(
        "/addon_configs/local"
        "/config/addons"
        "/usr/share/hassio/addons/local"
        "/data/addons/local"
    )
    
    for dir in "${directories[@]}"; do
        if [ -d "$dir" ]; then
            echo "$dir"
            return 0
        fi
    done
    
    return 1
}

# Function to install add-on
install_addon() {
    local addon_dir="$1/$ADDON_NAME"
    
    echo "📁 Creating add-on directory: $addon_dir"
    mkdir -p "$addon_dir"
    
    echo "📋 Copying add-on files..."
    
    # Copy essential files
    local files=(
        "config.json"
        "Dockerfile" 
        "run.sh"
        "build.yaml"
        "README.md"
        "CHANGELOG.md"
        "requirements.txt"
    )
    
    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            cp "$file" "$addon_dir/"
            echo "   ✅ $file"
        else
            echo "   ❌ $file (missing)"
        fi
    done
    
    # Copy smart_climate package
    if [ -d "smart_climate" ]; then
        cp -r "smart_climate" "$addon_dir/"
        echo "   ✅ smart_climate/ package"
    else
        echo "   ❌ smart_climate/ package (missing)"
        return 1
    fi
    
    # Set permissions
    chmod +x "$addon_dir/run.sh"
    
    echo "✅ Add-on files installed successfully"
}

# Function to validate installation
validate_installation() {
    local addon_dir="$1/$ADDON_NAME"
    
    echo "🔍 Validating installation..."
    
    # Check required files
    local required_files=(
        "config.json"
        "Dockerfile"
        "run.sh"
        "smart_climate/main.py"
    )
    
    for file in "${required_files[@]}"; do
        if [ -f "$addon_dir/$file" ]; then
            echo "   ✅ $file"
        else
            echo "   ❌ $file (missing)"
            return 1
        fi
    done
    
    echo "✅ Installation validation passed"
}

# Main installation process
main() {
    echo "🚀 Starting installation process..."
    
    # Check if we're in the right directory
    if [ ! -f "config.json" ] || [ ! -d "smart_climate" ]; then
        echo "❌ Error: Please run this script from the Smart Climate repository directory"
        echo "   Expected files: config.json, smart_climate/"
        exit 1
    fi
    
    # Detect Home Assistant
    if ! detect_ha_installation; then
        echo ""
        echo "💡 Manual Installation Instructions:"
        echo "   1. Copy all add-on files to: /addon_configs/local/$ADDON_NAME/"
        echo "   2. Restart Home Assistant"
        echo "   3. Go to Supervisor → Add-on Store → Local Add-ons"
        echo "   4. Install '$ADDON_TITLE'"
        exit 1
    fi
    
    # Find add-on directory
    echo "🔍 Looking for add-on directory..."
    if addon_base_dir=$(find_addon_directory); then
        echo "✅ Found: $addon_base_dir"
    else
        echo "❌ Could not find local add-on directory"
        echo "   Please create: /addon_configs/local/ or /config/addons/"
        exit 1
    fi
    
    # Install add-on
    if install_addon "$addon_base_dir"; then
        echo "✅ Installation completed"
    else
        echo "❌ Installation failed"
        exit 1
    fi
    
    # Validate installation
    if validate_installation "$addon_base_dir"; then
        echo "✅ Validation completed"
    else
        echo "❌ Validation failed"
        exit 1
    fi
    
    echo ""
    echo "🎉 Smart Climate Add-on installed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Restart Home Assistant"
    echo "   2. Go to Supervisor → Add-on Store"
    echo "   3. Find '$ADDON_TITLE' in Local Add-ons"
    echo "   4. Click Install and configure your sensors"
    echo "   5. Start the add-on"
    echo ""
    echo "📖 For configuration help, see: README.md"
}

# Run main function
main "$@"