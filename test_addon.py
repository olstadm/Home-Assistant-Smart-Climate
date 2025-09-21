#!/usr/bin/env python3
"""Simple test script to validate Smart Climate Add-on configuration."""
import json
import os
import sys
from pathlib import Path


def test_addon_structure():
    """Test that all required add-on files exist."""
    required_files = [
        'config.json',
        'Dockerfile',
        'run.sh',
        'smart_climate/__init__.py',
        'smart_climate/main.py',
        'smart_climate/ha_client.py',
        'smart_climate/climate_forecaster.py',
        'smart_climate/ml_climate_enhancer.py',
        'requirements.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required add-on files present")
    return True


def test_config_json():
    """Test that config.json is valid."""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        required_keys = ['name', 'version', 'slug', 'description', 'arch', 'options', 'schema']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            print(f"❌ Missing required keys in config.json: {missing_keys}")
            return False
        
        # Test that schema matches options
        options_keys = set(config['options'].keys())
        schema_keys = set(config['schema'].keys())
        
        if options_keys != schema_keys:
            print(f"❌ Options and schema keys don't match")
            print(f"   Options only: {options_keys - schema_keys}")
            print(f"   Schema only: {schema_keys - options_keys}")
            return False
        
        print("✅ config.json is valid")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config.json: {e}")
        return False
    except FileNotFoundError:
        print("❌ config.json not found")
        return False


def test_dockerfile():
    """Test that Dockerfile is properly structured."""
    try:
        with open('Dockerfile', 'r') as f:
            content = f.read()
        
        required_commands = ['FROM', 'WORKDIR', 'COPY', 'RUN', 'CMD']
        missing_commands = []
        
        for cmd in required_commands:
            if cmd not in content:
                missing_commands.append(cmd)
        
        if missing_commands:
            print(f"❌ Missing commands in Dockerfile: {missing_commands}")
            return False
        
        # Check for Python installation
        if 'python3' not in content:
            print("❌ Dockerfile doesn't install Python")
            return False
        
        print("✅ Dockerfile is properly structured")
        return True
        
    except FileNotFoundError:
        print("❌ Dockerfile not found")
        return False


def test_run_script():
    """Test that run.sh is executable and properly structured."""
    run_path = Path('run.sh')
    
    if not run_path.exists():
        print("❌ run.sh not found")
        return False
    
    # Check if executable
    if not os.access(run_path, os.X_OK):
        print("❌ run.sh is not executable")
        return False
    
    # Check content
    with open(run_path, 'r') as f:
        content = f.read()
    
    required_elements = ['bashio::', 'python3', 'smart_climate.main']
    missing_elements = []
    
    for element in required_elements:
        if element not in content:
            missing_elements.append(element)
    
    if missing_elements:
        print(f"❌ Missing elements in run.sh: {missing_elements}")
        return False
    
    print("✅ run.sh is properly configured")
    return True


def test_python_imports():
    """Test that Python modules can be imported."""
    sys.path.insert(0, os.getcwd())
    
    try:
        from smart_climate import __version__
        print(f"✅ Smart Climate package imports successfully (v{__version__})")
    except ImportError as e:
        print(f"❌ Failed to import smart_climate package: {e}")
        return False
    
    try:
        from smart_climate.ha_client import HomeAssistantClient
        print("✅ HomeAssistantClient imports successfully")
    except ImportError as e:
        print(f"❌ Failed to import HomeAssistantClient: {e}")
        return False
    
    try:
        from smart_climate.ml_climate_enhancer import MLClimateEnhancer
        print("✅ MLClimateEnhancer imports successfully")
    except ImportError as e:
        print(f"❌ Failed to import MLClimateEnhancer: {e}")
        return False
    
    try:
        from smart_climate.climate_forecaster import ClimateForecaster
        print("✅ ClimateForecaster imports successfully")
    except ImportError as e:
        print(f"❌ Failed to import ClimateForecaster: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("🧪 Testing Smart Climate Add-on Configuration\n")
    
    tests = [
        test_addon_structure,
        test_config_json,
        test_dockerfile,
        test_run_script,
        test_python_imports
    ]
    
    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Add-on is ready for installation.")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())