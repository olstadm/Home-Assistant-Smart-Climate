"""Home Assistant REST API Client for Smart Climate Add-on."""
import aiohttp
import asyncio
import json
import logging
from typing import Any, Dict, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class HomeAssistantClient:
    """Client for communicating with Home Assistant REST API."""
    
    def __init__(self, url: str, token: str):
        """Initialize the Home Assistant client.
        
        Args:
            url: Home Assistant URL
            token: Long-lived access token or supervisor token
        """
        self.url = url.rstrip('/')
        self.token = token
        self.session = None
        self._headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            headers=self._headers,
            timeout=aiohttp.ClientTimeout(total=10)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def get_state(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get the state of an entity.
        
        Args:
            entity_id: The entity ID to get state for
            
        Returns:
            Entity state dict or None if not found
        """
        try:
            url = f"{self.url}/api/states/{entity_id}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 404:
                    logger.warning(f"Entity {entity_id} not found")
                    return None
                else:
                    logger.error(f"Failed to get state for {entity_id}: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting state for {entity_id}: {e}")
            return None
    
    async def set_state(self, entity_id: str, state: Union[str, int, float], 
                       attributes: Optional[Dict[str, Any]] = None) -> bool:
        """Set the state of an entity.
        
        Args:
            entity_id: The entity ID to set state for
            state: The state value
            attributes: Optional attributes dict
            
        Returns:
            True if successful, False otherwise
        """
        try:
            url = f"{self.url}/api/states/{entity_id}"
            data = {
                'state': str(state),
                'attributes': attributes or {}
            }
            
            async with self.session.post(url, json=data) as response:
                if response.status in (200, 201):
                    return True
                else:
                    logger.error(f"Failed to set state for {entity_id}: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error setting state for {entity_id}: {e}")
            return False
    
    async def call_service(self, domain: str, service: str, entity_id: str = None, 
                          service_data: Dict[str, Any] = None) -> bool:
        """Call a Home Assistant service.
        
        Args:
            domain: Service domain
            service: Service name
            entity_id: Optional target entity
            service_data: Optional service data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            url = f"{self.url}/api/services/{domain}/{service}"
            data = service_data or {}
            
            if entity_id:
                data['entity_id'] = entity_id
                
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    return True
                else:
                    logger.error(f"Failed to call service {domain}.{service}: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error calling service {domain}.{service}: {e}")
            return False
    
    async def get_config(self) -> Optional[Dict[str, Any]]:
        """Get Home Assistant configuration.
        
        Returns:
            Configuration dict or None if failed
        """
        try:
            url = f"{self.url}/api/config"
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get config: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting config: {e}")
            return None
    
    def get_float_value(self, state_data: Optional[Dict[str, Any]], default: float = 0.0) -> float:
        """Extract float value from state data.
        
        Args:
            state_data: State data from get_state()
            default: Default value if extraction fails
            
        Returns:
            Float value or default
        """
        if not state_data:
            return default
            
        try:
            state = state_data.get('state', '')
            if state in ('unknown', 'unavailable', ''):
                return default
            return float(state)
        except (ValueError, TypeError):
            return default
    
    def get_string_value(self, state_data: Optional[Dict[str, Any]], default: str = '') -> str:
        """Extract string value from state data.
        
        Args:
            state_data: State data from get_state()
            default: Default value if extraction fails
            
        Returns:
            String value or default
        """
        if not state_data:
            return default
            
        state = state_data.get('state', '')
        if state in ('unknown', 'unavailable'):
            return default
        return str(state)