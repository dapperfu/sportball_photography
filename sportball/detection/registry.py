"""
Detection Tool Registry

Central registry for managing detection tools and their configurations.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

from typing import Dict, List, Optional, Any, Type
from pathlib import Path
import json
from loguru import logger

from .base import DetectionTool, DetectionConfig, DetectionToolFactory


class DetectionRegistry:
    """
    Central registry for detection tools.
    
    This provides a unified interface for managing detection tools,
    their configurations, and tool-agnostic operations.
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize detection registry.
        
        Args:
            config_file: Optional configuration file path
        """
        self.config_file = config_file
        self.tool_configs: Dict[str, DetectionConfig] = {}
        self.tool_instances: Dict[str, DetectionTool] = {}
        
        # Load configuration if provided
        if config_file and config_file.exists():
            self.load_config(config_file)
    
    def register_tool(self, tool_name: str, tool_class: Type[DetectionTool], 
                     config: Optional[DetectionConfig] = None) -> None:
        """
        Register a detection tool with optional configuration.
        
        Args:
            tool_name: Name of the tool
            tool_class: DetectionTool subclass
            config: Optional configuration for the tool
        """
        # Register with factory
        DetectionToolFactory.register_tool(tool_name, tool_class)
        
        # Store configuration
        if config:
            config.tool_name = tool_name
            self.tool_configs[tool_name] = config
        else:
            # Create default configuration
            default_config = DetectionConfig(tool_name=tool_name)
            self.tool_configs[tool_name] = default_config
        
        logger.info(f"Registered detection tool: {tool_name}")
    
    def get_tool(self, tool_name: str, config_override: Optional[Dict[str, Any]] = None) -> Optional[DetectionTool]:
        """
        Get a detection tool instance.
        
        Args:
            tool_name: Name of the tool
            config_override: Optional configuration overrides
            
        Returns:
            DetectionTool instance or None if not found
        """
        # Check if we already have an instance
        if tool_name in self.tool_instances:
            tool = self.tool_instances[tool_name]
            
            # Apply configuration overrides if provided
            if config_override:
                tool.update_config(config_override)
            
            return tool
        
        # Create new instance
        config = self.tool_configs.get(tool_name)
        if config_override and config:
            # Create a copy of the config and apply overrides
            config_dict = config.to_dict()
            config_dict.update(config_override)
            new_config = DetectionConfig()
            new_config.update_from_dict(config_dict)
            config = new_config
        
        tool = DetectionToolFactory.create_tool(tool_name, config)
        if tool:
            self.tool_instances[tool_name] = tool
        
        return tool
    
    def list_tools(self) -> List[str]:
        """
        List all registered tools.
        
        Returns:
            List of tool names
        """
        return DetectionToolFactory.list_tools()
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool information or None if not found
        """
        return DetectionToolFactory.get_tool_info(tool_name)
    
    def update_tool_config(self, tool_name: str, config_updates: Dict[str, Any]) -> bool:
        """
        Update configuration for a tool.
        
        Args:
            tool_name: Name of the tool
            config_updates: Configuration updates
            
        Returns:
            True if successful, False otherwise
        """
        if tool_name not in self.tool_configs:
            logger.error(f"Tool not found: {tool_name}")
            return False
        
        # Update stored configuration
        self.tool_configs[tool_name].update_from_dict(config_updates)
        
        # Update existing instance if it exists
        if tool_name in self.tool_instances:
            self.tool_instances[tool_name].update_config(config_updates)
        
        logger.info(f"Updated configuration for tool: {tool_name}")
        return True
    
    def save_config(self, config_file: Optional[Path] = None) -> bool:
        """
        Save current configuration to file.
        
        Args:
            config_file: Optional file path (uses instance config_file if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        target_file = config_file or self.config_file
        if not target_file:
            logger.error("No configuration file specified")
            return False
        
        try:
            # Prepare configuration data
            config_data = {
                'tool_configs': {
                    name: config.to_dict() 
                    for name, config in self.tool_configs.items()
                },
                'metadata': {
                    'version': '1.0.0',
                    'created_at': __import__('datetime').datetime.now().isoformat()
                }
            }
            
            # Ensure directory exists
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write configuration
            with open(target_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Saved configuration to: {target_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def load_config(self, config_file: Path) -> bool:
        """
        Load configuration from file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Load tool configurations
            tool_configs = config_data.get('tool_configs', {})
            for tool_name, config_dict in tool_configs.items():
                config = DetectionConfig()
                config.update_from_dict(config_dict)
                self.tool_configs[tool_name] = config
            
            self.config_file = config_file
            logger.info(f"Loaded configuration from: {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def detect_with_tool(self, tool_name: str, image_path: Path, 
                        config_override: Optional[Dict[str, Any]] = None,
                        **kwargs) -> Optional[Any]:
        """
        Perform detection using a specific tool.
        
        Args:
            tool_name: Name of the tool to use
            image_path: Path to the image file
            config_override: Optional configuration overrides
            **kwargs: Additional detection parameters
            
        Returns:
            Detection result or None if failed
        """
        tool = self.get_tool(tool_name, config_override)
        if not tool:
            logger.error(f"Could not get tool: {tool_name}")
            return None
        
        try:
            return tool.detect(image_path, **kwargs)
        except Exception as e:
            logger.error(f"Detection failed with tool {tool_name}: {e}")
            return None
    
    def detect_batch_with_tool(self, tool_name: str, image_paths: List[Path],
                              config_override: Optional[Dict[str, Any]] = None,
                              **kwargs) -> Optional[Dict[str, Any]]:
        """
        Perform batch detection using a specific tool.
        
        Args:
            tool_name: Name of the tool to use
            image_paths: List of image file paths
            config_override: Optional configuration overrides
            **kwargs: Additional detection parameters
            
        Returns:
            Dictionary of detection results or None if failed
        """
        tool = self.get_tool(tool_name, config_override)
        if not tool:
            logger.error(f"Could not get tool: {tool_name}")
            return None
        
        try:
            return tool.detect_batch(image_paths, **kwargs)
        except Exception as e:
            logger.error(f"Batch detection failed with tool {tool_name}: {e}")
            return None
    
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available tools.
        
        Returns:
            Dictionary mapping tool names to their information
        """
        tools_info = {}
        for tool_name in self.list_tools():
            tool_info = self.get_tool_info(tool_name)
            if tool_info:
                tools_info[tool_name] = tool_info
        return tools_info
    
    def clear_cache(self) -> None:
        """Clear all tool instances from cache."""
        self.tool_instances.clear()
        logger.info("Cleared tool instance cache")
    
    def reload_tool(self, tool_name: str) -> bool:
        """
        Reload a specific tool instance.
        
        Args:
            tool_name: Name of the tool to reload
            
        Returns:
            True if successful, False otherwise
        """
        if tool_name in self.tool_instances:
            del self.tool_instances[tool_name]
            logger.info(f"Reloaded tool: {tool_name}")
            return True
        else:
            logger.warning(f"Tool not in cache: {tool_name}")
            return False
