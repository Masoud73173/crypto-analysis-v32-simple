#!/usr/bin/env python3
"""
Configuration Management for Crypto Analysis Bot V3.2
====================================================
Advanced configuration handling with environment variables, validation, and defaults
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TelegramConfig:
    """Telegram configuration settings"""
    bot_token: str = ""
    chat_id: str = ""
    enabled: bool = True
    message_format: str = "markdown"
    max_message_length: int = 4096
    
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

@dataclass
class ExchangeConfig:
    """Exchange API configuration"""
    name: str = "binance"
    api_key: str = ""
    secret_key: str = ""
    sandbox: bool = False
    rate_limit: int = 100
    timeout: int = 30
    
    def is_configured(self) -> bool:
        return bool(self.api_key and self.secret_key)

@dataclass
class AnalysisConfig:
    """Analysis configuration settings"""
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"])
    interval: int = 7200  # 2 hours in seconds
    daily_summary_time: str = "10:00"
    timezone: str = "Asia/Tehran"
    
    # Signal thresholds
    strong_signal_threshold: float = 0.7
    volume_spike_threshold: float = 2.0
    price_change_threshold: float = 5.0
    
    # Technical analysis settings
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    bb_period: int = 20
    bb_std_dev: float = 2.0
    
    # Risk management
    max_signals_per_day: int = 10
    cooldown_period: int = 3600  # 1 hour
    
    def validate(self) -> bool:
        """Validate analysis configuration"""
        if not self.symbols:
            logger.error("❌ No symbols configured for analysis")
            return False
        
        if self.interval < 60:
            logger.error("❌ Analysis interval too short (minimum 60 seconds)")
            return False
        
        if not (0.0 <= self.strong_signal_threshold <= 1.0):
            logger.error("❌ Signal threshold must be between 0.0 and 1.0")
            return False
        
        return True

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    url: str = "sqlite:///crypto_bot.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10
    cleanup_days: int = 90
    
    def get_connection_string(self) -> str:
        return self.url

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    secret_key: str = ""
    encryption_key: str = ""
    api_rate_limit: int = 100
    request_timeout: int = 30
    max_failed_attempts: int = 5
    
    def generate_secret_key(self) -> str:
        """Generate a secure secret key"""
        import secrets
        return secrets.token_urlsafe(32)

@dataclass
class MLConfig:
    """Machine Learning configuration"""
    enabled: bool = False
    model_path: str = "models/"
    retrain_interval: int = 86400  # 24 hours
    prediction_window: int = 24  # hours
    confidence_threshold: float = 0.8
    
    # Model parameters
    feature_window: int = 50  # number of historical points
    prediction_horizon: int = 6  # hours ahead
    
    def is_enabled(self) -> bool:
        return self.enabled and os.path.exists(self.model_path)

@dataclass
class NotificationConfig:
    """Notification configuration"""
    email_enabled: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_user: str = ""
    email_password: str = ""
    email_to: str = ""
    
    discord_enabled: bool = False
    discord_webhook_url: str = ""
    
    def email_configured(self) -> bool:
        return bool(self.email_enabled and self.email_user and self.email_password)
    
    def discord_configured(self) -> bool:
        return bool(self.discord_enabled and self.discord_webhook_url)

class CryptoBotConfig:
    """Main configuration class for Crypto Analysis Bot"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.telegram = TelegramConfig()
        self.exchanges = {"binance": ExchangeConfig()}
        self.analysis = AnalysisConfig()
        self.database = DatabaseConfig()
        self.security = SecurityConfig()
        self.ml = MLConfig()
        self.notifications = NotificationConfig()
        
        # Bot metadata
        self.bot_name = "Crypto Analysis Bot"
        self.version = "3.2.0"
        self.environment = "production"
        self.debug = False
        
        # Load configuration
        self.load_config()
        self.validate_config()
    
    def load_config(self):
        """Load configuration from environment variables and files"""
        try:
            # Load from environment variables
            self._load_from_env()
            
            # Load from config file if provided
            if self.config_file and os.path.exists(self.config_file):
                self._load_from_file()
            
            logger.info("✅ Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading configuration: {e}")
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # Telegram configuration
        self.telegram.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        
        # Binance configuration
        binance_config = self.exchanges["binance"]
        binance_config.api_key = os.getenv("BINANCE_API_KEY", "")
        binance_config.secret_key = os.getenv("BINANCE_SECRET_KEY", "")
        
        # Analysis configuration
        self.analysis.interval = int(os.getenv("ANALYSIS_INTERVAL", "7200"))
        self.analysis.daily_summary_time = os.getenv("DAILY_SUMMARY_TIME", "10:00")
        self.analysis.timezone = os.getenv("TIMEZONE", "Asia/Tehran")
        
        # Signal thresholds
        self.analysis.strong_signal_threshold = float(os.getenv("STRONG_SIGNAL_THRESHOLD", "0.7"))
        self.analysis.volume_spike_threshold = float(os.getenv("VOLUME_SPIKE_THRESHOLD", "2.0"))
        self.analysis.price_change_threshold = float(os.getenv("PRICE_CHANGE_THRESHOLD", "5.0"))
        
        # Database configuration
        self.database.url = os.getenv("DATABASE_URL", "sqlite:///crypto_bot.db")
        
        # Security configuration
        self.security.secret_key = os.getenv("SECRET_KEY", "")
        self.security.encryption_key = os.getenv("ENCRYPTION_KEY", "")
        self.security.api_rate_limit = int(os.getenv("API_RATE_LIMIT", "100"))
        
        # ML configuration
        self.ml.enabled = os.getenv("ENABLE_ML_PREDICTIONS", "false").lower() == "true"
        self.ml.model_path = os.getenv("ML_MODEL_PATH", "models/")
        
        # Bot settings
        self.environment = os.getenv("BOT_MODE", "production")
        self.debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
        
        # Notification configuration
        self.notifications.email_enabled = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
        self.notifications.email_user = os.getenv("EMAIL_USER", "")
        self.notifications.email_password = os.getenv("EMAIL_PASSWORD", "")
        
        # Symbols configuration
        symbols_env = os.getenv("CRYPTO_SYMBOLS", "")
        if symbols_env:
            self.analysis.symbols = [s.strip() for s in symbols_env.split(",")]
    
    def _load_from_file(self):
        """Load configuration from YAML or JSON file"""
        try:
            with open(self.config_file, 'r') as file:
                if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                    config_data = yaml.safe_load(file)
                else:
                    config_data = json.load(file)
                
                # Apply configuration from file
                self._apply_config_data(config_data)
                
        except Exception as e:
            logger.error(f"❌ Error loading config file {self.config_file}: {e}")
    
    def _apply_config_data(self, config_data: Dict[str, Any]):
        """Apply configuration data from file"""
        # Update telegram config
        if 'telegram' in config_data:
            telegram_data = config_data['telegram']
            for key, value in telegram_data.items():
                if hasattr(self.telegram, key):
                    setattr(self.telegram, key, value)
        
        # Update analysis config
        if 'analysis' in config_data:
            analysis_data = config_data['analysis']
            for key, value in analysis_data.items():
                if hasattr(self.analysis, key):
                    setattr(self.analysis, key, value)
        
        # Update other configs similarly...
    
    def validate_config(self) -> bool:
        """Validate all configuration settings"""
        try:
            is_valid = True
            
            # Validate Telegram configuration
            if not self.telegram.is_configured():
                logger.warning("⚠️  Telegram not configured - bot will run without notifications")
            
            # Validate analysis configuration
            if not self.analysis.validate():
                is_valid = False
            
            # Generate secret key if missing
            if not self.security.secret_key:
                self.security.secret_key = self.security.generate_secret_key()
                logger.info("🔑 Generated new secret key")
            
            # Validate symbols
            if not self.analysis.symbols:
                logger.error("❌ No crypto symbols configured for analysis")
                is_valid = False
            
            if is_valid:
                logger.info("✅ Configuration validation passed")
            else:
                logger.error("❌ Configuration validation failed")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"❌ Error validating configuration: {e}")
            return False
    
    def save_config(self, output_file: str):
        """Save current configuration to file"""
        try:
            config_data = {
                'telegram': {
                    'bot_token': self.telegram.bot_token,
                    'chat_id': self.telegram.chat_id,
                    'enabled': self.telegram.enabled
                },
                'analysis': {
                    'symbols': self.analysis.symbols,
                    'interval': self.analysis.interval,
                    'daily_summary_time': self.analysis.daily_summary_time,
                    'strong_signal_threshold': self.analysis.strong_signal_threshold,
                    'volume_spike_threshold': self.analysis.volume_spike_threshold,
                    'price_change_threshold': self.analysis.price_change_threshold
                },
                'database': {
                    'url': self.database.url,
                    'cleanup_days': self.database.cleanup_days
                },
                'ml': {
                    'enabled': self.ml.enabled,
                    'model_path': self.ml.model_path,
                    'confidence_threshold': self.ml.confidence_threshold
                }
            }
            
            with open(output_file, 'w') as file:
                if output_file.endswith('.yaml') or output_file.endswith('.yml'):
                    yaml.dump(config_data, file, default_flow_style=False)
                else:
                    json.dump(config_data, file, indent=2)
            
            logger.info(f"💾 Configuration saved to {output_file}")
            
        except Exception as e:
            logger.error(f"❌ Error saving configuration: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            'bot_name': self.bot_name,
            'version': self.version,
            'environment': self.environment,
            'telegram_configured': self.telegram.is_configured(),
            'binance_configured': self.exchanges['binance'].is_configured(),
            'symbols_count': len(self.analysis.symbols),
            'analysis_interval': f"{self.analysis.interval // 3600}h {(self.analysis.interval % 3600) // 60}m",
            'daily_summary_time': self.analysis.daily_summary_time,
            'ml_enabled': self.ml.is_enabled(),
            'database_type': self.database.url.split(':')[0]
        }
    
    def __str__(self) -> str:
        """String representation of configuration"""
        summary = self.get_summary()
        return f"""
🤖 {summary['bot_name']} v{summary['version']}
Environment: {summary['environment']}
Telegram: {'✅' if summary['telegram_configured'] else '❌'}
Binance: {'✅' if summary['binance_configured'] else '❌'}
Symbols: {summary['symbols_count']} pairs
Analysis: Every {summary['analysis_interval']}
Daily Summary: {summary['daily_summary_time']}
ML: {'✅' if summary['ml_enabled'] else '❌'}
Database: {summary['database_type']}
        """.strip()

# Global configuration instance
config = None

def get_config(config_file: Optional[str] = None) -> CryptoBotConfig:
    """Get global configuration instance"""
    global config
    if config is None:
        config = CryptoBotConfig(config_file)
    return config

def reload_config(config_file: Optional[str] = None):
    """Reload configuration"""
    global config
    config = CryptoBotConfig(config_file)
    return config

if __name__ == "__main__":
    # Test the configuration system
    print("⚙️  Configuration Module Test")
    print("=" * 40)
    
    # Create test configuration
    config = CryptoBotConfig()
    
    # Display configuration summary
    print(config)
    
    # Test validation
    is_valid = config.validate_config()
    print(f"\n✅ Configuration valid: {is_valid}")
    
    # Test saving configuration
    config.save_config("test_config.json")
    print("✅ Configuration saved")
    
    # Clean up
    if os.path.exists("test_config.json"):
        os.remove("test_config.json")
    
    print("\n✅ Configuration module working correctly!")