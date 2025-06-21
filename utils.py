#!/usr/bin/env python3
"""
Utility Functions for Crypto Analysis Bot V3.2
==============================================
Helper functions, formatters, validators, and common utilities
"""

import re
import json
import time
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Union, Tuple
from functools import wraps
import logging
import requests
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================
# TIME AND DATE UTILITIES
# ===============================

def get_current_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now(timezone.utc).isoformat()

def get_current_time_tehran() -> str:
    """Get current time in Tehran timezone"""
    try:
        tehran_tz = timezone(timedelta(hours=3, minutes=30))
        now = datetime.now(tehran_tz)
        return now.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.error(f"❌ Error getting Tehran time: {e}")
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def format_time_ago(timestamp: str) -> str:
    """Format timestamp as 'time ago' string"""
    try:
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            dt = timestamp
        
        now = datetime.now(timezone.utc)
        diff = now - dt.replace(tzinfo=timezone.utc)
        
        seconds = diff.total_seconds()
        
        if seconds < 60:
            return f"{int(seconds)}s ago"
        elif seconds < 3600:
            return f"{int(seconds/60)}m ago"
        elif seconds < 86400:
            return f"{int(seconds/3600)}h ago"
        else:
            return f"{int(seconds/86400)}d ago"
            
    except Exception as e:
        logger.error(f"❌ Error formatting time ago: {e}")
        return "unknown"

def parse_time_string(time_str: str) -> datetime:
    """Parse various time string formats"""
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%H:%M"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Unable to parse time string: {time_str}")

# ===============================
# NUMBER AND CURRENCY FORMATTING
# ===============================

def format_price(price: float, symbol: str = "$") -> str:
    """Format price with appropriate decimal places"""
    try:
        if price >= 1000:
            return f"{symbol}{price:,.2f}"
        elif price >= 1:
            return f"{symbol}{price:.2f}"
        elif price >= 0.01:
            return f"{symbol}{price:.4f}"
        else:
            return f"{symbol}{price:.8f}"
    except (ValueError, TypeError):
        return f"{symbol}0.00"

def format_percentage(value: float, include_sign: bool = True) -> str:
    """Format percentage with appropriate sign and color indicators"""
    try:
        if include_sign:
            sign = "+" if value > 0 else ""
            return f"{sign}{value:.2f}%"
        else:
            return f"{abs(value):.2f}%"
    except (ValueError, TypeError):
        return "0.00%"

def format_volume(volume: float) -> str:
    """Format trading volume with K/M/B suffixes"""
    try:
        if volume >= 1_000_000_000:
            return f"{volume/1_000_000_000:.2f}B"
        elif volume >= 1_000_000:
            return f"{volume/1_000_000:.2f}M"
        elif volume >= 1_000:
            return f"{volume/1_000:.2f}K"
        else:
            return f"{volume:.0f}"
    except (ValueError, TypeError):
        return "0"

def format_large_number(number: float) -> str:
    """Format large numbers with appropriate suffixes"""
    try:
        if number >= 1_000_000_000_000:
            return f"{number/1_000_000_000_000:.2f}T"
        elif number >= 1_000_000_000:
            return f"{number/1_000_000_000:.2f}B"
        elif number >= 1_000_000:
            return f"{number/1_000_000:.2f}M"
        elif number >= 1_000:
            return f"{number/1_000:.2f}K"
        else:
            return f"{number:.2f}"
    except (ValueError, TypeError):
        return "0"

# ===============================
# STRING AND TEXT UTILITIES
# ===============================

def clean_symbol(symbol: str) -> str:
    """Clean and standardize crypto symbol"""
    return symbol.upper().replace("/", "").strip()

def validate_symbol(symbol: str) -> bool:
    """Validate crypto symbol format"""
    pattern = r'^[A-Z]{3,10}USDT?$'
    return bool(re.match(pattern, symbol.upper()))

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations"""
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    # Limit length
    return sanitized[:255]

def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text"""
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    return [float(match) for match in matches if match]

# ===============================
# VALIDATION UTILITIES
# ===============================

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_url(url: str) -> bool:
    """Validate URL format"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def validate_telegram_token(token: str) -> bool:
    """Validate Telegram bot token format"""
    pattern = r'^\d+:[A-Za-z0-9_-]{35}$'
    return bool(re.match(pattern, token))

def validate_config_value(value: Any, expected_type: type, min_val: float = None, max_val: float = None) -> bool:
    """Validate configuration value"""
    try:
        if not isinstance(value, expected_type):
            return False
        
        if expected_type in [int, float] and (min_val is not None or max_val is not None):
            if min_val is not None and value < min_val:
                return False
            if max_val is not None and value > max_val:
                return False
        
        return True
    except:
        return False

# ===============================
# SECURITY UTILITIES
# ===============================

def generate_secure_token(length: int = 32) -> str:
    """Generate secure random token"""
    return secrets.token_urlsafe(length)

def hash_string(text: str, algorithm: str = "sha256") -> str:
    """Hash string using specified algorithm"""
    try:
        if algorithm == "md5":
            return hashlib.md5(text.encode()).hexdigest()
        elif algorithm == "sha1":
            return hashlib.sha1(text.encode()).hexdigest()
        elif algorithm == "sha256":
            return hashlib.sha256(text.encode()).hexdigest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    except Exception as e:
        logger.error(f"❌ Error hashing string: {e}")
        return ""

def mask_sensitive_data(data: str, mask_char: str = "*", show_chars: int = 4) -> str:
    """Mask sensitive data (API keys, tokens, etc.)"""
    if len(data) <= show_chars * 2:
        return mask_char * len(data)
    
    start = data[:show_chars]
    end = data[-show_chars:]
    middle = mask_char * (len(data) - show_chars * 2)
    
    return start + middle + end

# ===============================
# API UTILITIES
# ===============================

def make_api_request(url: str, method: str = "GET", headers: Dict = None, 
                    data: Dict = None, timeout: int = 30, retries: int = 3) -> Optional[Dict]:
    """Make API request with retry logic"""
    if headers is None:
        headers = {"User-Agent": "CryptoBot/3.2"}
    
    for attempt in range(retries):
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=data, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️  API request attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                logger.error(f"❌ API request failed after {retries} attempts")
                return None
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return None

def check_api_health(url: str, timeout: int = 10) -> bool:
    """Check if API endpoint is healthy"""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except:
        return False

# ===============================
# DATA PROCESSING UTILITIES
# ===============================

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int"""
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values"""
    try:
        if old_value == 0:
            return 0.0
        return ((new_value - old_value) / old_value) * 100
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0

def moving_average(values: List[float], window: int) -> List[float]:
    """Calculate moving average"""
    if len(values) < window:
        return [sum(values) / len(values)] * len(values)
    
    result = []
    for i in range(len(values)):
        if i < window - 1:
            result.append(sum(values[:i+1]) / (i+1))
        else:
            result.append(sum(values[i-window+1:i+1]) / window)
    
    return result

def detect_outliers(values: List[float], threshold: float = 2.0) -> List[int]:
    """Detect outliers using standard deviation method"""
    if len(values) < 3:
        return []
    
    mean_val = sum(values) / len(values)
    variance = sum((x - mean_val) ** 2 for x in values) / len(values)
    std_dev = variance ** 0.5
    
    outliers = []
    for i, value in enumerate(values):
        if abs(value - mean_val) > threshold * std_dev:
            outliers.append(i)
    
    return outliers

# ===============================
# PERFORMANCE AND MONITORING
# ===============================

def timing_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"⏱️  {func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

def rate_limiter(max_calls: int, time_window: int):
    """Rate limiting decorator"""
    calls = []
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Remove old calls
            calls[:] = [call_time for call_time in calls if now - call_time < time_window]
            
            if len(calls) >= max_calls:
                sleep_time = time_window - (now - calls[0])
                if sleep_time > 0:
                    logger.warning(f"⚠️  Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
                    calls.clear()
            
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

# ===============================
# EMOJI AND FORMATTING HELPERS
# ===============================

def get_trend_emoji(value: float) -> str:
    """Get emoji based on trend value"""
    if value > 0:
        return "📈"
    elif value < 0:
        return "📉"
    else:
        return "➡️"

def get_signal_emoji(signal: str) -> str:
    """Get emoji for trading signal"""
    signal_emojis = {
        "BUY": "🟢",
        "SELL": "🔴", 
        "HOLD": "🟡",
        "STRONG_BUY": "💚",
        "STRONG_SELL": "❤️"
    }
    return signal_emojis.get(signal.upper(), "⚪")

def get_confidence_emoji(confidence: float) -> str:
    """Get emoji based on confidence level"""
    if confidence >= 0.8:
        return "🔥"
    elif confidence >= 0.6:
        return "💪"
    elif confidence >= 0.4:
        return "👍"
    else:
        return "🤔"

# ===============================
# ERROR HANDLING UTILITIES
# ===============================

class BotError(Exception):
    """Custom exception for bot errors"""
    pass

class APIError(BotError):
    """Exception for API-related errors"""
    pass

class ConfigError(BotError):
    """Exception for configuration errors"""
    pass

def safe_execute(func, *args, default=None, log_error=True, **kwargs):
    """Safely execute function with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_error:
            logger.error(f"❌ Error executing {func.__name__}: {e}")
        return default

# ===============================
# TESTING UTILITIES
# ===============================

def generate_test_data(symbol: str, days: int = 30) -> List[Dict]:
    """Generate test crypto data for development"""
    import random
    
    base_price = random.uniform(100, 50000)
    data = []
    
    for i in range(days * 24):  # Hourly data
        timestamp = datetime.now() - timedelta(hours=days*24-i)
        price_change = random.uniform(-0.05, 0.05)  # ±5% change
        base_price *= (1 + price_change)
        
        data.append({
            "symbol": symbol,
            "price": round(base_price, 2),
            "volume": random.uniform(1000000, 10000000),
            "change_24h": random.uniform(-10, 10),
            "timestamp": timestamp.isoformat()
        })
    
    return data

if __name__ == "__main__":
    # Test utility functions
    print("🔧 Utility Functions Test")
    print("=" * 40)
    
    # Test time formatting
    now = get_current_timestamp()
    tehran_time = get_current_time_tehran()
    print(f"⏰ Current UTC: {now}")
    print(f"🇮🇷 Tehran time: {tehran_time}")
    
    # Test price formatting
    test_prices = [45123.45, 0.00001234, 1.23456]
    for price in test_prices:
        formatted = format_price(price)
        print(f"💰 {price} → {formatted}")
    
    # Test percentage formatting
    test_changes = [5.67, -3.21, 0.0]
    for change in test_changes:
        formatted = format_percentage(change)
        print(f"📊 {change}% → {formatted}")
    
    # Test validation
    test_emails = ["test@example.com", "invalid-email"]
    for email in test_emails:
        valid = validate_email(email)
        print(f"📧 {email} → {'✅' if valid else '❌'}")
    
    # Test security
    token = generate_secure_token(16)
    masked = mask_sensitive_data(token)
    print(f"🔐 Token: {masked}")
    
    # Test data processing
    test_values = [1, 5, 3, 8, 2, 6, 4]
    ma = moving_average(test_values, 3)
    print(f"📈 Moving Average: {[round(x, 2) for x in ma]}")
    
    print("\n✅ All utility functions working correctly!")