#!/usr/bin/env python3
"""
Crypto Analysis Bot V3.2 - Enterprise Edition (Fixed)
====================================================
Safe integration of all 11 modules with graceful fallbacks and error handling
Telegram Integration + Database + ML + Advanced TA + Professional Configuration
"""

import os
import json
import logging
import time
import functools
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from typing import Dict, List, Any, Optional

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Safe timing decorator
def safe_timing_decorator(func):
    """Safe timing decorator that always works"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"⏱️  {func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

# Safe imports with fallbacks
try:
    from technical_analysis import TechnicalAnalysis, analyze_crypto_signals
    TA_AVAILABLE = True
    logger.info("✅ Technical Analysis module loaded")
except ImportError as e:
    TA_AVAILABLE = False
    logger.warning(f"⚠️  Technical Analysis module not available: {e}")

try:
    from database import CryptoBotDatabase, save_crypto_price, save_trading_signal
    DATABASE_AVAILABLE = True
    logger.info("✅ Database module loaded")
except ImportError as e:
    DATABASE_AVAILABLE = False
    logger.warning(f"⚠️  Database module not available: {e}")

try:
    from config import get_config, CryptoBotConfig
    CONFIG_AVAILABLE = True
    logger.info("✅ Config module loaded")
except ImportError as e:
    CONFIG_AVAILABLE = False
    logger.warning(f"⚠️  Config module not available: {e}")

try:
    from utils import (format_price, format_percentage, format_volume, 
                      get_current_timestamp, get_current_time_tehran,
                      safe_float, calculate_percentage_change, timing_decorator)
    UTILS_AVAILABLE = True
    logger.info("✅ Utils module loaded")
    # Use the imported timing decorator if available
    timing_decorator_func = timing_decorator
except ImportError as e:
    UTILS_AVAILABLE = False
    logger.warning(f"⚠️  Utils module not available: {e}")
    # Use our safe fallback
    timing_decorator_func = safe_timing_decorator
    
    # Define fallback functions
    def format_price(price, symbol="$"):
        try:
            if price >= 1000:
                return f"{symbol}{price:,.2f}"
            elif price >= 1:
                return f"{symbol}{price:.2f}"
            elif price >= 0.01:
                return f"{symbol}{price:.4f}"
            else:
                return f"{symbol}{price:.8f}"
        except:
            return f"{symbol}0.00"
    
    def format_percentage(value, include_sign=True):
        try:
            if include_sign:
                sign = "+" if value > 0 else ""
                return f"{sign}{value:.2f}%"
            else:
                return f"{abs(value):.2f}%"
        except:
            return "0.00%"
    
    def format_volume(volume):
        try:
            if volume >= 1_000_000_000:
                return f"{volume/1_000_000_000:.2f}B"
            elif volume >= 1_000_000:
                return f"{volume/1_000_000:.2f}M"
            elif volume >= 1_000:
                return f"{volume/1_000:.2f}K"
            else:
                return f"{volume:.0f}"
        except:
            return "0"
    
    def get_current_timestamp():
        return datetime.now().isoformat()
    
    def get_current_time_tehran():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def safe_float(value, default=0.0):
        try:
            return float(value)
        except:
            return default

try:
    from ml_models import CryptoPricePredictor, get_ml_signals
    ML_AVAILABLE = True
    logger.info("✅ ML Models module loaded")
except ImportError as e:
    ML_AVAILABLE = False
    logger.warning(f"⚠️  ML Models module not available: {e}")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Bot configuration
BOT_VERSION = "3.2.0-Enterprise-Fixed"
BOT_NAME = "Crypto Analysis Bot V3.2 Enterprise"

# Initialize components safely
database = None
config = None
ta_analyzer = None
ml_predictor = None

def safe_initialize_components():
    """Safely initialize all bot components with error handling"""
    global database, config, ta_analyzer, ml_predictor
    
    try:
        # Initialize configuration
        if CONFIG_AVAILABLE:
            try:
                config = get_config()
                logger.info("✅ Configuration system initialized")
            except Exception as e:
                logger.warning(f"⚠️  Config initialization failed: {e}")
        
        # Initialize database
        if DATABASE_AVAILABLE:
            try:
                database = CryptoBotDatabase()
                logger.info("✅ Database system initialized")
            except Exception as e:
                logger.warning(f"⚠️  Database initialization failed: {e}")
        
        # Initialize technical analysis
        if TA_AVAILABLE:
            try:
                ta_analyzer = TechnicalAnalysis()
                logger.info("✅ Technical Analysis system initialized")
            except Exception as e:
                logger.warning(f"⚠️  TA initialization failed: {e}")
        
        # Initialize ML predictor
        if ML_AVAILABLE:
            try:
                ml_predictor = CryptoPricePredictor()
                logger.info("✅ Machine Learning system initialized")
            except Exception as e:
                logger.warning(f"⚠️  ML initialization failed: {e}")
        
        logger.info("🚀 Component initialization completed")
        
    except Exception as e:
        logger.error(f"❌ Error in component initialization: {e}")

# Telegram configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7754175422:AAFOHaxVwphnfm43I_Y7BoVdSHmXKcgdQQA")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "6747516161")

# Crypto symbols to monitor
CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]

def send_telegram_message(message: str) -> bool:
    """Send message to Telegram with enterprise error handling"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("⚠️  Telegram not configured")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        logger.info("📱 Telegram message sent successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Telegram error: {e}")
        return False

@timing_decorator_func
def get_crypto_prices() -> List[Dict]:
    """Get crypto prices with professional error handling"""
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        all_data = response.json()
        
        # Filter our symbols and format data
        crypto_data = []
        for item in all_data:
            if item['symbol'] in CRYPTO_SYMBOLS:
                formatted_data = {
                    'symbol': item['symbol'],
                    'price': safe_float(item['lastPrice']),
                    'change_24h': safe_float(item['priceChangePercent']),
                    'volume': safe_float(item['volume']),
                    'high_24h': safe_float(item['highPrice']),
                    'low_24h': safe_float(item['lowPrice']),
                    'timestamp': get_current_timestamp()
                }
                crypto_data.append(formatted_data)
                
                # Save to database if available
                if DATABASE_AVAILABLE and database:
                    try:
                        save_crypto_price(
                            item['symbol'], 
                            formatted_data['price'], 
                            formatted_data['volume'], 
                            formatted_data['change_24h']
                        )
                    except Exception as e:
                        logger.warning(f"⚠️  Database save failed: {e}")
        
        logger.info(f"📊 Retrieved prices for {len(crypto_data)} cryptocurrencies")
        return crypto_data
        
    except Exception as e:
        logger.error(f"❌ Error fetching crypto prices: {e}")
        return []

def perform_advanced_analysis(crypto_data: List[Dict]) -> Dict[str, Any]:
    """Perform comprehensive analysis using available modules"""
    analysis_results = {
        'signals': [],
        'market_overview': {},
        'technical_analysis': {},
        'ml_predictions': {},
        'database_stats': {}
    }
    
    try:
        # Market overview
        if crypto_data:
            total_volume = sum(item['volume'] for item in crypto_data)
            gainers = [item for item in crypto_data if item['change_24h'] > 0]
            losers = [item for item in crypto_data if item['change_24h'] < 0]
            
            analysis_results['market_overview'] = {
                'total_volume': total_volume,
                'gainers_count': len(gainers),
                'losers_count': len(losers),
                'market_sentiment': 'Bullish' if len(gainers) > len(losers) else 'Bearish' if len(losers) > len(gainers) else 'Neutral'
            }
        
        # Technical Analysis
        if TA_AVAILABLE and ta_analyzer:
            for crypto in crypto_data:
                try:
                    # Get historical data for TA
                    if DATABASE_AVAILABLE and database:
                        historical = database.get_historical_prices(crypto['symbol'], 30)
                        if len(historical) > 20:
                            prices = [h['price'] for h in historical]
                            ta_signals = ta_analyzer.generate_signals(crypto['symbol'], prices)
                            analysis_results['technical_analysis'][crypto['symbol']] = ta_signals
                            
                            # Save strong signals
                            if ta_signals.get('strength', 0) > 0.7:
                                analysis_results['signals'].append({
                                    'symbol': crypto['symbol'],
                                    'type': ta_signals['signal'],
                                    'strength': ta_signals['strength'],
                                    'confidence': ta_signals['confidence'],
                                    'source': 'Technical Analysis',
                                    'analysis': ta_signals.get('analysis', '')
                                })
                                
                                # Save to database
                                if DATABASE_AVAILABLE:
                                    try:
                                        save_trading_signal(
                                            crypto['symbol'],
                                            ta_signals['signal'],
                                            ta_signals['strength'],
                                            ta_signals['confidence'],
                                            crypto['price'],
                                            ta_signals.get('indicators', {}),
                                            ta_signals.get('analysis', '')
                                        )
                                    except Exception as e:
                                        logger.warning(f"⚠️  Signal save failed: {e}")
                except Exception as e:
                    logger.warning(f"⚠️  TA analysis failed for {crypto['symbol']}: {e}")
        
        # ML Predictions
        if ML_AVAILABLE and ml_predictor:
            for crypto in crypto_data:
                try:
                    if DATABASE_AVAILABLE and database:
                        historical = database.get_historical_prices(crypto['symbol'], 50)
                        if len(historical) > 30:
                            ml_signals = get_ml_signals(crypto['symbol'], historical)
                            if ml_signals.get('success'):
                                analysis_results['ml_predictions'][crypto['symbol']] = ml_signals
                                
                                # Save strong ML signals
                                if ml_signals.get('strength', 0) > 0.7:
                                    analysis_results['signals'].append({
                                        'symbol': crypto['symbol'],
                                        'type': ml_signals['signal'],
                                        'strength': ml_signals['strength'],
                                        'confidence': ml_signals['confidence'],
                                        'source': 'Machine Learning',
                                        'analysis': ml_signals.get('analysis', '')
                                    })
                except Exception as e:
                    logger.warning(f"⚠️  ML analysis failed for {crypto['symbol']}: {e}")
        
        # Database statistics
        if DATABASE_AVAILABLE and database:
            try:
                analysis_results['database_stats'] = database.get_database_info()
            except Exception as e:
                logger.warning(f"⚠️  Database stats failed: {e}")
        
        logger.info(f"🧠 Advanced analysis completed: {len(analysis_results['signals'])} signals generated")
        return analysis_results
        
    except Exception as e:
        logger.error(f"❌ Error in advanced analysis: {e}")
        return analysis_results

def format_telegram_message(crypto_data: List[Dict], analysis: Dict = None) -> str:
    """Format comprehensive Telegram message"""
    if not crypto_data:
        return "❌ No crypto data available"
    
    current_time = get_current_time_tehran()
    
    message = f"""🤖 <b>Crypto Analysis Bot V3.2 Enterprise</b>
💰 <b>Current Prices</b>
⏰ {current_time}

"""
    
    for crypto in crypto_data:
        change_emoji = "🟢" if crypto['change_24h'] >= 0 else "🔴"
        price_formatted = format_price(crypto['price'])
        change_formatted = format_percentage(crypto['change_24h'])
        volume_formatted = format_volume(crypto['volume'])
        
        message += f"""{change_emoji} <b>{crypto['symbol']}</b>
💵 {price_formatted}
📈 24h: {change_formatted}
📊 Vol: {volume_formatted}

"""
    
    # Add analysis summary if available
    if analysis and analysis.get('signals'):
        message += f"🎯 <b>Signals Detected: {len(analysis['signals'])}</b>\n"
        for signal in analysis['signals'][:3]:  # Top 3 signals
            message += f"• {signal['symbol']}: {signal['type']} ({signal['source']})\n"
    
    if analysis and analysis.get('market_overview'):
        overview = analysis['market_overview']
        message += f"\n📊 Market: {overview.get('market_sentiment', 'Unknown')}"
    
    message += f"\n📡 Live data from Binance API"
    
    return message

def format_daily_summary(crypto_data: List[Dict], analysis: Dict = None) -> str:
    """Format comprehensive daily summary"""
    if not crypto_data:
        return "❌ No market data available"
    
    current_time = get_current_time_tehran()
    
    # Calculate market stats
    gainers = [c for c in crypto_data if c['change_24h'] > 0]
    losers = [c for c in crypto_data if c['change_24h'] < 0]
    total_volume = sum(c['volume'] for c in crypto_data)
    
    # Find top/worst performers
    top_performer = max(crypto_data, key=lambda x: x['change_24h']) if crypto_data else None
    worst_performer = min(crypto_data, key=lambda x: x['change_24h']) if crypto_data else None
    highest_volume = max(crypto_data, key=lambda x: x['volume']) if crypto_data else None
    
    # Market sentiment
    if len(gainers) > len(losers):
        sentiment = "🟢 Bullish"
    elif len(losers) > len(gainers):
        sentiment = "🔴 Bearish"
    else:
        sentiment = "🟡 Neutral"
    
    message = f"""🤖 <b>Crypto Analysis Bot V3.2 Enterprise</b>
📊 <b>Daily Market Summary</b>
⏰ {current_time}

📈 <b>Market Overview:</b>
🟢 Gainers: {len(gainers)}/{len(crypto_data)}
🔴 Losers: {len(losers)}/{len(crypto_data)}
📊 Total Volume: {format_volume(total_volume)}

🏆 <b>Top Performer:</b>
💰 {top_performer['symbol'] if top_performer else 'N/A'} - {format_price(top_performer['price']) if top_performer else 'N/A'}
📈 24h: {format_percentage(top_performer['change_24h']) if top_performer else 'N/A'}

📉 <b>Worst Performer:</b>
💰 {worst_performer['symbol'] if worst_performer else 'N/A'} - {format_price(worst_performer['price']) if worst_performer else 'N/A'}
📈 24h: {format_percentage(worst_performer['change_24h']) if worst_performer else 'N/A'}

🔥 <b>Highest Volume:</b>
💰 {highest_volume['symbol'] if highest_volume else 'N/A'}
📊 Volume: {format_volume(highest_volume['volume']) if highest_volume else 'N/A'}

🎯 <b>Signals Today:</b>
"""
    
    if analysis and analysis.get('signals'):
        message += f"📡 {len(analysis['signals'])} signals detected\n"
        for signal in analysis['signals'][:5]:  # Top 5 signals
            message += f"• {signal['symbol']}: {signal['type']} (Strength: {signal['strength']:.1f})\n"
    else:
        message += "💤 No strong signals - Market in consolidation\n"
    
    message += f"""
📊 <b>Market Sentiment:</b> {sentiment}

💡 <b>Today's Recommendation:</b>
"""
    
    if len(gainers) > len(losers) * 1.5:
        message += "📈 Bullish momentum - Consider DCA on dips"
    elif len(losers) > len(gainers) * 1.5:
        message += "🛡️ Bearish pressure - HODL and wait for reversal"
    else:
        message += "⚖️ Mixed signals - Wait for clearer direction"
    
    message += f"""

🤖 Next analysis in 2 hours
📊 Bot monitoring markets 24/7 🚀"""
    
    return message

# ===============================
# FLASK ROUTES - ENTERPRISE EDITION (FIXED)
# ===============================

@app.route('/', methods=['GET'])
def home():
    """Enterprise home endpoint with comprehensive info"""
    return jsonify({
        "bot_name": BOT_NAME,
        "version": BOT_VERSION,
        "status": "operational",
        "features": {
            "real_time_data": True,
            "telegram_integration": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
            "technical_analysis": TA_AVAILABLE,
            "machine_learning": ML_AVAILABLE,
            "database_storage": DATABASE_AVAILABLE,
            "configuration_management": CONFIG_AVAILABLE,
            "utilities": UTILS_AVAILABLE,
            "automation": True,
            "daily_summaries": True,
            "enterprise_edition": True
        },
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "prices": "/prices",
            "analyze": "/analyze",
            "analyze_advanced": "/analyze?advanced=true",
            "telegram_test": "/telegram/test",
            "telegram_prices": "/telegram/prices",
            "telegram_signals": "/telegram/signals",
            "telegram_daily_summary": "/telegram/daily-summary",
            "system_test": "/test"
        },
        "enterprise_endpoints": {
            "database_info": "/database/info" if DATABASE_AVAILABLE else "unavailable",
            "ml_status": "/ml/status" if ML_AVAILABLE else "unavailable",
            "config_summary": "/config/summary" if CONFIG_AVAILABLE else "unavailable",
            "technical_test": "/technical/test" if TA_AVAILABLE else "unavailable"
        },
        "automation": {
            "analysis_interval": "Every 2 hours",
            "daily_summary_time": "10:00 AM Tehran time",
            "monitoring": "24/7",
            "cloud_scheduler": "Active"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check"""
    return jsonify({
        "status": "healthy",
        "version": BOT_VERSION,
        "timestamp": get_current_timestamp(),
        "uptime_seconds": time.time() - start_time,
        "components": {
            "telegram_enabled": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
            "technical_analysis": TA_AVAILABLE,
            "machine_learning": ML_AVAILABLE,
            "database": DATABASE_AVAILABLE,
            "configuration": CONFIG_AVAILABLE,
            "utilities": UTILS_AVAILABLE
        },
        "automation_ready": True,
        "enterprise_features": True
    })

@app.route('/status', methods=['GET'])
def get_status():
    """Enterprise status with detailed information"""
    verbose = request.args.get('verbose', 'false').lower() == 'true'
    
    status = {
        "system": {
            "status": "operational",
            "version": BOT_VERSION,
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "uptime": str(timedelta(seconds=int(time.time() - start_time)))
        },
        "features": {
            "real_time_data": True,
            "telegram": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
            "technical_analysis": TA_AVAILABLE,
            "machine_learning": ML_AVAILABLE,
            "database": DATABASE_AVAILABLE,
            "configuration": CONFIG_AVAILABLE,
            "utilities": UTILS_AVAILABLE,
            "automation": True,
            "daily_summaries": True,
            "enterprise_edition": True
        },
        "telegram": {
            "enabled": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
            "bot_configured": bool(TELEGRAM_BOT_TOKEN),
            "chat_configured": bool(TELEGRAM_CHAT_ID)
        },
        "automation": {
            "analysis_interval": "2 hours",
            "daily_summary_time": "10:00 AM",
            "monitoring": "24/7",
            "next_analysis": "Every 2 hours",
            "next_summary": "Daily at 10:00 AM"
        },
        "performance": {
            "total_requests": getattr(app, 'request_count', 0),
            "requests_per_minute": getattr(app, 'rpm', 0.0)
        }
    }
    
    if verbose and DATABASE_AVAILABLE and database:
        try:
            status["database"] = database.get_database_info()
        except Exception as e:
            logger.warning(f"⚠️  Database info failed: {e}")
    
    if verbose and CONFIG_AVAILABLE and config:
        try:
            status["configuration"] = config.get_summary()
        except Exception as e:
            logger.warning(f"⚠️  Config info failed: {e}")
    
    return jsonify(status)

@app.route('/prices', methods=['GET'])
def get_prices():
    """Get current cryptocurrency prices"""
    crypto_data = get_crypto_prices()
    
    return jsonify({
        "success": True,
        "timestamp": get_current_timestamp(),
        "count": len(crypto_data),
        "data": crypto_data
    })

@app.route('/analyze', methods=['GET'])
def analyze_crypto():
    """Comprehensive crypto analysis"""
    advanced = request.args.get('advanced', 'false').lower() == 'true'
    telegram_send = request.args.get('telegram', 'false').lower() == 'true'
    
    crypto_data = get_crypto_prices()
    
    if not crypto_data:
        return jsonify({
            "success": False,
            "error": "Failed to retrieve crypto data",
            "timestamp": get_current_timestamp()
        })
    
    analysis_results = {}
    signals = []
    
    if advanced and (TA_AVAILABLE or ML_AVAILABLE):
        analysis_results = perform_advanced_analysis(crypto_data)
        signals = analysis_results.get('signals', [])
        analysis_type = f"Advanced Analysis (TA: {'✅' if TA_AVAILABLE else '❌'}, ML: {'✅' if ML_AVAILABLE else '❌'}, DB: {'✅' if DATABASE_AVAILABLE else '❌'})"
    else:
        # Basic analysis
        for crypto in crypto_data:
            if abs(crypto['change_24h']) > 5:  # Simple threshold
                signals.append({
                    'symbol': crypto['symbol'],
                    'type': 'BUY' if crypto['change_24h'] < -5 else 'SELL',
                    'strength': min(abs(crypto['change_24h']) / 10, 1.0),
                    'confidence': 0.6,
                    'source': 'Basic Analysis',
                    'analysis': f"24h change: {crypto['change_24h']:.2f}%"
                })
        analysis_type = "Basic Analysis"
    
    # Send to Telegram if requested
    telegram_sent = False
    if telegram_send and signals:
        message = format_telegram_message(crypto_data, {'signals': signals})
        telegram_sent = send_telegram_message(message)
    
    return jsonify({
        "success": True,
        "analysis_type": analysis_type,
        "timestamp": get_current_timestamp(),
        "symbols_analyzed": len(crypto_data),
        "signal_count": len(signals),
        "signals": signals,
        "telegram_sent": telegram_sent,
        "advanced_features": {
            "technical_analysis": TA_AVAILABLE,
            "machine_learning": ML_AVAILABLE,
            "database_storage": DATABASE_AVAILABLE
        }
    })

@app.route('/telegram/test', methods=['GET', 'POST'])
def telegram_test():
    """Test Telegram integration"""
    message = f"""🤖 <b>Crypto Analysis Bot V3.2 Enterprise (Fixed)</b>
✅ <b>Test Message</b>
⏰ {get_current_time_tehran()}

🎉 Bot is Online & Working!
📱 Telegram integration successful
🌐 Enterprise edition active
🚀 All systems operational

Enterprise Features:
📊 Technical Analysis: {'✅' if TA_AVAILABLE else '❌'}
🧠 Machine Learning: {'✅' if ML_AVAILABLE else '❌'}
🗄️ Database Storage: {'✅' if DATABASE_AVAILABLE else '❌'}
⚙️ Configuration: {'✅' if CONFIG_AVAILABLE else '❌'}
🔧 Utilities: {'✅' if UTILS_AVAILABLE else '❌'}

Ready to send crypto signals! 🚀"""
    
    success = send_telegram_message(message)
    
    return jsonify({
        "success": success,
        "message": "Test message sent to Telegram" if success else "Failed to send test message",
        "telegram_configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
        "timestamp": get_current_timestamp()
    })

@app.route('/telegram/prices', methods=['GET', 'POST'])
def telegram_prices():
    """Send current prices to Telegram"""
    crypto_data = get_crypto_prices()
    
    if not crypto_data:
        return jsonify({
            "success": False,
            "message": "Failed to get price data",
            "timestamp": get_current_timestamp()
        })
    
    message = format_telegram_message(crypto_data)
    success = send_telegram_message(message)
    
    return jsonify({
        "success": success,
        "message": "Price update sent to Telegram" if success else "Failed to send price update",
        "prices_count": len(crypto_data),
        "timestamp": get_current_timestamp()
    })

@app.route('/telegram/signals', methods=['GET', 'POST'])
def telegram_signals():
    """Analyze and send signals to Telegram"""
    crypto_data = get_crypto_prices()
    
    if not crypto_data:
        return jsonify({
            "success": False,
            "message": "Failed to get crypto data",
            "timestamp": get_current_timestamp()
        })
    
    # Perform advanced analysis
    analysis_results = perform_advanced_analysis(crypto_data)
    signals = analysis_results.get('signals', [])
    
    if signals:
        # Format signals message
        message = f"""🤖 <b>Crypto Analysis Bot V3.2 Enterprise (Fixed)</b>
📊 <b>Analysis Complete</b>
⏰ {get_current_time_tehran()}

🎯 <b>Signals Detected: {len(signals)}</b>

"""
        
        for signal in signals[:5]:  # Top 5 signals
            signal_emoji = "🟢" if signal['type'] in ['BUY', 'STRONG_BUY'] else "🔴"
            message += f"""{signal_emoji} <b>{signal['symbol']}</b>
📈 Signal: {signal['type']}
💪 Strength: {signal['strength']:.2f}
🎯 Confidence: {signal['confidence']:.2f}
🔍 Source: {signal['source']}

"""
        
        message += "Bot is monitoring markets 24/7 🚀"
        success = send_telegram_message(message)
        
    else:
        # No strong signals
        message = f"""🤖 <b>Crypto Analysis Bot V3.2 Enterprise (Fixed)</b>
📊 <b>Analysis Complete</b>
⏰ {get_current_time_tehran()}

📈 <b>Market Status: No Strong Signals</b>
🔍 <b>Symbols Analyzed:</b> {len(crypto_data)} cryptos
💡 <b>Recommendation:</b> HODL - Market in consolidation

Bot is monitoring markets 24/7 🚀"""
        success = send_telegram_message(message)
    
    return jsonify({
        "success": success,
        "message": "Signals sent to Telegram" if success else "Failed to send signals",
        "signal_count": len(signals),
        "symbols_analyzed": len(crypto_data),
        "timestamp": get_current_timestamp()
    })

@app.route('/telegram/daily-summary', methods=['GET', 'POST'])
def telegram_daily_summary():
    """Send comprehensive daily market summary to Telegram"""
    crypto_data = get_crypto_prices()
    
    if not crypto_data:
        return jsonify({
            "success": False,
            "message": "Failed to get market data",
            "timestamp": get_current_timestamp()
        })
    
    # Perform comprehensive analysis
    analysis_results = perform_advanced_analysis(crypto_data)
    
    # Format and send daily summary
    message = format_daily_summary(crypto_data, analysis_results)
    success = send_telegram_message(message)
    
    # Save market summary to database
    if DATABASE_AVAILABLE and database:
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            summary_data = {
                'total_volume': sum(c['volume'] for c in crypto_data),
                'gainers': len([c for c in crypto_data if c['change_24h'] > 0]),
                'losers': len([c for c in crypto_data if c['change_24h'] < 0]),
                'top_performer': max(crypto_data, key=lambda x: x['change_24h'])['symbol'] if crypto_data else '',
                'worst_performer': min(crypto_data, key=lambda x: x['change_24h'])['symbol'] if crypto_data else '',
                'signal_count': len(analysis_results.get('signals', [])),
                'market_sentiment': analysis_results.get('market_overview', {}).get('market_sentiment', 'Unknown')
            }
            database.save_market_summary(today, summary_data)
        except Exception as e:
            logger.warning(f"⚠️  Failed to save market summary: {e}")
    
    return jsonify({
        "success": success,
        "message": "Daily summary sent to Telegram" if success else "Failed to send daily summary",
        "signal_count": len(analysis_results.get('signals', [])),
        "symbols_analyzed": len(crypto_data),
        "market_summary": {
            "gainers": len([c for c in crypto_data if c['change_24h'] > 0]),
            "losers": len([c for c in crypto_data if c['change_24h'] < 0]),
            "total_volume": sum(c['volume'] for c in crypto_data)
        },
        "timestamp": get_current_timestamp()
    })

# ===============================
# ADVANCED ENTERPRISE ENDPOINTS (SAFE)
# ===============================

@app.route('/database/info', methods=['GET'])
def database_info():
    """Get database information and statistics"""
    if not DATABASE_AVAILABLE:
        return jsonify({
            "success": False,
            "error": "Database module not available",
            "database_available": False
        })
    
    if not database:
        return jsonify({
            "success": False,
            "error": "Database not initialized",
            "database_available": True
        })
    
    try:
        info = database.get_database_info()
        performance_stats = database.get_performance_stats(30)
        
        return jsonify({
            "success": True,
            "database_info": info,
            "performance_stats": performance_stats,
            "timestamp": get_current_timestamp()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": get_current_timestamp()
        })

@app.route('/ml/status', methods=['GET'])
def ml_status():
    """Get machine learning system status"""
    if not ML_AVAILABLE:
        return jsonify({
            "success": False,
            "ml_available": False,
            "error": "ML libraries not installed"
        })
    
    try:
        if ml_predictor:
            model_info = ml_predictor.get_model_info()
            return jsonify({
                "success": True,
                "ml_available": True,
                "model_info": model_info,
                "timestamp": get_current_timestamp()
            })
        else:
            return jsonify({
                "success": False,
                "ml_available": True,
                "error": "ML predictor not initialized"
            })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": get_current_timestamp()
        })

@app.route('/config/summary', methods=['GET'])
def config_summary():
    """Get configuration summary"""
    if not CONFIG_AVAILABLE:
        return jsonify({
            "success": False,
            "error": "Configuration system not available",
            "config_available": False
        })
    
    try:
        if config:
            summary = config.get_summary()
            return jsonify({
                "success": True,
                "configuration": summary,
                "timestamp": get_current_timestamp()
            })
        else:
            return jsonify({
                "success": False,
                "error": "Configuration not initialized"
            })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": get_current_timestamp()
        })

@app.route('/technical/test', methods=['GET'])
def technical_test():
    """Test technical analysis capabilities"""
    if not TA_AVAILABLE:
        return jsonify({
            "success": False,
            "ta_available": False,
            "error": "Technical Analysis libraries not available"
        })
    
    try:
        # Generate test data
        test_prices = [100 + i + (i % 5) * 2 for i in range(50)]
        
        if ta_analyzer:
            # Test RSI
            rsi = ta_analyzer.calculate_rsi(test_prices)
            
            # Test MACD
            macd_data = ta_analyzer.calculate_macd(test_prices)
            
            # Test signals
            signals = ta_analyzer.generate_signals("TESTUSDT", test_prices)
            
            return jsonify({
                "success": True,
                "ta_available": True,
                "test_results": {
                    "rsi_current": rsi[-1] if rsi else 0,
                    "macd_current": macd_data['macd'][-1] if macd_data['macd'] else 0,
                    "signal_generated": signals.get('signal', 'HOLD'),
                    "signal_strength": signals.get('strength', 0),
                    "signal_confidence": signals.get('confidence', 0)
                },
                "timestamp": get_current_timestamp()
            })
        else:
            return jsonify({
                "success": False,
                "ta_available": True,
                "error": "TA analyzer not initialized"
            })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": get_current_timestamp()
        })

@app.route('/test', methods=['GET'])
def test_apis():
    """Comprehensive system test"""
    test_results = {
        "local_server": True,
        "binance": False,
        "telegram": False,
        "database": False,
        "technical_analysis": False,
        "machine_learning": False,
        "configuration": False,
        "utilities": UTILS_AVAILABLE
    }
    
    # Test Binance API
    try:
        response = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
        test_results["binance"] = response.status_code == 200
    except:
        pass
    
    # Test Telegram
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"
            response = requests.get(url, timeout=10)
            test_results["telegram"] = response.status_code == 200
        except:
            pass
    
    # Test Database
    if DATABASE_AVAILABLE and database:
        try:
            info = database.get_database_info()
            test_results["database"] = True
        except:
            pass
    
    # Test Technical Analysis
    if TA_AVAILABLE and ta_analyzer:
        try:
            test_prices = [100, 101, 102, 103, 104]
            rsi = ta_analyzer.calculate_rsi(test_prices)
            test_results["technical_analysis"] = len(rsi) > 0
        except:
            pass
    
    # Test Machine Learning
    if ML_AVAILABLE and ml_predictor:
        try:
            info = ml_predictor.get_model_info()
            test_results["machine_learning"] = info.get("ml_available", False)
        except:
            pass
    
    # Test Configuration
    if CONFIG_AVAILABLE and config:
        try:
            summary = config.get_summary()
            test_results["configuration"] = True
        except:
            pass
    
    return jsonify({
        "success": True,
        "api_status": test_results,
        "telegram_configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
        "overall_health": sum(test_results.values()) / len(test_results),
        "components_available": {
            "technical_analysis": TA_AVAILABLE,
            "machine_learning": ML_AVAILABLE,
            "database": DATABASE_AVAILABLE,
            "configuration": CONFIG_AVAILABLE,
            "utilities": UTILS_AVAILABLE
        },
        "timestamp": get_current_timestamp()
    })

# ===============================
# STARTUP AND INITIALIZATION
# ===============================

def startup_notification():
    """Send startup notification to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info("📱 Telegram not configured - skipping startup notification")
        return
    
    try:
        message = f"""🤖 <b>Crypto Analysis Bot V3.2 Enterprise (Fixed)</b>
🚀 <b>Bot Started Successfully!</b>
⏰ {get_current_time_tehran()}

✅ <b>Enterprise Features Status:</b>
📊 Technical Analysis: {'✅' if TA_AVAILABLE else '❌'}
🧠 Machine Learning: {'✅' if ML_AVAILABLE else '❌'}
🗄️ Database Storage: {'✅' if DATABASE_AVAILABLE else '❌'}
⚙️ Configuration: {'✅' if CONFIG_AVAILABLE else '❌'}
🔧 Utilities: {'✅' if UTILS_AVAILABLE else '❌'}

📈 <b>Monitoring {len(CRYPTO_SYMBOLS)} crypto pairs</b>
🔔 <b>Ready to send intelligent signals</b>
⏰ <b>Analysis every 2 hours</b>
📊 <b>Daily summary at 10:00 AM</b>

Bot is now online and monitoring markets! 🚀"""
        
        success = send_telegram_message(message)
        if success:
            logger.info("📱 Startup notification sent to Telegram")
        else:
            logger.warning("⚠️  Failed to send startup notification")
            
    except Exception as e:
        logger.error(f"❌ Error sending startup notification: {e}")

# Request counter middleware
@app.before_request
def before_request():
    if not hasattr(app, 'request_count'):
        app.request_count = 0
    app.request_count += 1

# Initialize start time
start_time = time.time()

if __name__ == "__main__":
    # Print startup banner
    print("=" * 60)
    print(f"🚀 Starting {BOT_NAME}...")
    print(f"📱 Version: {BOT_VERSION}")
    print("📱 Features: Enterprise Edition with Safe Integration")
    print("=" * 60)
    
    # Initialize all components safely
    safe_initialize_components()
    
    # Print component status
    print(f"📊 Technical Analysis: {'✅' if TA_AVAILABLE else '❌'}")
    print(f"🧠 Machine Learning: {'✅' if ML_AVAILABLE else '❌'}")
    print(f"🗄️ Database Storage: {'✅' if DATABASE_AVAILABLE else '❌'}")
    print(f"⚙️ Configuration: {'✅' if CONFIG_AVAILABLE else '❌'}")
    print(f"🔧 Utilities: {'✅' if UTILS_AVAILABLE else '❌'}")
    print(f"📱 Telegram: {'✅' if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID else '❌'}")
    
    # Print available endpoints
    print("\n🌐 Available endpoints:")
    endpoints = [
        "GET  / - Enterprise home with features info",
        "GET  /health - Comprehensive health check", 
        "GET  /status - Enterprise system status",
        "GET  /prices - Current crypto prices",
        "GET  /analyze - Advanced crypto analysis",
        "GET  /analyze?advanced=true - Full enterprise analysis",
        "GET  /telegram/test - Test Telegram integration",
        "GET  /telegram/prices - Send prices to Telegram",
        "GET  /telegram/signals - Send signals to Telegram", 
        "GET  /telegram/daily-summary - Send daily summary",
        "GET  /test - Comprehensive system test"
    ]
    
    # Add enterprise endpoints if available
    if DATABASE_AVAILABLE:
        endpoints.append("GET  /database/info - Database statistics")
    if ML_AVAILABLE:
        endpoints.append("GET  /ml/status - Machine learning status")
    if CONFIG_AVAILABLE:
        endpoints.append("GET  /config/summary - Configuration summary")
    if TA_AVAILABLE:
        endpoints.append("GET  /technical/test - Technical analysis test")
    
    for endpoint in endpoints:
        print(f"   {endpoint}")
    
    print(f"\n🌐 Starting on http://localhost:8080")
    print("=" * 60)
    
    # Send startup notification
    startup_notification()
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=8080, debug=True)