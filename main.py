#!/usr/bin/env python3
"""
Crypto Analysis Bot V3.2 - Simple Version با Telegram Integration
Local Testing Version + Cloud Run Ready
"""

import os
import json
import logging
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests

# =============================================================================
# 🚀 FLASK APPLICATION SETUP
# =============================================================================

app = Flask(__name__)
CORS(app)

# Configuration
BOT_VERSION = "3.2.0-Simple-Telegram"
app_start_time = datetime.now()
total_requests = 0

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# 📱 TELEGRAM CONFIGURATION
# =============================================================================

# Telegram credentials
TELEGRAM_BOT_TOKEN = '7754175422:AAFOHaxVwphnfm43I_Y7BoVdSHmXKcgdQQA'
TELEGRAM_CHAT_ID = '55174977'

def send_telegram_sync(message):
    """Send message to Telegram synchronously"""
    try:
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            logger.warning("❌ Telegram credentials not configured")
            return False
        
        url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML',
            'disable_web_page_preview': True
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            logger.info("✅ Telegram message sent successfully")
            return True
        else:
            logger.error(f"❌ Telegram error {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"💥 Error sending Telegram message: {str(e)}")
        return False

def format_signal_message(signals):
    """Format crypto signals for Telegram"""
    if not signals:
        return f"""
🤖 <b>Crypto Analysis Bot V3.2</b>

📊 <b>Analysis Complete</b>
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📈 <b>Market Status:</b> No Strong Signals
🔍 <b>Symbols Analyzed:</b> 5 cryptos
💡 <b>Recommendation:</b> HODL - Market in consolidation

<i>Bot is monitoring markets 24/7</i> 🚀
"""
    
    message = f"""
🤖 <b>Crypto Analysis Bot V3.2</b>

🚨 <b>STRONG SIGNALS DETECTED!</b> 🚨
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
    
    for i, signal in enumerate(signals[:3], 1):  # Top 3 signals only
        action = "🟢 BUY" if signal['recommendation'] == 'BUY' else "🔴 SELL"
        
        message += f"""
📈 <b>Signal #{i}</b>
💰 <b>{signal['symbol']}</b> - ${signal['price']:,.2f}
{action} | 💪 Strength: {signal['signal_strength']}/100
📊 Trend: {signal['analysis'].get('trend', 'N/A')}
📈 24h Change: {signal['change_24h']:+.2f}%
🔥 Volatility: {signal['analysis'].get('volatility', 'N/A')}

"""
    
    message += """
⚠️ <b>Risk Disclaimer:</b> 
<i>This is automated analysis. Always DYOR!</i>

🤖 <i>Bot monitoring markets 24/7</i> 🚀
"""
    
    return message

def format_price_update(crypto_data):
    """Format price update for Telegram"""
    message = f"""
🤖 <b>Crypto Analysis Bot V3.2</b>

💰 <b>Current Prices</b> 💰
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
    
    for crypto in crypto_data:
        change_emoji = "🟢" if crypto['change_24h'] > 0 else "🔴" if crypto['change_24h'] < 0 else "🟡"
        
        message += f"""
{change_emoji} <b>{crypto['symbol']}</b>
💵 ${crypto['price']:,.2f}
📈 24h: {crypto['change_24h']:+.2f}%
📊 Vol: {crypto['volume']:,.0f}

"""
    
    message += "<i>Live data from Binance API</i> 📡"
    return message

# =============================================================================
# 📊 CRYPTO DATA FUNCTIONS
# =============================================================================

def get_crypto_prices():
    """Get current crypto prices from Binance API"""
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Filter top cryptos
            top_cryptos = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
            
            filtered_data = []
            for item in data:
                if item['symbol'] in top_cryptos:
                    filtered_data.append({
                        'symbol': item['symbol'],
                        'price': float(item['lastPrice']),
                        'change_24h': float(item['priceChangePercent']),
                        'volume': float(item['volume']),
                        'high_24h': float(item['highPrice']),
                        'low_24h': float(item['lowPrice'])
                    })
            
            return filtered_data
        else:
            logger.error(f"Binance API error: {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"Error fetching crypto prices: {str(e)}")
        return []

def analyze_simple_signals(crypto_data):
    """Simple analysis without TA-Lib"""
    signals = []
    
    for crypto in crypto_data:
        signal_strength = 0
        analysis = {}
        
        # Simple price change analysis
        change_24h = crypto['change_24h']
        
        if change_24h > 5:
            signal_strength += 30
            analysis['trend'] = 'Strong Bullish'
        elif change_24h > 2:
            signal_strength += 20
            analysis['trend'] = 'Bullish'
        elif change_24h < -5:
            signal_strength += 25
            analysis['trend'] = 'Strong Bearish'
        elif change_24h < -2:
            signal_strength += 15
            analysis['trend'] = 'Bearish'
        else:
            analysis['trend'] = 'Neutral'
        
        # Volume analysis
        if crypto['volume'] > 1000000:
            signal_strength += 20
            analysis['volume_status'] = 'High Volume'
        else:
            signal_strength += 5
            analysis['volume_status'] = 'Normal Volume'
        
        # Price range analysis
        price_range = ((crypto['high_24h'] - crypto['low_24h']) / crypto['price']) * 100
        
        if price_range > 10:
            signal_strength += 15
            analysis['volatility'] = 'High'
        elif price_range > 5:
            signal_strength += 10
            analysis['volatility'] = 'Medium'
        else:
            signal_strength += 5
            analysis['volatility'] = 'Low'
        
        # Create signal if strength > 50
        if signal_strength >= 50:
            signal = {
                'symbol': crypto['symbol'],
                'price': crypto['price'],
                'signal_strength': signal_strength,
                'recommendation': 'BUY' if change_24h > 0 else 'SELL',
                'analysis': analysis,
                'change_24h': change_24h,
                'timestamp': datetime.now().isoformat()
            }
            signals.append(signal)
    
    return signals

# =============================================================================
# 🌐 FLASK API ENDPOINTS
# =============================================================================

@app.route('/', methods=['GET'])
def home():
    """Welcome page"""
    return jsonify({
        'message': 'Welcome to Crypto Analysis Bot V3.2 Simple + Telegram',
        'version': BOT_VERSION,
        'status': 'running',
        'telegram_configured': bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
        'endpoints': {
            'health': '/health',
            'status': '/status',
            'analyze': '/analyze',
            'prices': '/prices',
            'telegram_test': '/telegram/test',
            'telegram_prices': '/telegram/prices',
            'telegram_signals': '/telegram/signals',
            'test': '/test'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint"""
    global total_requests
    total_requests += 1
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': BOT_VERSION,
        'uptime_seconds': (datetime.now() - app_start_time).total_seconds(),
        'telegram_enabled': bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
    })

@app.route('/status', methods=['GET'])
def get_status():
    """Comprehensive system status"""
    uptime = datetime.now() - app_start_time
    
    return jsonify({
        'system': {
            'status': 'operational',
            'version': BOT_VERSION,
            'uptime': str(uptime),
            'start_time': app_start_time.isoformat()
        },
        'performance': {
            'total_requests': total_requests,
            'requests_per_minute': total_requests / max(uptime.total_seconds() / 60, 1)
        },
        'features': {
            'simple_analysis': True,
            'ta_lib': False,
            'telegram': bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
            'real_time_data': True
        },
        'telegram': {
            'enabled': bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
            'bot_configured': bool(TELEGRAM_BOT_TOKEN),
            'chat_configured': bool(TELEGRAM_CHAT_ID)
        }
    })

@app.route('/prices', methods=['GET'])
def get_prices():
    """Get current crypto prices"""
    try:
        prices = get_crypto_prices()
        
        return jsonify({
            'success': True,
            'data': prices,
            'count': len(prices),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in prices endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Simple crypto analysis"""
    try:
        logger.info("🎯 Starting simple crypto analysis...")
        
        # Get parameters
        send_telegram = request.args.get('telegram', 'false').lower() == 'true'
        
        # Get crypto data
        crypto_data = get_crypto_prices()
        
        if not crypto_data:
            return jsonify({
                'success': False,
                'error': 'Failed to fetch crypto data'
            }), 500
        
        # Perform simple analysis
        signals = analyze_simple_signals(crypto_data)
        
        result = {
            'success': True,
            'signals': signals,
            'signal_count': len(signals),
            'analysis_type': 'Simple Analysis (No TA-Lib)',
            'symbols_analyzed': len(crypto_data),
            'timestamp': datetime.now().isoformat(),
            'telegram_sent': False
        }
        
        # Send to Telegram if requested or if strong signals found
        if send_telegram or len(signals) > 0:
            try:
                message = format_signal_message(signals)
                telegram_success = send_telegram_sync(message)
                result['telegram_sent'] = telegram_success
                
                if telegram_success:
                    logger.info("📱 Analysis sent to Telegram")
                else:
                    logger.warning("⚠️ Failed to send analysis to Telegram")
                    
            except Exception as e:
                logger.error(f"Error sending to Telegram: {str(e)}")
        
        logger.info(f"✅ Analysis completed: {len(signals)} signals found")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# =============================================================================
# 📱 TELEGRAM SPECIFIC ENDPOINTS
# =============================================================================

@app.route('/telegram/test', methods=['GET', 'POST'])
def telegram_test():
    """Test Telegram connectivity"""
    try:
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return jsonify({
                'success': False,
                'error': 'Telegram credentials not configured'
            }), 400
        
        test_message = f"""
🤖 <b>Crypto Analysis Bot V3.2</b>

✅ <b>Test Message</b>
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🎉 <b>Bot is Online & Working!</b>
📱 Telegram integration successful
🌐 Local/Cloud deployment active

<i>Ready to send crypto signals!</i> 🚀
"""
        
        success = send_telegram_sync(test_message)
        
        return jsonify({
            'success': success,
            'message': 'Test message sent to Telegram' if success else 'Failed to send test message',
            'telegram_configured': bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error testing Telegram: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/telegram/prices', methods=['GET', 'POST'])
def telegram_prices():
    """Send current prices to Telegram"""
    try:
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return jsonify({
                'success': False,
                'error': 'Telegram credentials not configured'
            }), 400
        
        # Get crypto data
        crypto_data = get_crypto_prices()
        
        if not crypto_data:
            return jsonify({
                'success': False,
                'error': 'Failed to fetch crypto data'
            }), 500
        
        # Format and send message
        message = format_price_update(crypto_data)
        success = send_telegram_sync(message)
        
        return jsonify({
            'success': success,
            'message': 'Price update sent to Telegram' if success else 'Failed to send price update',
            'prices_count': len(crypto_data),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error sending prices to Telegram: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/telegram/signals', methods=['GET', 'POST'])
def telegram_signals():
    """Analyze and send signals to Telegram"""
    try:
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return jsonify({
                'success': False,
                'error': 'Telegram credentials not configured'
            }), 400
        
        # Get crypto data and analyze
        crypto_data = get_crypto_prices()
        
        if not crypto_data:
            return jsonify({
                'success': False,
                'error': 'Failed to fetch crypto data'
            }), 500
        
        signals = analyze_simple_signals(crypto_data)
        
        # Format and send message
        message = format_signal_message(signals)
        success = send_telegram_sync(message)
        
        return jsonify({
            'success': success,
            'message': 'Signal analysis sent to Telegram' if success else 'Failed to send signals',
            'signal_count': len(signals),
            'symbols_analyzed': len(crypto_data),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error sending signals to Telegram: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/test', methods=['GET'])
def test_apis():
    """Test external APIs"""
    try:
        # Test Binance API
        binance_status = False
        try:
            response = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
            binance_status = response.status_code == 200
        except:
            pass
        
        # Test Telegram API
        telegram_status = False
        if TELEGRAM_BOT_TOKEN:
            try:
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"
                response = requests.get(url, timeout=5)
                telegram_status = response.status_code == 200
            except:
                pass
        
        return jsonify({
            'success': True,
            'api_status': {
                'binance': binance_status,
                'telegram': telegram_status,
                'local_server': True
            },
            'telegram_configured': bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# =============================================================================
# 🚀 APPLICATION STARTUP
# =============================================================================

if __name__ == '__main__':
    try:
        logger.info("🚀 Starting Crypto Analysis Bot V3.2 Simple + Telegram...")
        logger.info("📱 Features: Simple Analysis, Real-time Prices, Telegram Integration")
        
        # Check Telegram configuration
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            logger.info("📱 Telegram: Configured ✅")
            
            # Send startup notification
            startup_message = f"""
🤖 <b>Crypto Analysis Bot V3.2</b>

🚀 <b>Bot Started Successfully!</b>
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

✅ All systems operational
📊 Monitoring 5 crypto pairs
🔔 Ready to send signals

<i>Bot is now online and monitoring markets!</i> 🚀
"""
            try:
                send_telegram_sync(startup_message)
                logger.info("📱 Startup notification sent to Telegram")
            except Exception as e:
                logger.error(f"Failed to send startup notification: {str(e)}")
        else:
            logger.warning("📱 Telegram: Not configured ⚠️")
        
        # Print available endpoints
        logger.info("🌐 Available endpoints:")
        logger.info("   GET  / - Home page")
        logger.info("   GET  /health - Health check")
        logger.info("   GET  /status - System status")
        logger.info("   GET  /prices - Current crypto prices")
        logger.info("   GET  /analyze - Simple crypto analysis")
        logger.info("   GET  /telegram/test - Test Telegram")
        logger.info("   GET  /telegram/prices - Send prices to Telegram")
        logger.info("   GET  /telegram/signals - Send signals to Telegram")
        logger.info("   GET  /test - Test APIs")
        
        port = int(os.getenv('PORT', 8080))
        logger.info(f"🌐 Starting on http://localhost:{port}")
        
        app.run(
            host='0.0.0.0',
            port=port,
            debug=True,
            threaded=True
        )
        
    except Exception as e:
        logger.error(f"💥 Error starting application: {str(e)}")
        
        # Send error notification if possible
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            error_message = f"""
🤖 <b>Crypto Analysis Bot V3.2</b>

❌ <b>Startup Error!</b>
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

💥 Error: {str(e)}

<i>Bot failed to start properly</i> 🚨
"""
            try:
                send_telegram_sync(error_message)
            except:
                pass