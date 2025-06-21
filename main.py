#!/usr/bin/env python3
"""
Crypto Analysis Bot V3.2 - Simple Version (No TA-Lib)
Local Testing Version
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

# Simple configuration
BOT_VERSION = "3.2.0-Simple"
app_start_time = datetime.now()
total_requests = 0

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# 📊 SIMPLE CRYPTO DATA FUNCTIONS
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
        'message': 'Welcome to Crypto Analysis Bot V3.2 Simple',
        'version': BOT_VERSION,
        'status': 'running',
        'endpoints': {
            'health': '/health',
            'status': '/status',
            'analyze': '/analyze',
            'prices': '/prices'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global total_requests
    total_requests += 1
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': BOT_VERSION,
        'uptime_seconds': (datetime.now() - app_start_time).total_seconds()
    })

@app.route('/status', methods=['GET'])
def get_status():
    """System status"""
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
            'telegram': False,
            'real_time_data': True
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
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Analysis completed: {len(signals)} signals found")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
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
        
        return jsonify({
            'success': True,
            'api_status': {
                'binance': binance_status,
                'local_server': True
            },
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
        logger.info("🚀 Starting Crypto Analysis Bot V3.2 Simple...")
        logger.info("📱 Features: Simple Analysis, Real-time Prices, No TA-Lib")
        
        # Print available endpoints
        logger.info("🌐 Available endpoints:")
        logger.info("   GET  / - Home page")
        logger.info("   GET  /health - Health check")
        logger.info("   GET  /status - System status")
        logger.info("   GET  /prices - Current crypto prices")
        logger.info("   GET  /analyze - Simple crypto analysis")
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
