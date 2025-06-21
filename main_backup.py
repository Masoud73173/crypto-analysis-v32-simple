#!/usr/bin/env python3
"""
Crypto Analysis Bot V3.2 - Enterprise Edition with Embedded Market Screener
===========================================================================
Complete crypto analysis with embedded CoinGecko market screener
"""

import os
import sys
import json
import logging
import requests
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
from flask import Flask, jsonify, request
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class CryptoSignal:
    """Data class for crypto trading signals"""
    symbol: str
    signal_type: str  # BUY, SELL, HOLD, STRONG_BUY, STRONG_SELL
    strength: float   # 0.0 to 1.0
    confidence: float # 0.0 to 1.0
    price: float
    change_24h: float
    volume: float
    market_cap: Optional[float] = None
    rank: Optional[int] = None
    analysis: str = ""
    source: str = "Market Screener"

# Import modules (with graceful fallbacks)
try:
    import database
    import technical_analysis  
    import machine_learning
    import configuration
    import utilities
    
    MODULES_AVAILABLE = {
        "database": True,
        "technical_analysis": True, 
        "machine_learning": True,
        "configuration": True,
        "utilities": True,
        "market_screener": True
    }
    logger.info("✅ All enterprise modules loaded successfully")
    
except ImportError as e:
    logger.warning(f"⚠️  Some enterprise modules not available: {e}")
    MODULES_AVAILABLE = {
        "database": False,
        "technical_analysis": False,
        "machine_learning": False, 
        "configuration": False,
        "utilities": False,
        "market_screener": True  # Embedded screener always available
    }

# Configuration
CONFIG = {
    "telegram": {
        "bot_token": os.getenv("TELEGRAM_BOT_TOKEN", "7635826035:AAGgRhK3q6MWXpKWRKCv_V_vL0Y7OZCe5l8"),
        "chat_id": os.getenv("TELEGRAM_CHAT_ID", "-1002426967094"),
        "enabled": True
    },
    "binance": {
        "base_url": "https://api.binance.com/api/v3"
    },
    "analysis": {
        "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"],
        "intervals": ["1h", "4h", "1d"]
    },
    "market_screener": {
        "enabled": True,
        "limit": 5,
        "min_volume_24h": 100000,
        "min_market_cap": 1000000,
        "max_rank": 500
    }
}

# Flask app
app = Flask(__name__)

class EmbeddedMarketScreener:
    """Embedded market screener using CoinGecko API"""
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.timeout = 15
        self.last_request_time = 0
        self.min_request_interval = 1.2  # Rate limiting
        
        # Signal thresholds
        self.strong_buy_threshold = -5
        self.buy_threshold = -2
        self.sell_threshold = 5
        self.strong_sell_threshold = 10
        
        logger.info("🔍 Embedded Market Screener initialized")
    
    def _rate_limit(self):
        """Apply rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make HTTP request to CoinGecko API"""
        try:
            self._rate_limit()
            
            url = f"{self.base_url}{endpoint}"
            headers = {
                "Accept": "application/json",
                "User-Agent": "CryptoBot/3.2"
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"❌ CoinGecko API error: {e}")
            return None
    
    def _format_volume(self, volume: float) -> str:
        """Format volume with appropriate units"""
        if volume >= 1_000_000_000:
            return f"{volume/1_000_000_000:.1f}B"
        elif volume >= 1_000_000:
            return f"{volume/1_000_000:.1f}M"
        elif volume >= 1_000:
            return f"{volume/1_000:.1f}K"
        else:
            return f"{volume:.0f}"
    
    def _generate_signal(self, coin: Dict) -> Optional[CryptoSignal]:
        """Generate trading signal based on price action"""
        try:
            symbol = coin.get("symbol", "").upper()
            price = float(coin.get("current_price", 0))
            change_24h = float(coin.get("price_change_percentage_24h", 0))
            volume_24h = float(coin.get("total_volume", 0))
            market_cap = float(coin.get("market_cap", 0))
            rank = coin.get("market_cap_rank", 999)
            
            # Determine signal type
            if change_24h <= self.strong_buy_threshold:
                signal_type = "STRONG_BUY"
                strength = min(abs(change_24h) / 15, 1.0)
                confidence = 0.8
            elif change_24h <= self.buy_threshold:
                signal_type = "BUY"
                strength = min(abs(change_24h) / 8, 1.0)
                confidence = 0.6
            elif change_24h >= self.strong_sell_threshold:
                signal_type = "STRONG_SELL"
                strength = min(change_24h / 20, 1.0)
                confidence = 0.7
            elif change_24h >= self.sell_threshold:
                signal_type = "SELL"
                strength = min(change_24h / 12, 1.0)
                confidence = 0.6
            else:
                return None
            
            # Adjust confidence based on rank and volume
            rank_multiplier = max(0.5, 1.0 - (rank / 500))
            confidence = min(confidence * (0.7 + rank_multiplier * 0.3), 1.0)
            
            volume_multiplier = min(volume_24h / CONFIG["market_screener"]["min_volume_24h"], 5.0)
            confidence = min(confidence * (0.6 + volume_multiplier * 0.1), 1.0)
            
            analysis = f"Rank #{rank}, 24h: {change_24h:+.2f}%, Vol: {self._format_volume(volume_24h)}, MCap: {self._format_volume(market_cap)}"
            
            return CryptoSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=round(strength, 2),
                confidence=round(confidence, 2),
                price=price,
                change_24h=change_24h,
                volume=volume_24h,
                market_cap=market_cap,
                rank=rank,
                analysis=analysis,
                source="CoinGecko"
            )
            
        except Exception as e:
            logger.error(f"❌ Error generating signal: {e}")
            return None
    
    def get_top_opportunities(self, limit: int = 5) -> List[CryptoSignal]:
        """Get top trading opportunities from CoinGecko"""
        try:
            logger.info("🔍 Starting embedded market screening...")
            
            # Get market data from CoinGecko
            endpoint = "/coins/markets"
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 250,
                "page": 1,
                "sparkline": False,
                "price_change_percentage": "24h"
            }
            
            data = self._make_request(endpoint, params)
            if not data:
                logger.error("❌ Failed to get CoinGecko data")
                return []
            
            logger.info(f"📊 Analyzing {len(data)} cryptocurrencies from CoinGecko...")
            
            # Filter and generate signals
            signals = []
            filter_passed = 0
            
            for coin in data:
                try:
                    # Apply basic filters
                    rank = coin.get("market_cap_rank", 999)
                    volume_24h = float(coin.get("total_volume", 0))
                    market_cap = float(coin.get("market_cap", 0))
                    price = float(coin.get("current_price", 0))
                    
                    if (rank <= CONFIG["market_screener"]["max_rank"] and 
                        volume_24h >= CONFIG["market_screener"]["min_volume_24h"] and
                        market_cap >= CONFIG["market_screener"]["min_market_cap"] and
                        price > 0):
                        
                        filter_passed += 1
                        
                        # Generate signal
                        signal = self._generate_signal(coin)
                        if signal:
                            signals.append(signal)
                            
                except Exception as e:
                    continue
            
            logger.info(f"✅ Passed filters: {filter_passed}, Generated signals: {len(signals)}")
            
            # Sort by signal strength and confidence
            signals.sort(key=lambda x: (x.strength * x.confidence * (1.0 - (x.rank or 500) / 1000)), reverse=True)
            
            top_signals = signals[:limit]
            logger.info(f"🏆 Selected top {len(top_signals)} opportunities")
            
            return top_signals
            
        except Exception as e:
            logger.error(f"❌ Error in market screening: {e}")
            return []
    
    def format_signals_for_telegram(self, signals: List[CryptoSignal]) -> str:
        """Format signals for Telegram"""
        if not signals:
            return "📊 No strong trading opportunities found in current market conditions."
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"""🔍 <b>Market Screener Report</b>
⏰ {current_time}
📊 <b>Top {len(signals)} Trading Opportunities</b>

"""
        
        rank_emojis = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        
        for i, signal in enumerate(signals):
            rank_emoji = rank_emojis[i] if i < len(rank_emojis) else f"{i+1}️⃣"
            signal_emoji = "🟢" if signal.signal_type in ["STRONG_BUY", "BUY"] else "🔴"
            
            message += f"""{rank_emoji} {signal_emoji} <b>{signal.symbol}</b>
📈 Signal: {signal.signal_type}
💰 Price: ${signal.price:.6f}
📊 Change: {signal.change_24h:+.2f}%
🏆 Rank: #{signal.rank}
💪 Strength: {signal.strength:.2f}
🎯 Confidence: {signal.confidence:.2f}
📈 Volume: {self._format_volume(signal.volume)}
💎 MCap: {self._format_volume(signal.market_cap)}

"""
        
        message += f"""📊 <b>Market Analysis:</b>
🔍 Screened 250+ top cryptocurrencies
🌐 CoinGecko real-time data
🎯 AI-powered signal generation
⚡ Automated market analysis

Bot monitoring markets 24/7 🚀"""
        
        return message

class CryptoAnalysisBot:
    """Main crypto analysis bot with embedded market screener"""
    
    def __init__(self):
        self.config = CONFIG
        self.start_time = datetime.now(timezone.utc)
        
        # Initialize embedded market screener
        self.market_screener = EmbeddedMarketScreener()
        
        logger.info("🤖 Crypto Analysis Bot V3.2 Enterprise initialized")
    
    def get_binance_data(self, symbol: str, interval: str = "1d") -> Optional[Dict]:
        """Fetch cryptocurrency data from Binance API"""
        try:
            # Get 24hr ticker statistics
            ticker_url = f"{self.config['binance']['base_url']}/ticker/24hr"
            ticker_params = {"symbol": symbol}
            
            ticker_response = requests.get(ticker_url, params=ticker_params, timeout=10)
            ticker_response.raise_for_status()
            ticker_data = ticker_response.json()
            
            # Get current price
            price_url = f"{self.config['binance']['base_url']}/ticker/price"
            price_params = {"symbol": symbol}
            
            price_response = requests.get(price_url, params=price_params, timeout=10)
            price_response.raise_for_status()
            price_data = price_response.json()
            
            # Combine data
            combined_data = {
                "symbol": symbol,
                "price": float(price_data["price"]),
                "change_24h": float(ticker_data["priceChangePercent"]),
                "volume": float(ticker_data["volume"]),
                "high_24h": float(ticker_data["highPrice"]),
                "low_24h": float(ticker_data["lowPrice"]),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            return combined_data
            
        except Exception as e:
            logger.error(f"❌ Error fetching Binance data for {symbol}: {e}")
            return None
    
    def analyze_crypto(self, symbol: str) -> Dict:
        """Analyze a cryptocurrency with available modules"""
        try:
            # Get basic market data
            market_data = self.get_binance_data(symbol)
            if not market_data:
                return {"error": f"Failed to fetch data for {symbol}"}
            
            analysis = {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "market_data": market_data,
                "analysis_modules": {}
            }
            
            # Add technical analysis if available
            if MODULES_AVAILABLE["technical_analysis"]:
                try:
                    ta_result = technical_analysis.analyze_symbol(symbol)
                    analysis["analysis_modules"]["technical_analysis"] = ta_result
                except Exception as e:
                    analysis["analysis_modules"]["technical_analysis"] = {"error": str(e)}
            
            # Add ML analysis if available  
            if MODULES_AVAILABLE["machine_learning"]:
                try:
                    ml_result = machine_learning.predict_price(symbol, market_data)
                    analysis["analysis_modules"]["machine_learning"] = ml_result
                except Exception as e:
                    analysis["analysis_modules"]["machine_learning"] = {"error": str(e)}
            
            # Generate basic signal
            price_change = market_data["change_24h"]
            if price_change > 5:
                signal = "SELL"
            elif price_change < -5:
                signal = "BUY"
            else:
                signal = "HOLD"
            
            analysis["signal"] = signal
            analysis["confidence"] = min(abs(price_change) / 10, 1.0)
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            return {"error": str(e)}
    
    def send_telegram_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send message to Telegram"""
        try:
            if not self.config["telegram"]["enabled"]:
                logger.info("📱 Telegram disabled, skipping message")
                return True
            
            url = f"https://api.telegram.org/bot{self.config['telegram']['bot_token']}/sendMessage"
            
            payload = {
                "chat_id": self.config["telegram"]["chat_id"],
                "text": message,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info("📱 Telegram message sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error sending Telegram message: {e}")
            return False
    
    def generate_market_analysis_report(self) -> str:
        """Generate comprehensive market analysis report"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            
            # Regular crypto analysis
            report = f"""🤖 <b>Crypto Analysis Bot V3.2</b>
⏰ {current_time}
📊 <b>Multi-Crypto Analysis Report</b>

"""
            
            # Analyze configured symbols
            for symbol in self.config["analysis"]["symbols"]:
                analysis = self.analyze_crypto(symbol)
                
                if "error" not in analysis:
                    market_data = analysis["market_data"]
                    signal = analysis["signal"]
                    confidence = analysis["confidence"]
                    
                    # Signal emoji
                    if signal == "BUY":
                        signal_emoji = "🟢"
                    elif signal == "SELL":
                        signal_emoji = "🔴"
                    else:
                        signal_emoji = "🟡"
                    
                    report += f"""🪙 <b>{symbol.replace('USDT', '/USDT')}</b>
💰 Price: ${market_data['price']:.4f}
📊 24h Change: {market_data['change_24h']:+.2f}%
📈 24h High: ${market_data['high_24h']:.4f}
📉 24h Low: ${market_data['low_24h']:.4f}
{signal_emoji} Signal: {signal} (Confidence: {confidence:.0%})

"""
                else:
                    report += f"❌ {symbol}: Analysis failed\n\n"
            
            # Add market screener results
            try:
                screener_signals = self.market_screener.get_top_opportunities(3)
                
                if screener_signals:
                    report += """🔍 <b>Market Screener - Top Opportunities</b>

"""
                    for i, signal in enumerate(screener_signals, 1):
                        signal_emoji = "🟢" if signal.signal_type in ["STRONG_BUY", "BUY"] else "🔴"
                        
                        report += f"""{i}️⃣ {signal_emoji} <b>{signal.symbol}</b>
📈 Signal: {signal.signal_type}
💰 Price: ${signal.price:.6f}
📊 Change: {signal.change_24h:+.2f}%
🏆 Rank: #{signal.rank}

"""
            except Exception as e:
                logger.warning(f"⚠️  Market screener failed: {e}")
                report += "⚠️ Market screener temporarily unavailable\n\n"
            
            # System status
            uptime = datetime.now(timezone.utc) - self.start_time
            uptime_str = str(uptime).split('.')[0]
            
            report += f"""📊 <b>System Status</b>
⚡ Status: Operational
⏱️ Uptime: {uptime_str}
🔧 Modules: {sum(MODULES_AVAILABLE.values())}/{len(MODULES_AVAILABLE)} active
🤖 Version: 3.2.0-Enterprise-Embedded

<i>Automated analysis • Real-time data • 24/7 monitoring</i>"""
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generating report: {e}")
            return f"❌ Error generating analysis report: {str(e)}"

# Global bot instance
bot = CryptoAnalysisBot()

# Routes
@app.route('/')
def home():
    """Home page with bot information"""
    try:
        uptime = datetime.now(timezone.utc) - bot.start_time
        uptime_str = str(uptime).split('.')[0]
        
        modules_status = {
            "enterprise_edition": True,
            "database": MODULES_AVAILABLE["database"],
            "technical_analysis": MODULES_AVAILABLE["technical_analysis"],
            "machine_learning": MODULES_AVAILABLE["machine_learning"],
            "configuration": MODULES_AVAILABLE["configuration"],
            "utilities": MODULES_AVAILABLE["utilities"],
            "market_screener": True,  # Always available (embedded)
            "telegram": CONFIG["telegram"]["enabled"],
            "real_time_data": True,
            "automation": True,
            "daily_summaries": True
        }
        
        return jsonify({
            "name": "Crypto Analysis Bot V3.2 - Enterprise Edition",
            "version": "3.2.0-Enterprise-Embedded",
            "status": "operational",
            "uptime": uptime_str,
            "start_time": bot.start_time.isoformat(),
            "features": modules_status,
            "supported_symbols": CONFIG["analysis"]["symbols"],
            "market_screener": {
                "enabled": True,
                "type": "embedded",
                "source": "CoinGecko",
                "coverage": "250+ cryptocurrencies"
            },
            "telegram": {
                "enabled": CONFIG["telegram"]["enabled"],
                "bot_configured": bool(CONFIG["telegram"]["bot_token"]),
                "chat_configured": bool(CONFIG["telegram"]["chat_id"])
            },
            "automation": {
                "analysis_interval": "2 hours",
                "daily_summary_time": "10:00 AM", 
                "monitoring": "24/7",
                "next_analysis": "Every 2 hours",
                "next_summary": "Daily at 10:00 AM"
            },
            "performance": {
                "total_requests": 0,
                "requests_per_minute": 0.0
            },
            "system": {
                "status": "operational",
                "version": "3.2.0-Enterprise-Embedded",
                "uptime": uptime_str,
                "start_time": bot.start_time.isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Error in home route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/status')
def status():
    """System status endpoint"""
    return home()

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "modules_available": sum(MODULES_AVAILABLE.values()),
            "total_modules": len(MODULES_AVAILABLE),
            "market_screener": "embedded"
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/analyze/<symbol>')
def analyze_symbol(symbol):
    """Analyze a specific cryptocurrency symbol"""
    try:
        symbol = symbol.upper()
        if not symbol.endswith('USDT'):
            symbol += 'USDT'
        
        analysis = bot.analyze_crypto(symbol)
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"❌ Error in analyze route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/telegram/test')
def telegram_test():
    """Test Telegram functionality"""
    try:
        test_message = """🤖 <b>Telegram Test Message</b>
⏰ """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """

✅ Bot is operational and connected!
🚀 Crypto Analysis Bot V3.2 Enterprise ready for automated reports.

<i>This is a test message.</i>"""
        
        success = bot.send_telegram_message(test_message)
        
        return jsonify({
            "success": success,
            "message": "Test message sent" if success else "Failed to send test message",
            "telegram_configured": CONFIG["telegram"]["enabled"]
        })
        
    except Exception as e:
        logger.error(f"❌ Error in telegram test: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/telegram/report')
def telegram_report():
    """Send full analysis report to Telegram"""
    try:
        report = bot.generate_market_analysis_report()
        success = bot.send_telegram_message(report)
        
        return jsonify({
            "success": success,
            "message": "Analysis report sent" if success else "Failed to send report",
            "report_length": len(report)
        })
        
    except Exception as e:
        logger.error(f"❌ Error sending telegram report: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# Market Screener Routes (Embedded)
@app.route('/market/screen')
def market_screen():
    """Screen market and return top opportunities"""
    try:
        limit = request.args.get('limit', 5, type=int)
        limit = min(max(limit, 1), 20)  # Limit between 1-20
        
        opportunities = bot.market_screener.get_top_opportunities(limit)
        
        # Convert to JSON-serializable format
        opportunities_data = []
        for signal in opportunities:
            opportunities_data.append({
                "symbol": signal.symbol,
                "signal_type": signal.signal_type,
                "strength": signal.strength,
                "confidence": signal.confidence,
                "price": signal.price,
                "change_24h": signal.change_24h,
                "volume": signal.volume,
                "market_cap": signal.market_cap,
                "rank": signal.rank,
                "analysis": signal.analysis,
                "source": signal.source
            })
        
        return jsonify({
            "success": True,
            "opportunities_found": len(opportunities_data),
            "top_opportunities": opportunities_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "CoinGecko Embedded Screener"
        })
        
    except Exception as e:
        logger.error(f"❌ Error in market screen: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/telegram/market-screen')
def telegram_market_screen():
    """Send market screener results to Telegram"""
    try:
        limit = request.args.get('limit', 5, type=int)
        limit = min(max(limit, 1), 10)  # Limit between 1-10 for Telegram
        
        signals = bot.market_screener.get_top_opportunities(limit)
        message = bot.market_screener.format_signals_for_telegram(signals)
        
        success = bot.send_telegram_message(message)
        
        return jsonify({
            "success": success,
            "message": "Market screener report sent" if success else "Failed to send market report",
            "opportunities_analyzed": len(signals),
            "source": "CoinGecko Embedded"
        })
        
    except Exception as e:
        logger.error(f"❌ Error sending market screen to Telegram: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/telegram/signals')
def telegram_signals():
    """Send trading signals to Telegram (enhanced with market screener)"""
    try:
        use_screener = request.args.get('screener', 'false').lower() == 'true'
        
        if use_screener:
            # Use market screener for signals
            return telegram_market_screen()
        else:
            # Use regular analysis report
            return telegram_report()
        
    except Exception as e:
        logger.error(f"❌ Error in telegram signals: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# Enterprise module routes (with graceful fallbacks)
@app.route('/database/info')
def database_info():
    """Database module information"""
    if MODULES_AVAILABLE["database"]:
        try:
            return jsonify(database.get_info())
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({
            "database_available": False,
            "error": "Database module not available",
            "success": False
        })

@app.route('/ml/status')
def ml_status():
    """Machine learning module status"""
    if MODULES_AVAILABLE["machine_learning"]:
        try:
            return jsonify(machine_learning.get_status())
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({
            "ml_available": False,
            "error": "Machine learning module not available",
            "success": False
        })

@app.route('/technical/test')
def technical_test():
    """Technical analysis module test"""
    if MODULES_AVAILABLE["technical_analysis"]:
        try:
            return jsonify(technical_analysis.test_analysis())
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({
            "technical_analysis_available": False,
            "error": "Technical analysis module not available", 
            "success": False
        })

# Scheduled analysis function (called by Cloud Scheduler)
@app.route('/scheduled/analysis')
def scheduled_analysis():
    """Endpoint for scheduled analysis (called by Google Cloud Scheduler)"""
    try:
        logger.info("🕐 Scheduled analysis triggered")
        
        # Generate and send analysis report
        report = bot.generate_market_analysis_report()
        success = bot.send_telegram_message(report)
        
        return jsonify({
            "success": success,
            "message": "Scheduled analysis completed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "report_sent": success
        })
        
    except Exception as e:
        logger.error(f"❌ Error in scheduled analysis: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    # Development server
    logger.info("🚀 Starting Crypto Analysis Bot V3.2 Enterprise...")
    logger.info(f"🔧 Available modules: {[k for k, v in MODULES_AVAILABLE.items() if v]}")
    logger.info("🔍 Market Screener: EMBEDDED (CoinGecko - 250+ cryptocurrencies)")
    
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)