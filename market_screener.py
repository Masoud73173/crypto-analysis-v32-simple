#!/usr/bin/env python3
"""
Market Screener for Crypto Analysis Bot V3.2 - DEBUG VERSION
===========================================
CoinEx API integration with enhanced debugging for signal generation
"""

import os
import json
import time
import hmac
import hashlib
import requests
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    analysis: str = ""
    source: str = "Market Screener"

class CoinExAPI:
    """CoinEx API client with authentication and market data"""
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        # API Configuration
        self.api_key = api_key or os.getenv("COINEX_API_KEY", "YOUR_API_KEY_HERE")
        self.secret_key = secret_key or os.getenv("COINEX_SECRET_KEY", "YOUR_SECRET_HERE")
        
        # API URLs
        self.base_url = "https://api.coinex.com/v2"
        self.timeout = 30
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        logger.info("🔌 CoinEx API client initialized")
    
    def _generate_signature(self, method: str, endpoint: str, params: Dict = None) -> str:
        """Generate API signature for authenticated requests"""
        if not params:
            params = {}
        
        # Create query string
        query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        
        # Create message to sign
        message = f"{method}{endpoint}{query_string}"
        
        # Generate signature
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _rate_limit(self):
        """Apply rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, 
                     authenticated: bool = False) -> Optional[Dict]:
        """Make HTTP request to CoinEx API"""
        try:
            # Rate limiting
            self._rate_limit()
            
            # Prepare URL
            url = f"{self.base_url}{endpoint}"
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "CryptoBot/3.2"
            }
            
            # Add authentication if required
            if authenticated and self.api_key != "YOUR_API_KEY_HERE":
                timestamp = str(int(time.time() * 1000))
                headers["X-COINEX-KEY"] = self.api_key
                headers["X-COINEX-TIMESTAMP"] = timestamp
                
                # Add signature for authenticated requests
                if params:
                    signature = self._generate_signature(method, endpoint, params)
                    headers["X-COINEX-SIGNATURE"] = signature
            
            # Make request
            if method.upper() == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            elif method.upper() == "POST":
                response = requests.post(url, json=params, headers=headers, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Check CoinEx API response
            if data.get("code") != 0:
                logger.error(f"❌ CoinEx API error: {data.get('message', 'Unknown error')}")
                return None
            
            return data.get("data")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ HTTP request error: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return None
    
    def get_market_tickers(self, symbols: List[str] = None) -> List[Dict]:
        """Get ticker data for specified symbols or all symbols"""
        try:
            endpoint = "/spot/ticker"
            
            # Get all tickers
            data = self._make_request("GET", endpoint, {})
            if data:
                logger.info(f"📈 Retrieved ticker data for {len(data)} symbols")
                return data
            else:
                logger.warning("⚠️  No ticker data received")
                return []
                
        except Exception as e:
            logger.error(f"❌ Error getting market tickers: {e}")
            return []

class MarketScreener:
    """Advanced market screener for cryptocurrency analysis with enhanced debugging"""
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        self.coinex = CoinExAPI(api_key, secret_key)
        
        # MORE RELAXED screening parameters for debugging
        self.min_volume_24h = 10000       # Minimum 10K USDT volume (much lower)
        self.min_price = 0.000001         # Minimum price 
        self.max_price = 100000           # Maximum price
        
        # VERY SENSITIVE signal thresholds for debugging
        self.strong_buy_threshold = -5    # Price drop > 5% = strong buy
        self.buy_threshold = -2           # Price drop > 2% = buy  
        self.sell_threshold = 5           # Price rise > 5% = sell
        self.strong_sell_threshold = 10   # Price rise > 10% = strong sell
        
        logger.info("🔍 Market Screener initialized with DEBUG settings")
    
    def _passes_basic_filters(self, price: float, volume_24h: float) -> bool:
        """Check if crypto passes basic screening filters"""
        result = (
            self.min_price <= price <= self.max_price and
            volume_24h >= self.min_volume_24h
        )
        return result
    
    def _generate_signal(self, symbol: str, price: float, change_24h: float, 
                        volume: float, volume_value: float) -> Optional[CryptoSignal]:
        """Generate trading signal based on price action and volume"""
        try:
            # Determine signal type based on price change
            if change_24h <= self.strong_buy_threshold:
                signal_type = "STRONG_BUY"
                strength = min(abs(change_24h) / 20, 1.0)
                confidence = 0.8
            elif change_24h <= self.buy_threshold:
                signal_type = "BUY"
                strength = min(abs(change_24h) / 10, 1.0)
                confidence = 0.6
            elif change_24h >= self.strong_sell_threshold:
                signal_type = "STRONG_SELL"
                strength = min(change_24h / 25, 1.0)
                confidence = 0.7
            elif change_24h >= self.sell_threshold:
                signal_type = "SELL"
                strength = min(change_24h / 15, 1.0)
                confidence = 0.6
            else:
                # No strong signal - but log why
                return None
            
            # Adjust confidence based on volume
            volume_multiplier = min(volume_value / self.min_volume_24h, 3.0)
            confidence = min(confidence * (0.5 + volume_multiplier * 0.1), 1.0)
            
            # Generate analysis text
            analysis = f"24h: {change_24h:+.2f}%, Vol: {self._format_volume(volume_value)}"
            
            return CryptoSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=round(strength, 2),
                confidence=round(confidence, 2),
                price=price,
                change_24h=change_24h,
                volume=volume_value,
                analysis=analysis,
                source="CoinEx Market Screener"
            )
            
        except Exception as e:
            logger.error(f"❌ Error generating signal for {symbol}: {e}")
            return None
    
    def _format_volume(self, volume: float) -> str:
        """Format volume with appropriate units"""
        if volume >= 1_000_000:
            return f"{volume/1_000_000:.1f}M"
        elif volume >= 1_000:
            return f"{volume/1_000:.1f}K"
        else:
            return f"{volume:.0f}"
    
    def get_top_opportunities(self, limit: int = 5) -> List[CryptoSignal]:
        """Screen market and return top trading opportunities with detailed debugging"""
        try:
            logger.info("🔍 Starting market screening...")
            
            # Get all tickers
            all_tickers = self.coinex.get_market_tickers()
            if not all_tickers:
                logger.error("❌ Failed to get ticker data")
                return []
            
            logger.info(f"📊 Analyzing {len(all_tickers)} total markets...")
            
            # Detailed analysis with debugging
            signals = []
            usdt_count = 0
            processed_count = 0
            filter_passed = 0
            debug_examples = []
            
            for ticker in all_tickers:
                try:
                    symbol = ticker.get("market", "")
                    
                    # Filter USDT pairs
                    if not symbol.endswith("USDT"):
                        continue
                    
                    usdt_count += 1
                    
                    # Extract and debug data
                    price = float(ticker.get("last", 0))
                    open_price = float(ticker.get("open", 0))
                    close_price = float(ticker.get("close", 0))
                    volume = float(ticker.get("volume", 0))
                    volume_value = float(ticker.get("value", 0))  # Volume in USDT
                    
                    # Calculate 24h change percentage
                    if open_price > 0:
                        change_24h = ((close_price - open_price) / open_price) * 100
                    else:
                        change_24h = 0.0
                    
                    processed_count += 1
                    
                    # Store first 10 examples for debugging
                    if len(debug_examples) < 10:
                        debug_examples.append({
                            'symbol': symbol,
                            'price': price,
                            'open': open_price,
                            'close': close_price,
                            'change_24h': change_24h,
                            'volume_value': volume_value
                        })
                    
                    # Apply basic filters
                    if not self._passes_basic_filters(price, volume_value):
                        continue
                    
                    filter_passed += 1
                    
                    # Generate signal
                    signal = self._generate_signal(symbol, price, change_24h, volume, volume_value)
                    
                    if signal:
                        signals.append(signal)
                        
                except (ValueError, TypeError) as e:
                    logger.warning(f"⚠️  Error processing ticker for {ticker.get('market', 'unknown')}: {e}")
                    continue
            
            # Debug logging
            logger.info(f"🔍 DEBUG ANALYSIS:")
            logger.info(f"   📊 Total markets: {len(all_tickers)}")
            logger.info(f"   🎯 USDT pairs: {usdt_count}")
            logger.info(f"   ⚙️  Processed: {processed_count}")
            logger.info(f"   ✅ Passed filters: {filter_passed}")
            logger.info(f"   🚀 Generated signals: {len(signals)}")
            
            # Show examples
            logger.info(f"🔍 Example data (first 10 USDT pairs):")
            for example in debug_examples:
                logger.info(f"   {example['symbol']}: Last=${example['price']:.6f}, Open=${example['open']:.6f}, Change={example['change_24h']:+.2f}%, Vol=${example['volume_value']:.0f}")
            
            # Sort by signal strength and confidence
            signals.sort(key=lambda x: (x.strength * x.confidence), reverse=True)
            
            # Return top opportunities
            top_signals = signals[:limit]
            
            logger.info(f"🏆 Selected top {len(top_signals)} opportunities")
            
            # Show what we found
            if top_signals:
                logger.info("🎯 Top signals:")
                for signal in top_signals:
                    logger.info(f"   {signal.symbol}: {signal.signal_type} (Strength: {signal.strength}, Change: {signal.change_24h:+.2f}%)")
            
            return top_signals
            
        except Exception as e:
            logger.error(f"❌ Error getting top opportunities: {e}")
            return []
    
    def format_signals_for_telegram(self, signals: List[CryptoSignal]) -> str:
        """Format trading signals for Telegram message"""
        if not signals:
            return "📊 No strong trading opportunities found in current market conditions."
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"""🔍 <b>Market Screener Report</b>
⏰ {current_time}
📊 <b>Top {len(signals)} Trading Opportunities</b>

"""
        
        # Add ranking emojis
        rank_emojis = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        
        for i, signal in enumerate(signals):
            rank_emoji = rank_emojis[i] if i < len(rank_emojis) else f"{i+1}️⃣"
            
            # Signal emoji
            if signal.signal_type in ["STRONG_BUY", "BUY"]:
                signal_emoji = "🟢"
            elif signal.signal_type in ["STRONG_SELL", "SELL"]:
                signal_emoji = "🔴"
            else:
                signal_emoji = "🟡"
            
            message += f"""{rank_emoji} {signal_emoji} <b>{signal.symbol}</b>
📈 Signal: {signal.signal_type}
💰 Price: ${signal.price:.6f}
📊 Change: {signal.change_24h:+.2f}%
💪 Strength: {signal.strength:.2f}
🎯 Confidence: {signal.confidence:.2f}
📈 Volume: {self._format_volume(signal.volume)}

"""
        
        message += f"""📊 <b>Market Analysis:</b>
🔍 Screened 1300+ USDT pairs
🎯 Top performers selected
⚡ Real-time CoinEx data

Bot monitoring markets 24/7 🚀"""
        
        return message

# Convenience functions for easy integration
def screen_market(api_key: str = None, secret_key: str = None, limit: int = 5) -> List[CryptoSignal]:
    """Quick function to screen market and get top opportunities"""
    screener = MarketScreener(api_key, secret_key)
    return screener.get_top_opportunities(limit)

def get_market_screener_telegram_message(api_key: str = None, secret_key: str = None, limit: int = 5) -> str:
    """Quick function to get formatted Telegram message"""
    screener = MarketScreener(api_key, secret_key)
    signals = screener.get_top_opportunities(limit)
    return screener.format_signals_for_telegram(signals)

if __name__ == "__main__":
    # Test market screener with enhanced debugging
    print("🔍 Market Screener DEBUG Test")
    print("=" * 50)
    
    # Initialize screener
    screener = MarketScreener()
    
    print("📊 Note: Replace API keys in environment variables or parameters")
    print("🔑 COINEX_API_KEY=your_actual_api_key")
    print("🔑 COINEX_SECRET_KEY=your_actual_secret_key")
    
    # Test with demo data if API keys not configured
    if screener.coinex.api_key == "YOUR_API_KEY_HERE":
        print("\n⚠️  API keys not configured - using demo mode")
        
        # Create demo signals
        demo_signals = [
            CryptoSignal(
                symbol="ADAUSDT",
                signal_type="STRONG_BUY",
                strength=0.85,
                confidence=0.78,
                price=0.45,
                change_24h=-12.5,
                volume=2500000,
                analysis="24h: -12.50%, Vol: 2.5M"
            ),
            CryptoSignal(
                symbol="DOTUSDT", 
                signal_type="BUY",
                strength=0.72,
                confidence=0.65,
                price=6.23,
                change_24h=-8.2,
                volume=1800000,
                analysis="24h: -8.20%, Vol: 1.8M"
            )
        ]
        
        message = screener.format_signals_for_telegram(demo_signals)
        print("\n📱 Demo Telegram Message:")
        print(message.replace("<b>", "").replace("</b>", ""))
        
    else:
        print("\n🔍 Testing with real API (DEBUG MODE)...")
        try:
            signals = screener.get_top_opportunities(5)
            if signals:
                print(f"\n✅ SUCCESS! Found {len(signals)} opportunities:")
                for i, signal in enumerate(signals, 1):
                    print(f"   {i}. {signal.symbol}: {signal.signal_type} (Change: {signal.change_24h:+.2f}%, Strength: {signal.strength:.2f})")
            else:
                print("\n❌ No opportunities found - check debug logs above")
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    print("\n✅ Market Screener DEBUG module ready!")
    print("🚀 Check debug logs for detailed analysis!")