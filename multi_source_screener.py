#!/usr/bin/env python3
"""
CoinMarketCap & CoinGecko Market Screener for Crypto Analysis Bot V3.2
======================================================================
Multi-source cryptocurrency analysis using CMC and CoinGecko APIs
"""

import os
import json
import time
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
    rank: Optional[int] = None
    analysis: str = ""
    source: str = "Multi-Source Screener"

class CoinMarketCapAPI:
    """CoinMarketCap API client for cryptocurrency data"""
    
    def __init__(self, api_key: str = None):
        # API Configuration
        self.api_key = api_key or os.getenv("CMC_API_KEY", "")
        
        # API URLs
        self.base_url = "https://pro-api.coinmarketcap.com/v1"
        self.sandbox_url = "https://sandbox-api.coinmarketcap.com/v1"
        self.timeout = 15
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 500ms between requests (free tier: 333 calls/day)
        
        logger.info("🔌 CoinMarketCap API client initialized")
    
    def _rate_limit(self):
        """Apply rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make HTTP request to CoinMarketCap API"""
        try:
            # Rate limiting
            self._rate_limit()
            
            # Choose URL based on API key availability
            if self.api_key:
                url = f"{self.base_url}{endpoint}"
                headers = {
                    "X-CMC_PRO_API_KEY": self.api_key,
                    "Accept": "application/json"
                }
            else:
                # Use sandbox for testing without API key
                url = f"{self.sandbox_url}{endpoint}"
                headers = {"Accept": "application/json"}
            
            # Make request
            response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Check CMC response status
            if data.get("status", {}).get("error_code") != 0:
                logger.error(f"❌ CMC API error: {data.get('status', {}).get('error_message', 'Unknown error')}")
                return None
            
            return data.get("data", [])
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ CMC HTTP request error: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ CMC JSON decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ CMC unexpected error: {e}")
            return None
    
    def get_listings_latest(self, limit: int = 200) -> List[Dict]:
        """Get latest cryptocurrency listings"""
        try:
            endpoint = "/cryptocurrency/listings/latest"
            params = {
                "limit": limit,
                "sort": "market_cap",
                "cryptocurrency_type": "coins"
            }
            
            data = self._make_request(endpoint, params)
            
            if data:
                logger.info(f"📈 Retrieved CMC data for {len(data)} cryptocurrencies")
                return data
            else:
                logger.warning("⚠️  No CMC data received")
                return []
                
        except Exception as e:
            logger.error(f"❌ Error getting CMC listings: {e}")
            return []

class CoinGeckoAPI:
    """CoinGecko API client for cryptocurrency data"""
    
    def __init__(self):
        # API URLs (no key needed for public endpoints)
        self.base_url = "https://api.coingecko.com/api/v3"
        self.timeout = 15
        
        # Rate limiting (stricter for free tier)
        self.last_request_time = 0
        self.min_request_interval = 1.2  # 1.2 seconds between requests (50 calls/minute limit)
        
        logger.info("🔌 CoinGecko API client initialized")
    
    def _rate_limit(self):
        """Apply rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make HTTP request to CoinGecko API"""
        try:
            # Rate limiting
            self._rate_limit()
            
            # Prepare URL
            url = f"{self.base_url}{endpoint}"
            
            # Prepare headers
            headers = {
                "Accept": "application/json",
                "User-Agent": "CryptoBot/3.2"
            }
            
            # Make request
            response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ CoinGecko HTTP request error: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ CoinGecko JSON decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ CoinGecko unexpected error: {e}")
            return None
    
    def get_coins_markets(self, vs_currency: str = "usd", per_page: int = 250, page: int = 1) -> List[Dict]:
        """Get coins market data"""
        try:
            endpoint = "/coins/markets"
            params = {
                "vs_currency": vs_currency,
                "order": "market_cap_desc",
                "per_page": per_page,
                "page": page,
                "sparkline": False,
                "price_change_percentage": "24h"
            }
            
            data = self._make_request(endpoint, params)
            
            if data:
                logger.info(f"📈 Retrieved CoinGecko data for {len(data)} cryptocurrencies")
                return data
            else:
                logger.warning("⚠️  No CoinGecko data received")
                return []
                
        except Exception as e:
            logger.error(f"❌ Error getting CoinGecko markets: {e}")
            return []

class MultiSourceMarketScreener:
    """Advanced market screener using multiple cryptocurrency data sources"""
    
    def __init__(self, cmc_api_key: str = None):
        self.cmc = CoinMarketCapAPI(cmc_api_key)
        self.coingecko = CoinGeckoAPI()
        
        # RELAXED screening parameters for comprehensive analysis
        self.min_volume_24h = 100000      # Minimum 100K USD volume
        self.min_market_cap = 1000000     # Minimum 1M USD market cap
        self.max_rank = 500               # Top 500 coins by market cap
        
        # SENSITIVE signal thresholds for opportunities
        self.strong_buy_threshold = -5    # Price drop > 5% = strong buy
        self.buy_threshold = -2           # Price drop > 2% = buy  
        self.sell_threshold = 5           # Price rise > 5% = sell
        self.strong_sell_threshold = 10   # Price rise > 10% = strong sell
        
        logger.info("🔍 Multi-Source Market Screener initialized")
    
    def _normalize_cmc_data(self, cmc_data: List[Dict]) -> List[Dict]:
        """Normalize CoinMarketCap data to standard format"""
        normalized = []
        
        for coin in cmc_data:
            try:
                quote = coin.get("quote", {}).get("USD", {})
                
                normalized_coin = {
                    "id": coin.get("id"),
                    "symbol": coin.get("symbol"),
                    "name": coin.get("name"),
                    "rank": coin.get("cmc_rank"),
                    "price": float(quote.get("price", 0)),
                    "change_24h": float(quote.get("percent_change_24h", 0)),
                    "volume_24h": float(quote.get("volume_24h", 0)),
                    "market_cap": float(quote.get("market_cap", 0)),
                    "source": "CoinMarketCap"
                }
                
                normalized.append(normalized_coin)
                
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"⚠️  Error normalizing CMC data for {coin.get('symbol', 'unknown')}: {e}")
                continue
        
        return normalized
    
    def _normalize_coingecko_data(self, cg_data: List[Dict]) -> List[Dict]:
        """Normalize CoinGecko data to standard format"""
        normalized = []
        
        for coin in cg_data:
            try:
                normalized_coin = {
                    "id": coin.get("id"),
                    "symbol": coin.get("symbol", "").upper(),
                    "name": coin.get("name"),
                    "rank": coin.get("market_cap_rank"),
                    "price": float(coin.get("current_price", 0)),
                    "change_24h": float(coin.get("price_change_percentage_24h", 0)),
                    "volume_24h": float(coin.get("total_volume", 0)),
                    "market_cap": float(coin.get("market_cap", 0)),
                    "source": "CoinGecko"
                }
                
                normalized.append(normalized_coin)
                
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"⚠️  Error normalizing CoinGecko data for {coin.get('symbol', 'unknown')}: {e}")
                continue
        
        return normalized
    
    def _passes_basic_filters(self, coin: Dict) -> bool:
        """Check if cryptocurrency passes basic screening filters"""
        return (
            coin.get("rank", 999) <= self.max_rank and
            coin.get("volume_24h", 0) >= self.min_volume_24h and
            coin.get("market_cap", 0) >= self.min_market_cap and
            coin.get("price", 0) > 0
        )
    
    def _generate_signal(self, coin: Dict) -> Optional[CryptoSignal]:
        """Generate trading signal based on price action and fundamentals"""
        try:
            symbol = coin.get("symbol", "")
            price = coin.get("price", 0)
            change_24h = coin.get("change_24h", 0)
            volume_24h = coin.get("volume_24h", 0)
            market_cap = coin.get("market_cap", 0)
            rank = coin.get("rank", 999)
            
            # Determine signal type based on price change
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
                # No strong signal
                return None
            
            # Adjust confidence based on market cap rank (lower rank = higher confidence)
            rank_multiplier = max(0.5, 1.0 - (rank / 500))
            confidence = min(confidence * (0.7 + rank_multiplier * 0.3), 1.0)
            
            # Adjust confidence based on volume
            volume_multiplier = min(volume_24h / self.min_volume_24h, 5.0)
            confidence = min(confidence * (0.6 + volume_multiplier * 0.1), 1.0)
            
            # Generate analysis text
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
                source=coin.get("source", "Multi-Source")
            )
            
        except Exception as e:
            logger.error(f"❌ Error generating signal for {coin.get('symbol', 'unknown')}: {e}")
            return None
    
    def _format_volume(self, volume: float) -> str:
        """Format volume/market cap with appropriate units"""
        if volume >= 1_000_000_000:
            return f"{volume/1_000_000_000:.1f}B"
        elif volume >= 1_000_000:
            return f"{volume/1_000_000:.1f}M"
        elif volume >= 1_000:
            return f"{volume/1_000:.1f}K"
        else:
            return f"{volume:.0f}"
    
    def get_top_opportunities(self, limit: int = 5, use_both_sources: bool = True) -> List[CryptoSignal]:
        """Screen market using multiple sources and return top trading opportunities"""
        try:
            logger.info("🔍 Starting multi-source market screening...")
            
            all_coins = []
            
            # Get data from CoinGecko (more reliable for free tier)
            logger.info("📊 Fetching data from CoinGecko...")
            cg_data = self.coingecko.get_coins_markets(per_page=250, page=1)
            if cg_data:
                normalized_cg = self._normalize_coingecko_data(cg_data)
                all_coins.extend(normalized_cg)
                logger.info(f"✅ Added {len(normalized_cg)} coins from CoinGecko")
            
            # Optionally get data from CoinMarketCap (if API key available)
            if use_both_sources and self.cmc.api_key:
                logger.info("📊 Fetching data from CoinMarketCap...")
                cmc_data = self.cmc.get_listings_latest(limit=200)
                if cmc_data:
                    normalized_cmc = self._normalize_cmc_data(cmc_data)
                    # Merge with CoinGecko data (prefer CMC for duplicates)
                    cmc_symbols = {coin["symbol"] for coin in normalized_cmc}
                    all_coins = [coin for coin in all_coins if coin["symbol"] not in cmc_symbols]
                    all_coins.extend(normalized_cmc)
                    logger.info(f"✅ Added {len(normalized_cmc)} coins from CoinMarketCap")
            
            if not all_coins:
                logger.error("❌ Failed to get data from any source")
                return []
            
            logger.info(f"📊 Analyzing {len(all_coins)} total cryptocurrencies...")
            
            # Screen and generate signals
            signals = []
            filter_passed = 0
            debug_examples = []
            
            for coin in all_coins:
                try:
                    # Store first 10 examples for debugging
                    if len(debug_examples) < 10:
                        debug_examples.append({
                            'symbol': coin.get('symbol'),
                            'price': coin.get('price'),
                            'change_24h': coin.get('change_24h'),
                            'rank': coin.get('rank'),
                            'volume': coin.get('volume_24h'),
                            'source': coin.get('source')
                        })
                    
                    # Apply basic filters
                    if not self._passes_basic_filters(coin):
                        continue
                    
                    filter_passed += 1
                    
                    # Generate signal
                    signal = self._generate_signal(coin)
                    
                    if signal:
                        signals.append(signal)
                        
                except Exception as e:
                    logger.warning(f"⚠️  Error processing coin {coin.get('symbol', 'unknown')}: {e}")
                    continue
            
            # Debug logging
            logger.info(f"🔍 MULTI-SOURCE DEBUG ANALYSIS:")
            logger.info(f"   📊 Total cryptocurrencies: {len(all_coins)}")
            logger.info(f"   ✅ Passed filters: {filter_passed}")
            logger.info(f"   🚀 Generated signals: {len(signals)}")
            
            # Show examples
            logger.info(f"🔍 Example data (first 10 coins):")
            for example in debug_examples:
                logger.info(f"   {example['symbol']}: ${example['price']:.6f}, {example['change_24h']:+.2f}%, Rank #{example['rank']}, Vol ${self._format_volume(example['volume'])} ({example['source']})")
            
            # Sort by signal strength, confidence, and market cap rank
            signals.sort(key=lambda x: (x.strength * x.confidence * (1.0 - (x.rank or 500) / 1000)), reverse=True)
            
            # Return top opportunities
            top_signals = signals[:limit]
            
            logger.info(f"🏆 Selected top {len(top_signals)} opportunities")
            
            # Show what we found
            if top_signals:
                logger.info("🎯 Top signals:")
                for signal in top_signals:
                    logger.info(f"   {signal.symbol}: {signal.signal_type} (Rank #{signal.rank}, Change: {signal.change_24h:+.2f}%, Strength: {signal.strength:.2f})")
            
            return top_signals
            
        except Exception as e:
            logger.error(f"❌ Error getting top opportunities: {e}")
            return []
    
    def format_signals_for_telegram(self, signals: List[CryptoSignal]) -> str:
        """Format trading signals for Telegram message"""
        if not signals:
            return "📊 No strong trading opportunities found in current market conditions."
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"""🔍 <b>Multi-Source Market Screener</b>
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
🏆 Rank: #{signal.rank}
💪 Strength: {signal.strength:.2f}
🎯 Confidence: {signal.confidence:.2f}
📈 Volume: {self._format_volume(signal.volume)}
💎 MCap: {self._format_volume(signal.market_cap)}

"""
        
        message += f"""📊 <b>Market Analysis:</b>
🔍 Screened 250+ top cryptocurrencies
🌐 Multi-source data (CoinGecko + CMC)
🎯 AI-powered signal generation
⚡ Real-time market data

Bot monitoring markets 24/7 🚀"""
        
        return message

# Convenience functions for easy integration
def screen_multi_source_market(cmc_api_key: str = None, limit: int = 5) -> List[CryptoSignal]:
    """Quick function to screen market using multiple sources"""
    screener = MultiSourceMarketScreener(cmc_api_key)
    return screener.get_top_opportunities(limit)

def get_multi_source_telegram_message(cmc_api_key: str = None, limit: int = 5) -> str:
    """Quick function to get formatted Telegram message from multi-source data"""
    screener = MultiSourceMarketScreener(cmc_api_key)
    signals = screener.get_top_opportunities(limit)
    return screener.format_signals_for_telegram(signals)

if __name__ == "__main__":
    # Test multi-source market screener
    print("🔍 Multi-Source Market Screener Test")
    print("=" * 60)
    
    # Initialize screener (CMC API key optional)
    cmc_key = os.getenv("CMC_API_KEY", "")
    screener = MultiSourceMarketScreener(cmc_key)
    
    if cmc_key:
        print("🔑 Using CoinMarketCap API key - full features enabled")
    else:
        print("📊 Using CoinGecko only (no CMC key) - still powerful!")
    
    print("🔍 Testing with live multi-source data...")
    
    # Test with real APIs
    try:
        signals = screener.get_top_opportunities(5)
        if signals:
            print(f"\n✅ SUCCESS! Found {len(signals)} opportunities:")
            for i, signal in enumerate(signals, 1):
                print(f"   {i}. {signal.symbol}: {signal.signal_type} (Rank #{signal.rank}, Change: {signal.change_24h:+.2f}%, Strength: {signal.strength:.2f})")
            
            # Show formatted Telegram message
            message = screener.format_signals_for_telegram(signals)
            print(f"\n📱 Telegram Message Preview:")
            print(message.replace("<b>", "").replace("</b>", ""))
            
        else:
            print("\n❌ No opportunities found - market might be stable or check API limits")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print("\n✅ Multi-Source Market Screener ready!")
    print("🚀 Comprehensive and reliable - integrate with main bot!")
    print("💡 Tip: Set CMC_API_KEY environment variable for enhanced features")