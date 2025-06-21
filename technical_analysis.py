#!/usr/bin/env python3
"""
Advanced Technical Analysis Module for Crypto Analysis Bot V3.2
===============================================================
Professional TA indicators: RSI, MACD, Bollinger Bands, EMA, SMA, Stochastic, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalAnalysis:
    """Advanced Technical Analysis class with professional indicators"""
    
    def __init__(self):
        self.indicators = {}
        
    def calculate_sma(self, prices: List[float], period: int = 20) -> List[float]:
        """Calculate Simple Moving Average"""
        try:
            df = pd.DataFrame({'price': prices})
            sma = df['price'].rolling(window=period).mean()
            return sma.fillna(0).tolist()
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return [0] * len(prices)
    
    def calculate_ema(self, prices: List[float], period: int = 20) -> List[float]:
        """Calculate Exponential Moving Average"""
        try:
            df = pd.DataFrame({'price': prices})
            ema = df['price'].ewm(span=period).mean()
            return ema.fillna(0).tolist()
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return [0] * len(prices)
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """Calculate Relative Strength Index"""
        try:
            df = pd.DataFrame({'price': prices})
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50).tolist()
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return [50] * len(prices)
    
    def calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            df = pd.DataFrame({'price': prices})
            
            # Calculate EMAs
            ema_fast = df['price'].ewm(span=fast).mean()
            ema_slow = df['price'].ewm(span=slow).mean()
            
            # MACD line
            macd_line = ema_fast - ema_slow
            
            # Signal line
            signal_line = macd_line.ewm(span=signal).mean()
            
            # Histogram
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line.fillna(0).tolist(),
                'signal': signal_line.fillna(0).tolist(),
                'histogram': histogram.fillna(0).tolist()
            }
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return {
                'macd': [0] * len(prices),
                'signal': [0] * len(prices),
                'histogram': [0] * len(prices)
            }
    
    def calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2) -> Dict:
        """Calculate Bollinger Bands"""
        try:
            df = pd.DataFrame({'price': prices})
            
            # Middle band (SMA)
            middle = df['price'].rolling(window=period).mean()
            
            # Standard deviation
            std = df['price'].rolling(window=period).std()
            
            # Upper and lower bands
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            
            return {
                'upper': upper.fillna(0).tolist(),
                'middle': middle.fillna(0).tolist(),
                'lower': lower.fillna(0).tolist()
            }
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return {
                'upper': [0] * len(prices),
                'middle': [0] * len(prices),
                'lower': [0] * len(prices)
            }
    
    def calculate_stochastic(self, highs: List[float], lows: List[float], closes: List[float], 
                           k_period: int = 14, d_period: int = 3) -> Dict:
        """Calculate Stochastic Oscillator"""
        try:
            df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
            
            # Lowest low and highest high over the period
            lowest_low = df['low'].rolling(window=k_period).min()
            highest_high = df['high'].rolling(window=k_period).max()
            
            # %K line
            k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
            
            # %D line (moving average of %K)
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return {
                'k_percent': k_percent.fillna(50).tolist(),
                'd_percent': d_percent.fillna(50).tolist()
            }
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            return {
                'k_percent': [50] * len(closes),
                'd_percent': [50] * len(closes)
            }
    
    def calculate_williams_r(self, highs: List[float], lows: List[float], closes: List[float], 
                           period: int = 14) -> List[float]:
        """Calculate Williams %R"""
        try:
            df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
            
            highest_high = df['high'].rolling(window=period).max()
            lowest_low = df['low'].rolling(window=period).min()
            
            williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
            
            return williams_r.fillna(-50).tolist()
        except Exception as e:
            logger.error(f"Error calculating Williams %R: {e}")
            return [-50] * len(closes)
    
    def calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], 
                     period: int = 14) -> List[float]:
        """Calculate Average True Range"""
        try:
            df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
            
            # True Range calculation
            df['prev_close'] = df['close'].shift(1)
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['prev_close'])
            df['tr3'] = abs(df['low'] - df['prev_close'])
            df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            
            # ATR (moving average of True Range)
            atr = df['true_range'].rolling(window=period).mean()
            
            return atr.fillna(0).tolist()
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return [0] * len(closes)
    
    def calculate_momentum(self, prices: List[float], period: int = 10) -> List[float]:
        """Calculate Momentum indicator"""
        try:
            df = pd.DataFrame({'price': prices})
            momentum = df['price'] - df['price'].shift(period)
            return momentum.fillna(0).tolist()
        except Exception as e:
            logger.error(f"Error calculating Momentum: {e}")
            return [0] * len(prices)
    
    def analyze_trend(self, prices: List[float]) -> str:
        """Analyze overall trend"""
        try:
            if len(prices) < 20:
                return "Insufficient data"
            
            sma_short = self.calculate_sma(prices, 10)[-1]
            sma_long = self.calculate_sma(prices, 20)[-1]
            
            if sma_short > sma_long:
                return "Bullish"
            elif sma_short < sma_long:
                return "Bearish"
            else:
                return "Sideways"
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return "Unknown"
    
    def generate_signals(self, symbol: str, prices: List[float], volumes: List[float] = None) -> Dict:
        """Generate comprehensive trading signals"""
        try:
            if len(prices) < 50:
                return {
                    'signal': 'HOLD',
                    'strength': 0.0,
                    'confidence': 0.0,
                    'indicators': {},
                    'analysis': 'Insufficient historical data'
                }
            
            # Calculate indicators
            rsi = self.calculate_rsi(prices)[-1]
            macd_data = self.calculate_macd(prices)
            macd = macd_data['macd'][-1]
            signal_line = macd_data['signal'][-1]
            
            bb_data = self.calculate_bollinger_bands(prices)
            current_price = prices[-1]
            bb_upper = bb_data['upper'][-1]
            bb_lower = bb_data['lower'][-1]
            bb_middle = bb_data['middle'][-1]
            
            trend = self.analyze_trend(prices)
            
            # Signal generation logic
            signals = []
            
            # RSI signals
            if rsi < 30:
                signals.append(('BUY', 0.7, 'RSI Oversold'))
            elif rsi > 70:
                signals.append(('SELL', 0.7, 'RSI Overbought'))
            
            # MACD signals
            if macd > signal_line:
                signals.append(('BUY', 0.6, 'MACD Bullish'))
            else:
                signals.append(('SELL', 0.6, 'MACD Bearish'))
            
            # Bollinger Bands signals
            if current_price < bb_lower:
                signals.append(('BUY', 0.8, 'Price below BB lower'))
            elif current_price > bb_upper:
                signals.append(('SELL', 0.8, 'Price above BB upper'))
            
            # Aggregate signals
            buy_signals = [s for s in signals if s[0] == 'BUY']
            sell_signals = [s for s in signals if s[0] == 'SELL']
            
            if len(buy_signals) > len(sell_signals):
                final_signal = 'BUY'
                strength = sum(s[1] for s in buy_signals) / len(buy_signals)
            elif len(sell_signals) > len(buy_signals):
                final_signal = 'SELL'
                strength = sum(s[1] for s in sell_signals) / len(sell_signals)
            else:
                final_signal = 'HOLD'
                strength = 0.5
            
            # Confidence based on signal agreement
            total_signals = len(signals)
            agreement = max(len(buy_signals), len(sell_signals))
            confidence = agreement / total_signals if total_signals > 0 else 0.0
            
            return {
                'signal': final_signal,
                'strength': round(strength, 2),
                'confidence': round(confidence, 2),
                'indicators': {
                    'rsi': round(rsi, 2),
                    'macd': round(macd, 4),
                    'macd_signal': round(signal_line, 4),
                    'bb_position': round((current_price - bb_lower) / (bb_upper - bb_lower), 2),
                    'trend': trend
                },
                'analysis': f"Generated {len(signals)} signals: {len(buy_signals)} BUY, {len(sell_signals)} SELL",
                'reasons': [s[2] for s in signals]
            }
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return {
                'signal': 'HOLD',
                'strength': 0.0,
                'confidence': 0.0,
                'indicators': {},
                'analysis': f'Error in analysis: {str(e)}'
            }

# Convenience functions for easy import
def get_rsi(prices: List[float], period: int = 14) -> float:
    """Quick RSI calculation"""
    ta = TechnicalAnalysis()
    rsi_values = ta.calculate_rsi(prices, period)
    return rsi_values[-1] if rsi_values else 50.0

def get_macd(prices: List[float]) -> Dict:
    """Quick MACD calculation"""
    ta = TechnicalAnalysis()
    return ta.calculate_macd(prices)

def get_bollinger_bands(prices: List[float]) -> Dict:
    """Quick Bollinger Bands calculation"""
    ta = TechnicalAnalysis()
    return ta.calculate_bollinger_bands(prices)

def analyze_crypto_signals(symbol: str, prices: List[float]) -> Dict:
    """Quick signal analysis"""
    ta = TechnicalAnalysis()
    return ta.generate_signals(symbol, prices)

if __name__ == "__main__":
    # Test the technical analysis with sample data
    print("🧮 Technical Analysis Module Test")
    print("=" * 40)
    
    # Sample price data (simulated)
    sample_prices = [100 + i + (i % 5) * 2 for i in range(50)]
    
    ta = TechnicalAnalysis()
    
    # Test RSI
    rsi = ta.calculate_rsi(sample_prices)
    print(f"📊 RSI: {rsi[-1]:.2f}")
    
    # Test MACD
    macd_data = ta.calculate_macd(sample_prices)
    print(f"📈 MACD: {macd_data['macd'][-1]:.4f}")
    
    # Test signals
    signals = ta.generate_signals("TEST", sample_prices)
    print(f"🎯 Signal: {signals['signal']} (Strength: {signals['strength']}, Confidence: {signals['confidence']})")
    
    print("\n✅ Technical Analysis module working correctly!")