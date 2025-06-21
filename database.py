#!/usr/bin/env python3
"""
Database Management for Crypto Analysis Bot V3.2
===============================================
SQLite database for storing historical data, signals, and performance tracking
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoBotDatabase:
    """Database manager for crypto bot data storage"""
    
    def __init__(self, db_path: str = "crypto_bot.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Price history table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS price_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        price REAL NOT NULL,
                        volume REAL,
                        change_24h REAL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        source TEXT DEFAULT 'binance'
                    )
                """)
                
                # Signals table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        strength REAL,
                        confidence REAL,
                        price REAL,
                        indicators TEXT,
                        analysis TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        sent_telegram BOOLEAN DEFAULT FALSE
                    )
                """)
                
                # Bot performance table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS bot_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE UNIQUE,
                        total_signals INTEGER DEFAULT 0,
                        buy_signals INTEGER DEFAULT 0,
                        sell_signals INTEGER DEFAULT 0,
                        accuracy_rate REAL DEFAULT 0.0,
                        avg_confidence REAL DEFAULT 0.0,
                        market_trend TEXT,
                        notes TEXT
                    )
                """)
                
                # User settings table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_settings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        setting_name TEXT UNIQUE NOT NULL,
                        setting_value TEXT NOT NULL,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Analysis history table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analysis_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        analysis_type TEXT NOT NULL,
                        symbols_analyzed TEXT,
                        execution_time REAL,
                        results TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Market summary table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS market_summaries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE UNIQUE,
                        total_volume REAL,
                        gainers INTEGER DEFAULT 0,
                        losers INTEGER DEFAULT 0,
                        top_performer TEXT,
                        worst_performer TEXT,
                        market_sentiment TEXT,
                        summary_data TEXT
                    )
                """)
                
                conn.commit()
                logger.info("✅ Database initialized successfully")
                
        except Exception as e:
            logger.error(f"❌ Error initializing database: {e}")
    
    def save_price_data(self, symbol: str, price: float, volume: float = None, 
                       change_24h: float = None, source: str = "binance"):
        """Save price data to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO price_history (symbol, price, volume, change_24h, source)
                    VALUES (?, ?, ?, ?, ?)
                """, (symbol, price, volume, change_24h, source))
                conn.commit()
                logger.debug(f"💾 Saved price data for {symbol}: ${price}")
        except Exception as e:
            logger.error(f"❌ Error saving price data: {e}")
    
    def save_signal(self, symbol: str, signal_type: str, strength: float, 
                   confidence: float, price: float, indicators: Dict, 
                   analysis: str, sent_telegram: bool = False):
        """Save trading signal to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO signals (symbol, signal_type, strength, confidence, 
                                       price, indicators, analysis, sent_telegram)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol, signal_type, strength, confidence, price, 
                     json.dumps(indicators), analysis, sent_telegram))
                conn.commit()
                logger.info(f"🎯 Saved {signal_type} signal for {symbol}")
        except Exception as e:
            logger.error(f"❌ Error saving signal: {e}")
    
    def get_historical_prices(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get historical price data for a symbol"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT price, volume, change_24h, timestamp 
                    FROM price_history 
                    WHERE symbol = ? AND timestamp >= datetime('now', '-{} days')
                    ORDER BY timestamp ASC
                """.format(days), (symbol,))
                
                rows = cursor.fetchall()
                return [
                    {
                        'price': row[0],
                        'volume': row[1],
                        'change_24h': row[2],
                        'timestamp': row[3]
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"❌ Error getting historical prices: {e}")
            return []
    
    def get_recent_signals(self, symbol: str = None, limit: int = 10) -> List[Dict]:
        """Get recent trading signals"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if symbol:
                    cursor.execute("""
                        SELECT symbol, signal_type, strength, confidence, price, 
                               indicators, analysis, timestamp, sent_telegram
                        FROM signals 
                        WHERE symbol = ?
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (symbol, limit))
                else:
                    cursor.execute("""
                        SELECT symbol, signal_type, strength, confidence, price, 
                               indicators, analysis, timestamp, sent_telegram
                        FROM signals 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (limit,))
                
                rows = cursor.fetchall()
                return [
                    {
                        'symbol': row[0],
                        'signal_type': row[1],
                        'strength': row[2],
                        'confidence': row[3],
                        'price': row[4],
                        'indicators': json.loads(row[5]) if row[5] else {},
                        'analysis': row[6],
                        'timestamp': row[7],
                        'sent_telegram': row[8]
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"❌ Error getting recent signals: {e}")
            return []
    
    def update_daily_performance(self, date: str = None):
        """Update daily performance statistics"""
        try:
            if not date:
                date = datetime.now().strftime('%Y-%m-%d')
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get today's signals
                cursor.execute("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN signal_type = 'BUY' THEN 1 ELSE 0 END) as buy_signals,
                           SUM(CASE WHEN signal_type = 'SELL' THEN 1 ELSE 0 END) as sell_signals,
                           AVG(confidence) as avg_confidence
                    FROM signals 
                    WHERE DATE(timestamp) = ?
                """, (date,))
                
                row = cursor.fetchone()
                total_signals = row[0] or 0
                buy_signals = row[1] or 0
                sell_signals = row[2] or 0
                avg_confidence = row[3] or 0.0
                
                # Insert or update performance record
                cursor.execute("""
                    INSERT OR REPLACE INTO bot_performance 
                    (date, total_signals, buy_signals, sell_signals, avg_confidence)
                    VALUES (?, ?, ?, ?, ?)
                """, (date, total_signals, buy_signals, sell_signals, avg_confidence))
                
                conn.commit()
                logger.info(f"📊 Updated daily performance for {date}")
                
        except Exception as e:
            logger.error(f"❌ Error updating daily performance: {e}")
    
    def save_market_summary(self, date: str, summary_data: Dict):
        """Save daily market summary"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO market_summaries 
                    (date, total_volume, gainers, losers, top_performer, 
                     worst_performer, market_sentiment, summary_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    date,
                    summary_data.get('total_volume', 0),
                    summary_data.get('gainers', 0),
                    summary_data.get('losers', 0),
                    summary_data.get('top_performer', ''),
                    summary_data.get('worst_performer', ''),
                    summary_data.get('market_sentiment', ''),
                    json.dumps(summary_data)
                ))
                conn.commit()
                logger.info(f"📈 Saved market summary for {date}")
        except Exception as e:
            logger.error(f"❌ Error saving market summary: {e}")
    
    def get_performance_stats(self, days: int = 30) -> Dict:
        """Get bot performance statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Overall stats
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_days,
                        SUM(total_signals) as total_signals,
                        SUM(buy_signals) as total_buy,
                        SUM(sell_signals) as total_sell,
                        AVG(avg_confidence) as overall_confidence
                    FROM bot_performance 
                    WHERE date >= date('now', '-{} days')
                """.format(days))
                
                row = cursor.fetchone()
                
                return {
                    'total_days': row[0] or 0,
                    'total_signals': row[1] or 0,
                    'total_buy_signals': row[2] or 0,
                    'total_sell_signals': row[3] or 0,
                    'average_confidence': round(row[4] or 0.0, 2),
                    'signals_per_day': round((row[1] or 0) / max(row[0] or 1, 1), 2)
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting performance stats: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to manage database size"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old price history
                cursor.execute("""
                    DELETE FROM price_history 
                    WHERE timestamp < datetime('now', '-{} days')
                """.format(days_to_keep))
                
                # Delete old signals (keep longer for analysis)
                cursor.execute("""
                    DELETE FROM signals 
                    WHERE timestamp < datetime('now', '-{} days')
                """.format(days_to_keep * 2))
                
                # Delete old analysis history
                cursor.execute("""
                    DELETE FROM analysis_history 
                    WHERE timestamp < datetime('now', '-{} days')
                """.format(days_to_keep))
                
                conn.commit()
                logger.info(f"🧹 Cleaned up data older than {days_to_keep} days")
                
        except Exception as e:
            logger.error(f"❌ Error cleaning up old data: {e}")
    
    def export_data(self, table_name: str, output_file: str):
        """Export table data to CSV"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                df.to_csv(output_file, index=False)
                logger.info(f"📁 Exported {table_name} to {output_file}")
        except Exception as e:
            logger.error(f"❌ Error exporting data: {e}")
    
    def get_database_info(self) -> Dict:
        """Get database information and statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get table sizes
                tables = ['price_history', 'signals', 'bot_performance', 
                         'analysis_history', 'market_summaries']
                info = {}
                
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    info[table] = count
                
                # Database file size
                if os.path.exists(self.db_path):
                    file_size = os.path.getsize(self.db_path)
                    info['file_size_mb'] = round(file_size / (1024 * 1024), 2)
                
                return info
                
        except Exception as e:
            logger.error(f"❌ Error getting database info: {e}")
            return {}

# Convenience functions
def save_crypto_price(symbol: str, price: float, volume: float = None, change_24h: float = None):
    """Quick function to save crypto price"""
    db = CryptoBotDatabase()
    db.save_price_data(symbol, price, volume, change_24h)

def save_trading_signal(symbol: str, signal_type: str, strength: float, 
                       confidence: float, price: float, indicators: Dict, analysis: str):
    """Quick function to save trading signal"""
    db = CryptoBotDatabase()
    db.save_signal(symbol, signal_type, strength, confidence, price, indicators, analysis)

def get_crypto_history(symbol: str, days: int = 30) -> List[Dict]:
    """Quick function to get crypto price history"""
    db = CryptoBotDatabase()
    return db.get_historical_prices(symbol, days)

if __name__ == "__main__":
    # Test the database functionality
    print("🗄️  Database Module Test")
    print("=" * 40)
    
    # Initialize database
    db = CryptoBotDatabase("test_crypto_bot.db")
    
    # Test saving price data
    db.save_price_data("BTCUSDT", 45000.0, 1000000.0, -2.5)
    print("✅ Price data saved")
    
    # Test saving signal
    indicators = {'rsi': 75, 'macd': 0.01}
    db.save_signal("BTCUSDT", "SELL", 0.8, 0.9, 45000.0, indicators, "RSI overbought")
    print("✅ Signal saved")
    
    # Test getting data
    history = db.get_historical_prices("BTCUSDT", 7)
    signals = db.get_recent_signals(limit=5)
    
    print(f"📊 Historical records: {len(history)}")
    print(f"🎯 Recent signals: {len(signals)}")
    
    # Get database info
    info = db.get_database_info()
    print(f"📈 Database info: {info}")
    
    print("\n✅ Database module working correctly!")
    
    # Clean up test database
    os.remove("test_crypto_bot.db")