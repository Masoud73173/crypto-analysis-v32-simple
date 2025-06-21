#!/usr/bin/env python3
"""
Machine Learning Models for Crypto Analysis Bot V3.2
===================================================
Advanced AI/ML features for price prediction and intelligent signal generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
import os
import pickle
import json

# ML Libraries (optional imports - graceful fallback if not available)
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("⚠️  ML libraries not available. Install: pip install scikit-learn")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("⚠️  TensorFlow not available. Install: pip install tensorflow")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoPricePredictor:
    """Advanced ML-based crypto price prediction system"""
    
    def __init__(self, model_path: str = "models/"):
        self.model_path = model_path
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_column = "price"
        
        # Ensure models directory exists
        os.makedirs(model_path, exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'class': RandomForestRegressor if ML_AVAILABLE else None,
                'params': {'n_estimators': 100, 'random_state': 42, 'max_depth': 10}
            },
            'gradient_boost': {
                'class': GradientBoostingRegressor if ML_AVAILABLE else None,
                'params': {'n_estimators': 100, 'random_state': 42, 'max_depth': 6}
            },
            'ridge': {
                'class': Ridge if ML_AVAILABLE else None,
                'params': {'alpha': 1.0, 'random_state': 42}
            }
        }
    
    def prepare_features(self, price_data: List[Dict]) -> pd.DataFrame:
        """Prepare features for ML models"""
        try:
            df = pd.DataFrame(price_data)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Basic features
            df['price_lag_1'] = df['price'].shift(1)
            df['price_lag_6'] = df['price'].shift(6)
            df['price_lag_24'] = df['price'].shift(24)
            
            # Moving averages
            df['ma_5'] = df['price'].rolling(window=5).mean()
            df['ma_10'] = df['price'].rolling(window=10).mean()
            df['ma_20'] = df['price'].rolling(window=20).mean()
            
            # Price changes
            df['price_change_1h'] = df['price'].pct_change(1)
            df['price_change_6h'] = df['price'].pct_change(6)
            df['price_change_24h'] = df['price'].pct_change(24)
            
            # Volatility features
            df['volatility_5'] = df['price'].rolling(window=5).std()
            df['volatility_10'] = df['price'].rolling(window=10).std()
            
            # Volume features (if available)
            if 'volume' in df.columns:
                df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
                df['volume_change'] = df['volume'].pct_change()
                df['price_volume_ratio'] = df['price'] / (df['volume'] + 1)
            
            # Time-based features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            
            # Technical indicators
            df['rsi'] = self._calculate_rsi(df['price'])
            df['bb_position'] = self._calculate_bb_position(df['price'])
            
            # Drop NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error preparing features: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_bb_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands"""
        try:
            ma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = ma + (2 * std)
            lower = ma - (2 * std)
            position = (prices - lower) / (upper - lower)
            return position.fillna(0.5)
        except:
            return pd.Series([0.5] * len(prices), index=prices.index)
    
    def train_models(self, symbol: str, historical_data: List[Dict]) -> Dict[str, Any]:
        """Train multiple ML models for price prediction"""
        if not ML_AVAILABLE:
            logger.error("❌ ML libraries not available")
            return {"success": False, "error": "ML libraries not installed"}
        
        try:
            # Prepare features
            df = self.prepare_features(historical_data)
            if len(df) < 50:
                return {"success": False, "error": "Insufficient data for training"}
            
            # Define feature columns
            self.feature_columns = [col for col in df.columns 
                                  if col not in ['timestamp', 'price', 'symbol']]
            
            # Prepare training data
            X = df[self.feature_columns]
            y = df[self.target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Store scaler
            self.scalers[symbol] = scaler
            
            # Train models
            results = {}
            
            for model_name, config in self.model_configs.items():
                if config['class'] is None:
                    continue
                
                try:
                    # Initialize and train model
                    model = config['class'](**config['params'])
                    model.fit(X_train_scaled, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                    
                    results[model_name] = {
                        'mse': mse,
                        'mae': mae,
                        'r2_score': r2,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }
                    
                    # Store model
                    self.models[f"{symbol}_{model_name}"] = model
                    
                    logger.info(f"✅ Trained {model_name} for {symbol}: R2={r2:.4f}")
                    
                except Exception as e:
                    logger.error(f"❌ Error training {model_name}: {e}")
                    continue
            
            # Save models and scaler
            self._save_models(symbol)
            
            return {
                "success": True,
                "symbol": symbol,
                "models_trained": len(results),
                "results": results,
                "data_points": len(df),
                "features": len(self.feature_columns)
            }
            
        except Exception as e:
            logger.error(f"❌ Error in model training: {e}")
            return {"success": False, "error": str(e)}
    
    def predict_price(self, symbol: str, current_data: List[Dict], 
                     hours_ahead: int = 1) -> Dict[str, Any]:
        """Predict future price using ensemble of trained models"""
        try:
            # Load models if not in memory
            if not any(key.startswith(symbol) for key in self.models.keys()):
                self._load_models(symbol)
            
            # Prepare features
            df = self.prepare_features(current_data)
            if len(df) == 0:
                return {"success": False, "error": "Could not prepare features"}
            
            # Get latest features
            latest_features = df[self.feature_columns].iloc[-1:].values
            
            # Scale features
            if symbol in self.scalers:
                scaler = self.scalers[symbol]
                latest_features = scaler.transform(latest_features)
            
            # Make predictions with all models
            predictions = {}
            model_weights = {}
            
            for model_key, model in self.models.items():
                if model_key.startswith(symbol):
                    model_name = model_key.replace(f"{symbol}_", "")
                    try:
                        pred = model.predict(latest_features)[0]
                        predictions[model_name] = pred
                        
                        # Simple weight based on model type (can be improved)
                        if 'random_forest' in model_name:
                            model_weights[model_name] = 0.4
                        elif 'gradient_boost' in model_name:
                            model_weights[model_name] = 0.4
                        else:
                            model_weights[model_name] = 0.2
                            
                    except Exception as e:
                        logger.warning(f"⚠️  Error with model {model_name}: {e}")
                        continue
            
            if not predictions:
                return {"success": False, "error": "No models available for prediction"}
            
            # Calculate ensemble prediction
            total_weight = sum(model_weights.values())
            if total_weight > 0:
                ensemble_pred = sum(pred * model_weights.get(name, 0) 
                                  for name, pred in predictions.items()) / total_weight
            else:
                ensemble_pred = sum(predictions.values()) / len(predictions)
            
            # Calculate prediction confidence
            pred_values = list(predictions.values())
            pred_std = np.std(pred_values)
            pred_mean = np.mean(pred_values)
            confidence = max(0, 1 - (pred_std / pred_mean)) if pred_mean > 0 else 0
            
            # Current price for comparison
            current_price = df['price'].iloc[-1]
            
            # Calculate expected change
            price_change = ((ensemble_pred - current_price) / current_price) * 100
            
            return {
                "success": True,
                "symbol": symbol,
                "current_price": current_price,
                "predicted_price": ensemble_pred,
                "price_change_percent": price_change,
                "confidence": confidence,
                "hours_ahead": hours_ahead,
                "individual_predictions": predictions,
                "models_used": len(predictions),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error making prediction: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_ml_signals(self, symbol: str, historical_data: List[Dict]) -> Dict[str, Any]:
        """Generate trading signals using ML predictions"""
        try:
            # Make prediction
            prediction = self.predict_price(symbol, historical_data, hours_ahead=6)
            
            if not prediction["success"]:
                return {"success": False, "error": prediction.get("error")}
            
            price_change = prediction["price_change_percent"]
            confidence = prediction["confidence"]
            
            # Signal generation logic
            signal_type = "HOLD"
            signal_strength = 0.5
            
            # Strong signals
            if abs(price_change) > 5 and confidence > 0.7:
                if price_change > 0:
                    signal_type = "STRONG_BUY"
                    signal_strength = min(0.9, confidence + 0.1)
                else:
                    signal_type = "STRONG_SELL"
                    signal_strength = min(0.9, confidence + 0.1)
            
            # Regular signals
            elif abs(price_change) > 2 and confidence > 0.5:
                if price_change > 0:
                    signal_type = "BUY"
                    signal_strength = confidence
                else:
                    signal_type = "SELL"
                    signal_strength = confidence
            
            return {
                "success": True,
                "symbol": symbol,
                "signal": signal_type,
                "strength": signal_strength,
                "confidence": confidence,
                "predicted_change": price_change,
                "prediction_horizon": "6 hours",
                "ml_powered": True,
                "analysis": f"ML predicts {price_change:+.2f}% change with {confidence:.1%} confidence"
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating ML signals: {e}")
            return {"success": False, "error": str(e)}
    
    def _save_models(self, symbol: str):
        """Save trained models and scalers"""
        try:
            symbol_models = {k: v for k, v in self.models.items() if k.startswith(symbol)}
            
            # Save models
            models_file = os.path.join(self.model_path, f"{symbol}_models.pkl")
            with open(models_file, 'wb') as f:
                pickle.dump(symbol_models, f)
            
            # Save scaler
            if symbol in self.scalers:
                scaler_file = os.path.join(self.model_path, f"{symbol}_scaler.pkl")
                with open(scaler_file, 'wb') as f:
                    pickle.dump(self.scalers[symbol], f)
            
            # Save feature columns
            features_file = os.path.join(self.model_path, f"{symbol}_features.json")
            with open(features_file, 'w') as f:
                json.dump(self.feature_columns, f)
            
            logger.info(f"💾 Saved ML models for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Error saving models: {e}")
    
    def _load_models(self, symbol: str):
        """Load trained models and scalers"""
        try:
            # Load models
            models_file = os.path.join(self.model_path, f"{symbol}_models.pkl")
            if os.path.exists(models_file):
                with open(models_file, 'rb') as f:
                    symbol_models = pickle.load(f)
                self.models.update(symbol_models)
            
            # Load scaler
            scaler_file = os.path.join(self.model_path, f"{symbol}_scaler.pkl")
            if os.path.exists(scaler_file):
                with open(scaler_file, 'rb') as f:
                    self.scalers[symbol] = pickle.load(f)
            
            # Load feature columns
            features_file = os.path.join(self.model_path, f"{symbol}_features.json")
            if os.path.exists(features_file):
                with open(features_file, 'r') as f:
                    self.feature_columns = json.load(f)
            
            logger.info(f"📂 Loaded ML models for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
    
    def get_model_info(self, symbol: str = None) -> Dict[str, Any]:
        """Get information about trained models"""
        try:
            info = {
                "ml_available": ML_AVAILABLE,
                "tensorflow_available": TF_AVAILABLE,
                "models_path": self.model_path,
                "total_models": len(self.models),
                "trained_symbols": list(set(key.split('_')[0] for key in self.models.keys()))
            }
            
            if symbol:
                symbol_models = [k for k in self.models.keys() if k.startswith(symbol)]
                info[f"{symbol}_models"] = symbol_models
                info[f"{symbol}_scaler_available"] = symbol in self.scalers
            
            return info
            
        except Exception as e:
            logger.error(f"❌ Error getting model info: {e}")
            return {"error": str(e)}

class LSTMPredictor:
    """LSTM-based deep learning price predictor"""
    
    def __init__(self, model_path: str = "models/"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.sequence_length = 60
        
    def create_sequences(self, data: np.array) -> Tuple[np.array, np.array]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def train_lstm(self, price_data: List[float], symbol: str) -> Dict[str, Any]:
        """Train LSTM model for price prediction"""
        if not TF_AVAILABLE:
            return {"success": False, "error": "TensorFlow not available"}
        
        try:
            # Prepare data
            prices = np.array(price_data).reshape(-1, 1)
            
            # Scale data
            self.scaler = MinMaxScaler()
            scaled_prices = self.scaler.fit_transform(prices)
            
            # Create sequences
            X, y = self.create_sequences(scaled_prices)
            
            if len(X) < 50:
                return {"success": False, "error": "Insufficient data for LSTM training"}
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build LSTM model
            self.model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_test, y_test),
                verbose=0
            )
            
            # Save model
            model_file = os.path.join(self.model_path, f"{symbol}_lstm.h5")
            self.model.save(model_file)
            
            # Save scaler
            scaler_file = os.path.join(self.model_path, f"{symbol}_lstm_scaler.pkl")
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            return {
                "success": True,
                "symbol": symbol,
                "training_loss": history.history['loss'][-1],
                "validation_loss": history.history['val_loss'][-1],
                "epochs": len(history.history['loss'])
            }
            
        except Exception as e:
            logger.error(f"❌ Error training LSTM: {e}")
            return {"success": False, "error": str(e)}

# Convenience functions
def train_crypto_models(symbol: str, historical_data: List[Dict]) -> Dict[str, Any]:
    """Quick function to train ML models for a crypto symbol"""
    predictor = CryptoPricePredictor()
    return predictor.train_models(symbol, historical_data)

def predict_crypto_price(symbol: str, current_data: List[Dict]) -> Dict[str, Any]:
    """Quick function to predict crypto price"""
    predictor = CryptoPricePredictor()
    return predictor.predict_price(symbol, current_data)

def get_ml_signals(symbol: str, historical_data: List[Dict]) -> Dict[str, Any]:
    """Quick function to get ML-based trading signals"""
    predictor = CryptoPricePredictor()
    return predictor.generate_ml_signals(symbol, historical_data)

if __name__ == "__main__":
    # Test ML functionality
    print("🧠 Machine Learning Module Test")
    print("=" * 40)
    
    if not ML_AVAILABLE:
        print("❌ ML libraries not available")
        print("Install with: pip install scikit-learn")
    else:
        print("✅ ML libraries available")
    
    if not TF_AVAILABLE:
        print("❌ TensorFlow not available")  
        print("Install with: pip install tensorflow")
    else:
        print("✅ TensorFlow available")
    
    # Test with dummy data
    if ML_AVAILABLE:
        predictor = CryptoPricePredictor()
        
        # Generate test data
        test_data = []
        base_price = 45000
        for i in range(100):
            price_change = np.random.normal(0, 0.02)
            base_price *= (1 + price_change)
            test_data.append({
                "price": base_price,
                "volume": np.random.uniform(1000000, 5000000),
                "timestamp": (datetime.now() - timedelta(hours=100-i)).isoformat()
            })
        
        # Test model training
        result = predictor.train_models("TESTUSDT", test_data)
        print(f"🎯 Training result: {result['success']}")
        
        if result['success']:
            # Test prediction
            pred_result = predictor.predict_price("TESTUSDT", test_data[-50:])
            print(f"📈 Prediction: {pred_result['success']}")
            
            if pred_result['success']:
                change = pred_result['price_change_percent']
                confidence = pred_result['confidence']
                print(f"💡 Predicted change: {change:+.2f}% (confidence: {confidence:.1%})")
    
    print("\n✅ ML module test completed!")