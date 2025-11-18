"""
Project Configuration - Path Management
Tüm proje path'lerini merkezi olarak yönetir
"""
import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')

def get_paths(symbol):
    """
    Belirtilen hisse için tüm path'leri döndür
    
    Args:
        symbol: Hisse sembolü (örn: 'IBM', 'TSLA')
        
    Returns:
        dict: Tüm path'leri içeren dictionary
    """
    symbol = symbol.upper()
    
    return {
        # Raw data
        'raw_csv': os.path.join(BASE_DIR, f'data/raw/{symbol}_earnings_with_q4.csv'),
        
        # Processed data directory
        'processed_dir': os.path.join(BASE_DIR, f'data/processed/{symbol}'),
        'X_train': os.path.join(BASE_DIR, f'data/processed/{symbol}/X_train.csv'),
        'y_train': os.path.join(BASE_DIR, f'data/processed/{symbol}/y_train.csv'),
        'X_test': os.path.join(BASE_DIR, f'data/processed/{symbol}/X_test.csv'),
        'y_test': os.path.join(BASE_DIR, f'data/processed/{symbol}/y_test.csv'),
        'features': os.path.join(BASE_DIR, f'data/processed/{symbol}/feature_names.txt'),
        'data_info': os.path.join(BASE_DIR, f'data/processed/{symbol}/data_info.json'),
        
        # Models directory
        'models_dir': os.path.join(BASE_DIR, f'models/{symbol}'),
        'rf_model': os.path.join(BASE_DIR, f'models/{symbol}/rf_model.pkl'),
        'xgb_model': os.path.join(BASE_DIR, f'models/{symbol}/xgb_model.pkl'),
        'lr_model': os.path.join(BASE_DIR, f'models/{symbol}/lr_model.pkl'),
        'preprocessor': os.path.join(BASE_DIR, f'models/{symbol}/preprocessor.pkl'),
        'training_results': os.path.join(BASE_DIR, f'models/{symbol}/training_results.json'),
        
        # Results directory
        'results_dir': os.path.join(BASE_DIR, f'results/{symbol}'),
        'timeseries_cv_dir': os.path.join(BASE_DIR, f'results/{symbol}/timeseries_cv'),
        'evaluation_report': os.path.join(BASE_DIR, f'results/{symbol}/evaluation_report.txt'),
        'model_comparison': os.path.join(BASE_DIR, f'results/{symbol}/model_comparison.csv'),
        'misclassified': os.path.join(BASE_DIR, f'results/{symbol}/misclassified_samples.csv'),
    }

def get_fundamentals_paths(symbol):
    """
    Fundamentals-only pipeline için path'leri döndür
    (Analyst estimates olmadan, sadece actual data kullanır)
    
    Args:
        symbol: Hisse sembolü (örn: 'IBM', 'TSLA')
        
    Returns:
        dict: Fundamentals-only path'leri
    """
    symbol = symbol.upper()
    
    return {
        # Raw data (fundamentals only)
        'raw_csv': os.path.join(BASE_DIR, f'data/raw/{symbol}_only_fundamental.csv'),
        
        # Processed data directory (separate folder)
        'processed_dir': os.path.join(BASE_DIR, f'data/processed/{symbol}/only_fundamentals_processed'),
        'X_train': os.path.join(BASE_DIR, f'data/processed/{symbol}/only_fundamentals_processed/X_train.csv'),
        'y_train': os.path.join(BASE_DIR, f'data/processed/{symbol}/only_fundamentals_processed/y_train.csv'),
        'X_test': os.path.join(BASE_DIR, f'data/processed/{symbol}/only_fundamentals_processed/X_test.csv'),
        'y_test': os.path.join(BASE_DIR, f'data/processed/{symbol}/only_fundamentals_processed/y_test.csv'),
        'features': os.path.join(BASE_DIR, f'data/processed/{symbol}/only_fundamentals_processed/feature_names.txt'),
        'data_info': os.path.join(BASE_DIR, f'data/processed/{symbol}/only_fundamentals_processed/data_info.json'),
        
        # Models directory (separate folder)
        'models_dir': os.path.join(BASE_DIR, f'models/{symbol}/only_fundamentals'),
        'rf_model': os.path.join(BASE_DIR, f'models/{symbol}/only_fundamentals/rf_model.pkl'),
        'xgb_model': os.path.join(BASE_DIR, f'models/{symbol}/only_fundamentals/xgb_model.pkl'),
        'lr_model': os.path.join(BASE_DIR, f'models/{symbol}/only_fundamentals/lr_model.pkl'),
        'preprocessor': os.path.join(BASE_DIR, f'models/{symbol}/only_fundamentals/preprocessor.pkl'),
        'training_results': os.path.join(BASE_DIR, f'models/{symbol}/only_fundamentals/training_results.json'),
        
        # Results directory (separate folder)
        'results_dir': os.path.join(BASE_DIR, f'results/{symbol}/only_fundamentals'),
        'timeseries_cv_dir': os.path.join(BASE_DIR, f'results/{symbol}/only_fundamentals/timeseries_cv'),
        'evaluation_report': os.path.join(BASE_DIR, f'results/{symbol}/only_fundamentals/evaluation_report.txt'),
        'model_comparison': os.path.join(BASE_DIR, f'results/{symbol}/only_fundamentals/model_comparison.csv'),
        'misclassified': os.path.join(BASE_DIR, f'results/{symbol}/only_fundamentals/misclassified_samples.csv'),
    }

def ensure_directories(symbol):
    """Gerekli klasörleri oluştur"""
    paths = get_paths(symbol)
    
    os.makedirs(paths['processed_dir'], exist_ok=True)
    os.makedirs(paths['models_dir'], exist_ok=True)
    os.makedirs(paths['results_dir'], exist_ok=True)
    
    return paths

def ensure_fundamentals_directories(symbol):
    """Fundamentals-only pipeline için klasörleri oluştur"""
    paths = get_fundamentals_paths(symbol)
    
    os.makedirs(paths['processed_dir'], exist_ok=True)
    os.makedirs(paths['models_dir'], exist_ok=True)
    os.makedirs(paths['results_dir'], exist_ok=True)
    os.makedirs(paths['timeseries_cv_dir'], exist_ok=True)
    
    return paths

