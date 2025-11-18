#!/usr/bin/env python3
"""
Batch ML Pipeline Runner - Run pipeline for all S&P 500 companies
Processes all companies in data/raw/ directory

Usage: python3 run_batch_pipeline.py
"""

import os
import glob
import subprocess
import time
from datetime import datetime
import json

# Configuration
DATA_DIR = "data/raw"
SKIP_FETCH = True  # Skip fetching, we already have the data
LOG_FILE = f"batch_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def get_symbols():
    """Get list of symbols from batch_symbols.txt"""
    if os.path.exists('batch_symbols.txt'):
        with open('batch_symbols.txt', 'r') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        # Fallback: read from CSV files
        csv_files = glob.glob(os.path.join(DATA_DIR, "*_earnings_with_q4.csv"))
        symbols = []
        for fp in csv_files:
            filename = os.path.basename(fp)
            symbol = filename.replace('_earnings_with_q4.csv', '')
            if os.path.getsize(fp) > 1000:  # >1KB
                symbols.append(symbol)
        return sorted(symbols)

def run_pipeline_for_symbol(symbol, log_handle):
    """Run full pipeline for a single symbol using run_pipeline.py"""
    log_msg = f"\n{'='*80}\n"
    log_msg += f"Processing: {symbol} - {datetime.now()}\n"
    log_msg += f"{'='*80}\n"
    print(log_msg, end='')
    log_handle.write(log_msg)
    log_handle.flush()
    
    start_time = time.time()
    
    # Use run_pipeline.py with --skip-fetch
    cmd = f"python3 run_pipeline.py --symbol {symbol} --skip-fetch"
    
    results = {
        'symbol': symbol,
        'start_time': datetime.now().isoformat(),
    }
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=600  # 10 min timeout total
        )
        
        total_duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {symbol}: SUCCESS ({total_duration:.1f}s)")
            results['status'] = 'success'
            results['duration'] = total_duration
            log_handle.write(f"✅ SUCCESS ({total_duration:.1f}s)\n\n")
        else:
            print(f"❌ {symbol}: FAILED ({total_duration:.1f}s)")
            results['status'] = 'failed'
            results['duration'] = total_duration
            results['error'] = result.stderr[-200:] if result.stderr else ''
            log_handle.write(f"❌ FAILED ({total_duration:.1f}s)\n")
            log_handle.write(f"Error: {result.stderr[-200:]}\n\n")
            
    except subprocess.TimeoutExpired:
        total_duration = 600
        print(f"❌ {symbol}: TIMEOUT")
        results['status'] = 'timeout'
        results['duration'] = total_duration
        log_handle.write(f"❌ TIMEOUT\n\n")
    except Exception as e:
        total_duration = time.time() - start_time
        print(f"❌ {symbol}: ERROR - {str(e)[:50]}")
        results['status'] = 'error'
        results['duration'] = total_duration
        results['error'] = str(e)
        log_handle.write(f"❌ ERROR: {str(e)}\n\n")
    
    results['end_time'] = datetime.now().isoformat()
    log_handle.flush()
    
    return results

def main():
    print("="*80)
    print("BATCH ML PIPELINE - S&P 500 COMPANIES")
    print("="*80)
    
    # Get symbols list
    symbols = get_symbols()
    
    print(f"\nFound {len(symbols)} companies")
    print(f"Log file: {LOG_FILE}")
    
    # Estimate time
    avg_time_per_symbol = 100  # seconds (conservative estimate)
    total_estimated = (len(symbols) * avg_time_per_symbol) / 3600
    print(f"Estimated time: ~{total_estimated:.1f} hours")
    print(f"\nFirst 5: {symbols[:5]}")
    print(f"Last 5: {symbols[-5:]}")
    
    # Auto-start in background (no input required)
    print(f"\n🚀 Starting batch processing...")
    
    # Start processing
    start_time = time.time()
    all_results = []
    
    with open(LOG_FILE, 'w') as log:
        log.write(f"Batch Pipeline Started: {datetime.now()}\n")
        log.write(f"Total symbols: {len(symbols)}\n")
        log.write("="*80 + "\n\n")
        
        success_count = 0
        failed_count = 0
        
        for idx, symbol in enumerate(symbols, 1):
            print(f"\n[{idx}/{len(symbols)}] {symbol}")
            log.write(f"[{idx}/{len(symbols)}] {symbol}\n")
            
            result = run_pipeline_for_symbol(symbol, log)
            all_results.append(result)
            
            if result.get('status') == 'success':
                success_count += 1
            else:
                failed_count += 1
            
            # Progress report every 10 symbols
            if idx % 10 == 0:
                elapsed = time.time() - start_time
                avg_per_symbol = elapsed / idx
                remaining = (len(symbols) - idx) * avg_per_symbol
                
                progress = f"\n{'='*80}\n"
                progress += f"PROGRESS: {idx}/{len(symbols)} ({idx/len(symbols)*100:.1f}%)\n"
                progress += f"Success: {success_count}, Failed: {failed_count}\n"
                progress += f"Elapsed: {elapsed/60:.1f}min, Remaining: ~{remaining/60:.1f}min\n"
                progress += f"{'='*80}\n"
                print(progress)
                log.write(progress)
                log.flush()
        
        # Final summary
        total_time = time.time() - start_time
        
        summary = f"\n{'='*80}\n"
        summary += "BATCH PIPELINE COMPLETE\n"
        summary += f"{'='*80}\n\n"
        summary += f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)\n"
        summary += f"Processed: {len(symbols)} companies\n"
        summary += f"Success: {success_count}\n"
        summary += f"Failed: {failed_count}\n"
        summary += f"Success rate: {success_count/len(symbols)*100:.1f}%\n"
        summary += f"\n{'='*80}\n"
        
        print(summary)
        log.write(summary)
        
        # Save results JSON
        results_file = LOG_FILE.replace('.log', '_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration': total_time,
                'total_symbols': len(symbols),
                'success_count': success_count,
                'failed_count': failed_count,
                'results': all_results
            }, f, indent=2)
        
        print(f"Results saved: {results_file}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user!")
        print("Progress has been logged.")

