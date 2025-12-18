#!/usr/bin/env python
"""
Profiling script for ch_learn.py
Run this with: python -m cProfile -o profile_output.prof profile_ch_learn.py
Or: kernprof -l -v profile_ch_learn.py
"""

import cProfile
import pstats
import io
from pathlib import Path

def profile_with_cprofile():
    """Profile ch_learn.py using cProfile."""
    print("Starting cProfile profiling...")
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Import and run the main function
    profiler.enable()
    
    # Import ch_learn and run
    import sys
    sys.argv = ['ch_learn.py', '--epochs', '2', '--no-wandb', '--no-resume']
    
    from ch_learn import main
    main()
    
    profiler.disable()
    
    # Save stats to file
    profiler.dump_stats('profile_output.prof')
    
    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(50)  # Top 50 functions
    
    print(s.getvalue())
    
    # Save to text file
    with open('profile_report.txt', 'w') as f:
        ps = pstats.Stats(profiler, stream=f).sort_stats('cumulative')
        ps.print_stats()
    
    print("\nProfile saved to:")
    print("  - profile_output.prof (binary)")
    print("  - profile_report.txt (human-readable)")
    print("\nView with: python -m pstats profile_output.prof")

if __name__ == "__main__":
    profile_with_cprofile()
