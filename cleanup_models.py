#!/usr/bin/env python3
"""
Cleanup script to remove problematic model files and start fresh
"""

import os
import glob

def cleanup_models():
    """Remove old model files that may have serialization issues"""
    
    print("🧹 Cleaning up old model files...")
    
    # Files to remove
    files_to_remove = [
        'best_model.pth',
        'best_model_scaler.pkl',
        '*.pth',
        '*.pkl'
    ]
    
    removed_count = 0
    
    for pattern in files_to_remove:
        if '*' in pattern:
            # Handle glob patterns
            for file_path in glob.glob(pattern):
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        print(f"  ✓ Removed: {file_path}")
                        removed_count += 1
                    except Exception as e:
                        print(f"  ❌ Failed to remove {file_path}: {e}")
        else:
            # Handle individual files
            if os.path.exists(pattern):
                try:
                    os.remove(pattern)
                    print(f"  ✓ Removed: {pattern}")
                    removed_count += 1
                except Exception as e:
                    print(f"  ❌ Failed to remove {pattern}: {e}")
    
    if removed_count == 0:
        print("  ℹ️ No model files found to remove")
    else:
        print(f"  ✅ Removed {removed_count} files")
    
    print("\n🚀 You can now run training again with:")
    print("   python trade-2.py")

if __name__ == "__main__":
    cleanup_models()
