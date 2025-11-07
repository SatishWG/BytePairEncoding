"""
Script to save the merges dictionary from the notebook to a JSON file.

Run this script in the notebook after training the BPE model,
or copy the merges dictionary and run this script separately.

Usage in notebook:
    # After training (when merges dictionary is available):
    exec(open('save_merges.py').read())
    save_merges_to_file(merges, 'merges.json', vocab_size=5000)
"""

import json
from typing import Dict, Tuple


def save_merges_to_file(
    merges: Dict[Tuple[int, int], int],
    filepath: str = "merges.json",
    vocab_size: int = 5000
):
    """
    Save the merges dictionary to a JSON file.
    
    Args:
        merges: Dictionary mapping byte pairs (tuple) to token IDs
        filepath: Path to save the JSON file
        vocab_size: Total vocabulary size
    """
    # Convert tuple keys to strings for JSON serialization
    merges_json = {f"{k[0]},{k[1]}": v for k, v in merges.items()}
    
    data = {
        "merges": merges_json,
        "vocab_size": vocab_size,
        "num_merges": len(merges),
        "description": "BPE merges trained on Hindi Wikipedia dataset (1000 articles)"
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Saved {len(merges)} merges to {filepath}")
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Number of merges: {len(merges)}")


# Example: If you have merges in the notebook, run:
# save_merges_to_file(merges, 'merges.json', vocab_size=5000)

