"""
Copy this code into a new cell in your notebook after training the BPE model.

This will save the merges dictionary to a JSON file that can be used
with the standalone BPE encoder.
"""

# After training (when merges dictionary is available):
import json

# Convert merges dictionary to JSON format
merges_json = {f"{k[0]},{k[1]}": v for k, v in merges.items()}

data = {
    "merges": merges_json,
    "vocab_size": 5000,
    "num_merges": len(merges),
    "description": "BPE merges trained on Hindi Wikipedia dataset (1000 articles)"
}

# Save to file
with open("merges.json", 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print(f"✅ Saved {len(merges)} merges to merges.json")
print(f"   Vocabulary size: 5000")
print(f"   Number of merges: {len(merges)}")

