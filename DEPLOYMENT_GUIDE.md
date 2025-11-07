# Deployment Guide: BPE Encoder to Hugging Face Spaces

This guide explains how to deploy the BPE encoder to Hugging Face Spaces.

## Quick Start

### Step 1: Generate merges.json from Notebook

Add this code cell to your notebook **after training** (when `merges` dictionary is available):

```python
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
```

### Step 2: Prepare Files for Upload

You need these files in your Hugging Face Space:

1. **`bpe_encoder.py`** ✅ (already created)
2. **`app.py`** ✅ (already created)
3. **`merges.json`** ⚠️ (generate from notebook)
4. **`requirements.txt`** ✅ (use `requirements_hf_spaces.txt`)

### Step 3: Create Hugging Face Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Fill in:
   - **Name:** `bpe-encoder-hindi` (or your choice)
   - **SDK:** `Gradio`
   - **Visibility:** Public or Private
4. Click "Create Space"

### Step 4: Upload Files

Upload these files to your Space:

```
your-space/
├── app.py                      # Gradio interface
├── bpe_encoder.py              # BPE encoder class
├── merges.json                 # Trained merges (from notebook)
├── requirements.txt            # Dependencies (copy from requirements_hf_spaces.txt)
└── README.md                   # Space description (optional)
```

**Note:** Copy `requirements_hf_spaces.txt` content to `requirements.txt` in the Space.

### Step 5: Deploy

The Space will automatically deploy! Check the "Logs" tab if there are any issues.

## File Descriptions

### `bpe_encoder.py`
- Standalone BPE encoder class
- No external dependencies (only standard library)
- Can encode/decode Hindi text
- Loads merges from JSON file

### `app.py`
- Gradio web interface
- Three tabs: Encode, Decode, Round-trip Test
- User-friendly UI for testing the encoder

### `merges.json`
- Trained model weights (merges dictionary)
- Generated from notebook after training
- Contains 4,744 merges for 5,000 token vocabulary

### `requirements.txt`
- Only needs: `gradio>=4.0.0`
- The encoder itself has no dependencies

## Testing Locally

Before uploading, test locally:

```bash
# Install dependencies
pip install gradio

# Run the app
python app.py
```

Then open http://localhost:7860 in your browser.

## Troubleshooting

### Error: "merges.json not found"
- Make sure you generated `merges.json` from the notebook
- Check that the file is uploaded to the Space

### Error: "Invalid token format"
- When decoding, use comma-separated integers: `256, 257, 258`
- No spaces or brackets needed

### Deployment fails
- Check the "Logs" tab in Hugging Face Spaces
- Ensure `requirements.txt` only contains `gradio>=4.0.0`
- Verify all files are uploaded correctly

## Example Usage

Once deployed, you can:

1. **Encode text:**
   - Input: `हम होंगे कामयाब`
   - Output: Token IDs and compression stats

2. **Decode tokens:**
   - Input: `256, 257, 258, 259`
   - Output: Decoded Hindi text

3. **Round-trip test:**
   - Input: Any Hindi text
   - Output: Verification that encoding/decoding works correctly

## Model Information

- **Vocabulary Size:** 5,000 tokens
- **Number of Merges:** 4,744
- **Compression Ratio:** 9.39X
- **Training Data:** Hindi Wikipedia (1000 articles)
- **Base Tokens:** 256 (UTF-8 bytes)

## Next Steps

After deployment, you can:
- Share the Space URL with others
- Embed it in websites
- Use the API for programmatic access
- Extend the interface with more features

