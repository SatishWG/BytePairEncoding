# BPE Encoder for Hugging Face Spaces

This directory contains the Byte Pair Encoding (BPE) encoder implementation that can be deployed to Hugging Face Spaces.

## Files for Hugging Face Spaces

### Required Files:
1. **`bpe_encoder.py`** - Main encoder class (standalone, no dependencies)
2. **`merges.json`** - Trained merges dictionary (must be generated from notebook)
3. **`app.py`** - Gradio interface for the Space (create this)

### Optional Files:
- `example_usage.py` - Example of how to use the encoder
- `requirements.txt` - Python dependencies (if needed)

## Setup Instructions

### Step 1: Generate merges.json

In your notebook, after training the BPE model, run this code:

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

### Step 2: Create app.py for Gradio

Create a `app.py` file with this content:

```python
import gradio as gr
from bpe_encoder import BPEEncoder

# Load encoder
encoder = BPEEncoder.from_file("merges.json")

def encode_text(text):
    if not text:
        return "Please enter some text."
    
    tokens = encoder.encode(text)
    return {
        "tokens": tokens,
        "num_tokens": len(tokens),
        "original_bytes": len(text.encode('utf-8')),
        "compression": f"{len(text.encode('utf-8')) / len(tokens):.2f}X" if tokens else "N/A"
    }

def decode_tokens(tokens_str):
    if not tokens_str:
        return "Please enter token IDs."
    
    try:
        # Parse comma-separated token IDs
        tokens = [int(x.strip()) for x in tokens_str.split(',')]
        text = encoder.decode(tokens)
        return text
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# BPE Encoder for Hindi Text")
    gr.Markdown("Byte Pair Encoding tokenizer trained on Hindi Wikipedia dataset")
    
    with gr.Tab("Encode"):
        text_input = gr.Textbox(
            label="Input Text (Hindi)",
            placeholder="Enter Hindi text here...",
            lines=5
        )
        encode_btn = gr.Button("Encode")
        encode_output = gr.JSON(label="Encoded Tokens")
        
        encode_btn.click(
            fn=encode_text,
            inputs=text_input,
            outputs=encode_output
        )
    
    with gr.Tab("Decode"):
        tokens_input = gr.Textbox(
            label="Token IDs (comma-separated)",
            placeholder="256, 257, 258, ...",
            lines=3
        )
        decode_btn = gr.Button("Decode")
        decode_output = gr.Textbox(label="Decoded Text", lines=5)
        
        decode_btn.click(
            fn=decode_tokens,
            inputs=tokens_input,
            outputs=decode_output
        )
    
    with gr.Tab("Encode & Decode"):
        text_input2 = gr.Textbox(
            label="Input Text",
            placeholder="Enter Hindi text here...",
            lines=3
        )
        roundtrip_btn = gr.Button("Encode & Decode")
        roundtrip_output = gr.Textbox(label="Round-trip Result", lines=5)
        
        def roundtrip(text):
            tokens = encoder.encode(text)
            decoded = encoder.decode(tokens)
            return f"Original: {text}\n\nDecoded: {decoded}\n\nMatch: {text == decoded}"
        
        roundtrip_btn.click(
            fn=roundtrip,
            inputs=text_input2,
            outputs=roundtrip_output
        )

demo.launch()
```

### Step 3: Create requirements.txt

```txt
gradio>=4.0.0
```

### Step 4: Upload to Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Upload these files:
   - `bpe_encoder.py`
   - `merges.json`
   - `app.py`
   - `requirements.txt`
3. Set the Space SDK to "Gradio"
4. The Space will automatically deploy!

## File Structure for Hugging Face Spaces

```
your-space/
├── app.py              # Gradio interface
├── bpe_encoder.py      # BPE encoder class
├── merges.json         # Trained merges (generated from notebook)
├── requirements.txt    # Dependencies
└── README.md           # Space description
```

## Testing Locally

Before uploading, test locally:

```bash
pip install gradio
python app.py
```

Then open http://localhost:7860 in your browser.

## Notes

- The `bpe_encoder.py` file is standalone and has no external dependencies (except standard library)
- The `merges.json` file contains the trained model weights (merges dictionary)
- Make sure `merges.json` is generated with the correct vocabulary size (5000)
- The encoder works with Hindi text but can be adapted for other languages

