"""
Gradio app for BPE Encoder - Ready for Hugging Face Spaces

This creates a web interface for the BPE encoder that can be deployed
to Hugging Face Spaces.
"""

import gradio as gr
from bpe_encoder import BPEEncoder

# Load encoder
try:
    encoder = BPEEncoder.from_file("merges.json")
    print(f"✅ Loaded encoder with vocabulary size: {encoder.get_vocab_size()}")
except FileNotFoundError:
    print("❌ merges.json not found! Please generate it from the notebook.")
    encoder = None


def encode_text(text):
    """Encode text to tokens."""
    if not encoder:
        return {"error": "Encoder not loaded. Please check merges.json file."}
    
    if not text:
        return {"error": "Please enter some text."}
    
    try:
        tokens = encoder.encode(text)
        original_bytes = len(text.encode('utf-8'))
        compression = original_bytes / len(tokens) if tokens else 0
        
        return {
            "tokens": tokens,
            "num_tokens": len(tokens),
            "original_bytes": original_bytes,
            "compression_ratio": f"{compression:.2f}X"
        }
    except Exception as e:
        return {"error": str(e)}


def decode_tokens(tokens_str):
    """Decode tokens back to text."""
    if not encoder:
        return "❌ Encoder not loaded. Please check merges.json file."
    
    if not tokens_str:
        return "Please enter token IDs."
    
    try:
        # Parse comma-separated token IDs
        tokens = [int(x.strip()) for x in tokens_str.split(',') if x.strip()]
        text = encoder.decode(tokens)
        return text
    except ValueError:
        return "❌ Error: Invalid token format. Please use comma-separated integers (e.g., 256, 257, 258)"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def roundtrip(text):
    """Encode and decode text to verify round-trip."""
    if not encoder:
        return "❌ Encoder not loaded. Please check merges.json file."
    
    if not text:
        return "Please enter some text."
    
    try:
        tokens = encoder.encode(text)
        decoded = encoder.decode(tokens)
        match = text == decoded
        
        result = f"""**Original Text:**
{text}

**Encoded Tokens:** {tokens[:20]}{'...' if len(tokens) > 20 else ''}
**Number of Tokens:** {len(tokens)}

**Decoded Text:**
{decoded}

**Round-trip Match:** {'✅ Yes' if match else '❌ No'}"""
        
        return result
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="BPE Encoder for Hindi Text") as demo:
    gr.Markdown("""
    # 🔤 Byte Pair Encoding (BPE) Tokenizer
    
    This is a BPE tokenizer trained on **Hindi Wikipedia dataset** with a vocabulary size of **5,000 tokens**.
    
    - **Compression Ratio:** 9.39X
    - **Number of Merges:** 4,744
    - **Original Token Length:** 18.36M bytes → **Compressed:** 1.96M tokens
    
    ### How to use:
    1. **Encode:** Enter Hindi text and get token IDs
    2. **Decode:** Enter comma-separated token IDs and get text back
    3. **Round-trip:** Test encoding and decoding to verify correctness
    """)
    
    with gr.Tab("📝 Encode"):
        gr.Markdown("### Encode Hindi text to token IDs")
        text_input = gr.Textbox(
            label="Input Text (Hindi)",
            placeholder="हम होंगे कामयाब",
            lines=5
        )
        encode_btn = gr.Button("Encode", variant="primary")
        encode_output = gr.JSON(label="Encoded Result")
        
        encode_btn.click(
            fn=encode_text,
            inputs=text_input,
            outputs=encode_output
        )
    
    with gr.Tab("🔓 Decode"):
        gr.Markdown("### Decode token IDs back to text")
        tokens_input = gr.Textbox(
            label="Token IDs (comma-separated)",
            placeholder="256, 257, 258, 259, ...",
            lines=3
        )
        decode_btn = gr.Button("Decode", variant="primary")
        decode_output = gr.Textbox(label="Decoded Text", lines=5)
        
        decode_btn.click(
            fn=decode_tokens,
            inputs=tokens_input,
            outputs=decode_output
        )
    
    with gr.Tab("🔄 Round-trip Test"):
        gr.Markdown("### Test encoding and decoding (verify correctness)")
        text_input2 = gr.Textbox(
            label="Input Text",
            placeholder="हम होंगे कामयाब",
            lines=3
        )
        roundtrip_btn = gr.Button("Encode & Decode", variant="primary")
        roundtrip_output = gr.Markdown(label="Round-trip Result")
        
        roundtrip_btn.click(
            fn=roundtrip,
            inputs=text_input2,
            outputs=roundtrip_output
        )
    
    gr.Markdown("""
    ---
    ### 📊 Model Information
    - **Training Data:** Hindi Wikipedia (1000 articles)
    - **Vocabulary Size:** 5,000 tokens
    - **Base Tokens:** 256 (UTF-8 bytes)
    - **Merges:** 4,744
    - **Compression:** 9.39X
    """)

if __name__ == "__main__":
    demo.launch()

