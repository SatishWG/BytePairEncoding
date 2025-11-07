"""
Example usage of the BPE encoder.

This demonstrates how to use the BPE encoder in a Hugging Face Space
or other application.
"""

from bpe_encoder import BPEEncoder


def main():
    # Load the encoder from saved merges file
    # Make sure merges.json exists (created by running save_merges.py in notebook)
    try:
        encoder = BPEEncoder.from_file("merges.json")
        print(f"✅ Loaded encoder with vocabulary size: {encoder.get_vocab_size()}")
        print(f"   Number of merges: {encoder.get_num_merges()}")
    except FileNotFoundError:
        print("❌ merges.json not found!")
        print("   Please run save_merges.py in the notebook first to create merges.json")
        return
    
    # Example text in Hindi
    text = "हम होंगे कामयाब"
    print(f"\n📝 Original text: {text}")
    
    # Encode text to tokens
    tokens = encoder.encode(text)
    print(f"🔢 Encoded tokens: {tokens}")
    print(f"   Number of tokens: {len(tokens)}")
    print(f"   Original text length (bytes): {len(text.encode('utf-8'))}")
    print(f"   Compression: {len(text.encode('utf-8')) / len(tokens):.2f}X")
    
    # Decode tokens back to text
    decoded_text = encoder.decode(tokens)
    print(f"\n📖 Decoded text: {decoded_text}")
    print(f"✅ Round-trip successful: {text == decoded_text}")


if __name__ == "__main__":
    main()

