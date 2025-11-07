"""
Byte Pair Encoding (BPE) Encoder for Hindi Text

This module provides a standalone BPE encoder/decoder that can be used
in Hugging Face Spaces or other applications.

Usage:
    from bpe_encoder import BPEEncoder
    
    # Load encoder with saved merges
    encoder = BPEEncoder.from_file("merges.json")
    
    # Encode text
    tokens = encoder.encode("हम होंगे कामयाब")
    
    # Decode tokens
    text = encoder.decode(tokens)
"""

import json
from typing import List, Dict, Tuple, Optional


class BPEEncoder:
    """
    Byte Pair Encoding encoder/decoder.
    
    This class implements BPE tokenization for Hindi text, trained on
    the Hindi Wikipedia dataset with a vocabulary size of 5000 tokens.
    """
    
    def __init__(self, merges: Dict[Tuple[int, int], int], vocab_size: int = 5000):
        """
        Initialize the BPE encoder.
        
        Args:
            merges: Dictionary mapping byte pairs (tuple) to token IDs
            vocab_size: Total vocabulary size (default: 5000)
        """
        self.merges = merges
        self.vocab_size = vocab_size
        self._vocab = None
        self._build_vocab()
    
    def _build_vocab(self):
        """Build the vocabulary dictionary from merges."""
        # Start with base 256 byte tokens
        vocab = {idx: bytes([idx]) for idx in range(256)}
        
        # Build vocabulary by applying merges in order
        # We need to sort merges by their token ID to apply in correct order
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        
        for (p0, p1), idx in sorted_merges:
            vocab[idx] = vocab[p0] + vocab[p1]
        
        self._vocab = vocab
    
    @staticmethod
    def _get_stats(ids: List[int]) -> Dict[Tuple[int, int], int]:
        """Count frequency of all consecutive byte pairs."""
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    @staticmethod
    def _merge(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        """Replace all occurrences of a pair with a new token ID."""
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def encode(self, text: str) -> List[int]:
        """
        Encode a text string into a list of token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs
        """
        # Start with UTF-8 byte encoding
        tokens = list(text.encode("utf-8"))
        
        # Apply merges greedily (in order of creation)
        while len(tokens) >= 2:
            stats = self._get_stats(tokens)
            
            # Find the pair with the lowest merge index (earliest merge)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            
            if pair not in self.merges:
                break  # No more merges possible
            
            idx = self.merges[pair]
            tokens = self._merge(tokens, pair, idx)
        
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token IDs back to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        # Concatenate byte sequences for each token
        tokens = b"".join(self._vocab[idx] for idx in ids)
        
        # Decode from UTF-8
        text = tokens.decode("utf-8", errors="replace")
        return text
    
    def save(self, filepath: str):
        """
        Save the encoder (merges dictionary) to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        # Convert tuple keys to lists for JSON serialization
        merges_json = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        
        data = {
            "merges": merges_json,
            "vocab_size": self.vocab_size,
            "num_merges": len(self.merges)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def from_file(cls, filepath: str) -> 'BPEEncoder':
        """
        Load encoder from a saved JSON file.
        
        Args:
            filepath: Path to the JSON file containing merges
            
        Returns:
            BPEEncoder instance
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert string keys back to tuples
        merges = {}
        for key, value in data["merges"].items():
            k1, k2 = map(int, key.split(','))
            merges[(k1, k2)] = value
        
        return cls(merges, data.get("vocab_size", 5000))
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.vocab_size
    
    def get_num_merges(self) -> int:
        """Get the number of merges."""
        return len(self.merges)


# Example usage
if __name__ == "__main__":
    # This is just for testing - in practice, load from saved file
    print("BPE Encoder for Hindi Text")
    print("To use this encoder:")
    print("1. Save merges from notebook using save_merges.py")
    print("2. Load encoder: encoder = BPEEncoder.from_file('merges.json')")
    print("3. Encode: tokens = encoder.encode('हम होंगे कामयाब')")
    print("4. Decode: text = encoder.decode(tokens)")

