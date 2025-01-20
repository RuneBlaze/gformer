from typing import List, Union

import torch


class NewickTokenizer:
    """
    Tokenizer for Newick tree strings with taxa labels 0-255.

    Special tokens:
    - LEFT_PAREN: (
    - RIGHT_PAREN: )
    - EOI: End of input
    - EOS: End of sequence
    """

    # Special token values (placed after taxa labels 0-255)
    LEFT_PAREN = 256
    RIGHT_PAREN = 257
    EOI = 258  # End of input
    EOS = 259  # End of sequence

    def __init__(self):
        self.vocab_size = 260  # 256 taxa + 4 special tokens

    def encode(self, newick_str: str) -> List[int]:
        """
        Encode a Newick tree string into a list of token IDs.

        Args:
            newick_str: A Newick format tree string (e.g., "(0,(1,2))")

        Returns:
            List of integer token IDs
        """
        tokens = []
        current_number = ""

        for char in newick_str:
            if char == "(":
                tokens.append(self.LEFT_PAREN)
            elif char == ")":
                if current_number:
                    tokens.append(int(current_number))
                    current_number = ""
                tokens.append(self.RIGHT_PAREN)
            elif char.isdigit():
                current_number += char
            elif char in [",", " "]:
                if current_number:
                    tokens.append(int(current_number))
                    current_number = ""

        # Handle any remaining number
        if current_number:
            tokens.append(int(current_number))

        # Validate taxa labels
        for token in tokens:
            if isinstance(token, int) and token < 256:
                if token < 0 or token > 255:
                    raise ValueError(f"Taxa label {token} out of valid range [0-255]")

        # Add EOS token
        tokens.append(self.EOS)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """
        Decode a list of token IDs back into a Newick tree string.

        Args:
            tokens: List of integer token IDs

        Returns:
            Newick format tree string
        """
        result = []

        for token in tokens:
            if token == self.LEFT_PAREN:
                result.append("(")
            elif token == self.RIGHT_PAREN:
                result.append(")")
            elif token == self.EOS:
                break
            elif token == self.EOI:
                continue
            elif 0 <= token <= 255:
                result.append(str(token))
            else:
                raise ValueError(f"Invalid token ID: {token}")

        return ",".join("".join(result).split(","))

    def encode_batch(self, newick_strings: List[str]) -> torch.Tensor:
        """
        Encode a batch of Newick strings into a padded tensor.

        Args:
            newick_strings: List of Newick format tree strings

        Returns:
            torch.Tensor: Padded tensor of token IDs
        """
        encoded = [self.encode(s) for s in newick_strings]
        max_len = max(len(s) for s in encoded)

        # Pad with EOI token
        padded = [s + [self.EOI] * (max_len - len(s)) for s in encoded]
        return torch.tensor(padded, dtype=torch.long)
