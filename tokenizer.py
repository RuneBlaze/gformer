from typing import List

from constants import EOS, INTERNAL_NODE, PAD, VOCAB_SIZE


class NewickTokenizer:
    """
    Grammar-based tokenizer for Newick tree strings with taxa labels 0-255.
    This tokenizer emits tokens for internal nodes and leaves following a pre-order traversal.
    """

    INTERNAL_NODE = INTERNAL_NODE
    EOS = EOS
    PAD = PAD  # Add PAD token constant

    def __init__(self):
        self.vocab_size = VOCAB_SIZE  # Use VOCAB_SIZE from constants.py

    def encode(self, newick_str: str) -> List[int]:
        """
        Encode a Newick tree string into a list of tokens using a grammar-based approach.

        Args:
            newick_str: A Newick format tree string (e.g., "(1,(2,3));")

        Returns:
            List of integer tokens

        Raises:
            ValueError: If a leaf node value is negative or exceeds the maximum vocabulary size
        """

        def parse_subtree(index: int) -> (List[int], int):
            tokens = []
            if newick_str[index] == "(":
                tokens.append(self.INTERNAL_NODE)
                index += 1
                left_tokens, index = parse_subtree(index)
                tokens.extend(left_tokens)
                index += 1  # Skip the comma
                right_tokens, index = parse_subtree(index)
                tokens.extend(right_tokens)
                index += 1  # Skip the closing parenthesis
            else:
                # Parse a leaf label
                number = ""
                while index < len(newick_str) and newick_str[index].isdigit():
                    number += newick_str[index]
                    index += 1
                leaf_value = int(number)
                if leaf_value < 0 or leaf_value >= self.vocab_size - 3:  # -3 for INTERNAL_NODE, EOS, and PAD
                    raise ValueError(f"Leaf value {leaf_value} is outside valid range [0, {self.vocab_size - 4}]")
                tokens.append(leaf_value)
            return tokens, index

        tokens, index = parse_subtree(0)
        tokens.append(self.EOS)  # Add EOS token at the end
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """
        Decode a list of tokens back into a Newick tree string.
        Ignores PAD tokens during decoding.

        Args:
            tokens: List of integer tokens

        Returns:
            Newick format tree string
        """

        def build_subtree(index: int) -> (str, int):
            # Skip any PAD tokens
            while index < len(tokens) and tokens[index] == self.PAD:
                index += 1

            if index >= len(tokens):
                return "", index

            if tokens[index] == self.INTERNAL_NODE:
                index += 1
                left_subtree, index = build_subtree(index)
                right_subtree, index = build_subtree(index)
                subtree = f"({left_subtree},{right_subtree})"
            else:
                # Leaf node
                subtree = str(tokens[index])
                index += 1
            return subtree, index

        tree, _ = build_subtree(0)
        return tree + ";"  # Append the Newick end character

    def encode_batch(self, newick_strings: List[str]) -> List[List[int]]:
        """
        Encode a batch of Newick strings.

        Args:
            newick_strings: List of Newick format tree strings

        Returns:
            List of lists of integer tokens
        """
        return [self.encode(s) for s in newick_strings]

    def decode_batch(self, token_batches: List[List[int]]) -> List[str]:
        """
        Decode a batch of token lists.

        Args:
            token_batches: List of lists of integer tokens

        Returns:
            List of Newick format tree strings
        """
        return [self.decode(tokens) for tokens in token_batches]


if __name__ == "__main__":
    # Example usage
    stree = "((10,13),(((4,9),(2,(11,(12,14)))),((6,(1,(15,5))),(0,(7,(3,8))))));"
    tokenizer = NewickTokenizer()
    print("Original tree:", stree)

    encoded = tokenizer.encode(stree)
    print("Encoded tokens:", encoded)

    decoded = tokenizer.decode(encoded)
    print("Decoded tree:", decoded)
    print([17, 10, 13, 17, 17, 17, 4, 9, 17, 2, 17, 11, 17, 12, 14, 17, 17, 6, 17, 1, 17, 15, 5, 17, 0, 17, 7, 17, 3, 8])
    print(tokenizer.decode(
        [17, 10, 13, 17, 17, 17, 4, 9, 17, 2, 17, 11, 17, 12, 14, 17, 17, 6, 17, 1, 17, 15, 5, 17, 0, 17, 7, 17, 3, 8]
    ))
