from typing import List, Union

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
        """
        # Remove leading/trailing whitespace and trailing semicolon if present
        import treeswift as ts
        parsed = ts.read_tree_newick(newick_str)
        parsed.suppress_unifurcations()
        newick_str = parsed.newick()
        if '[&R]' in newick_str:
            newick_str = newick_str.replace('[&R]', '')
        newick_str = newick_str.strip()
        if newick_str.endswith(";"):
            newick_str = newick_str[:-1]

        def parse_subtree(index: int) -> (List[int], int):
            # Skip whitespace (if any)
            while index < len(newick_str) and newick_str[index].isspace():
                index += 1
            tokens = []
            if newick_str[index] == "(":
                tokens.append(self.INTERNAL_NODE)
                index += 1  # Skip the opening parenthesis
                # Parse left subtree
                left_tokens, index = parse_subtree(index)
                tokens.extend(left_tokens)
                # Skip whitespace
                while index < len(newick_str) and newick_str[index].isspace():
                    index += 1
                # Expect a comma delimiter
                if index >= len(newick_str) or newick_str[index] != ",":
                    raise ValueError(f"Expected ',' at position {index} in {newick_str}")
                index += 1  # Skip the comma
                # Parse right subtree
                right_tokens, index = parse_subtree(index)
                tokens.extend(right_tokens)
                # Skip whitespace
                while index < len(newick_str) and newick_str[index].isspace():
                    index += 1
                # Expect the closing parenthesis
                if index >= len(newick_str) or newick_str[index] != ")":
                    raise ValueError(f"Expected ')' at position {index} in {newick_str}")
                index += 1  # Skip the closing parenthesis
            else:
                # Parse a leaf label (a number)
                number = ""
                while index < len(newick_str) and newick_str[index].isdigit():
                    number += newick_str[index]
                    index += 1
                if number == "":
                    raise ValueError(f"Expected a digit at position {index} in {newick_str}")
                tokens.append(int(number))
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
    stree = "(((1,14),(9,6)),((((15,(0,10)),(5,4)),12),((((7,11),13),(2,8)),3)));"
    tokenizer = NewickTokenizer()
    print("Original tree:", stree)

    encoded = tokenizer.encode(stree)
    print("Encoded tokens:", encoded)

    decoded = tokenizer.decode(encoded)
    print("Decoded tree:", decoded)
