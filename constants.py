# Constants
MAX_TAXA = 16  # Maximum number of taxa (0-255)
VOCAB_SIZE = MAX_TAXA + 3  # 256 taxa + 2 special tokens (INTERNAL_NODE, EOS)
MAX_SEQUENCE_LENGTH = 256  # Following GPT2-small
MAX_GTREES = 300  # or a smaller/larger limit you know won't exceed the actual data

# Special tokens (match tokenizer values)
INTERNAL_NODE = VOCAB_SIZE - 2  # Internal node token
EOS = VOCAB_SIZE - 1  # End of sequence token
PAD = VOCAB_SIZE - 3  # Padding token
