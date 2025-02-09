import argparse
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import TreeDataset
from layers import ModelConfig, TreeTransformer
from tokenizer import NewickTokenizer
from constants import EOS, MAX_TAXA

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with TreeTransformer model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model config YAML"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data file (.jsonl or .parquet)"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of examples to show"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference"
    )
    return parser.parse_args()

def show_completions(model, val_loader, device, tokenizer, num_examples=10):
    """Show example completions from the model with constrained generation"""
    model.eval()
    examples_shown = 0
    val_iter = iter(val_loader)

    with torch.no_grad():
        while examples_shown < num_examples:
            try:
                tree_encodings, target_seq = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                tree_encodings, target_seq = next(val_iter)

            tree_encodings = tree_encodings.to(device)
            target_seq = target_seq.to(device)

            for i in range(min(num_examples - examples_shown, target_seq.size(0))):
                # Get ground truth completion
                gt_completion = tokenizer.decode(target_seq[i].cpu().tolist())

                # Start with just first token (usually INTERNAL_NODE)
                curr_seq = target_seq[i:i+1, :1]

                # Track which taxa have been seen
                seen_taxa = set()

                # Generate tokens until we hit EOS or max length
                for pos in range(512):
                    # Create causal mask for current sequence length
                    attention_mask = torch.triu(
                        torch.ones(curr_seq.size(1), curr_seq.size(1)), 
                        diagonal=1
                    ).bool().to(device)

                    # Get next token prediction
                    logits = model(
                        tree_encodings=tree_encodings[i:i+1],
                        output_tokens=curr_seq,
                        attention_mask=attention_mask,
                    )

                    # Get probabilities for next token
                    probs = F.softmax(logits[:, -1:, :], dim=-1)

                    # Modify probabilities based on constraints
                    if len(seen_taxa) < MAX_TAXA:
                        # Set EOS probability to 0 until all taxa are seen
                        probs[0, 0, EOS] = 0

                        # Set probabilities of seen taxa to 0 to prevent repeats
                        for taxon in seen_taxa:
                            probs[0, 0, taxon] = 0

                        # Renormalize probabilities
                        probs = probs / probs.sum()

                    # Get most likely next token
                    next_token = probs.argmax(dim=-1)
                    next_token_val = next_token.item()

                    # Update seen taxa if this is a taxon
                    if 0 <= next_token_val < MAX_TAXA:
                        seen_taxa.add(next_token_val)

                    # Break if we hit EOS and all taxa have been seen
                    if next_token_val == EOS and len(seen_taxa) >= MAX_TAXA:
                        break

                    # Add predicted token to sequence
                    curr_seq = torch.cat([curr_seq, next_token], dim=1)

                pred_completion = tokenizer.decode(curr_seq[0].cpu().tolist())

                # Print comparison
                print(f"\nExample {examples_shown + 1}:")
                print(f"Ground truth: {gt_completion}")
                print(f"Prediction:   {pred_completion}")
                print(f"Taxa seen: {sorted(list(seen_taxa))}")

                examples_shown += 1
                if examples_shown >= num_examples:
                    break

def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model config and initialize model
    model_config = ModelConfig.from_yaml(args.config)
    model = TreeTransformer(model_config).to(device)

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Checkpoint loaded successfully")

    # Initialize tokenizer
    tokenizer = NewickTokenizer()

    # Create validation dataset and dataloader
    val_dataset = TreeDataset(
        args.data,
        seed=args.seed ^ 0xdeadbeef,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
    )

    # Run inference
    logger.info(f"Running inference on {args.num_examples} examples...")
    model.eval()
    show_completions(model, val_loader, device, tokenizer, args.num_examples)

if __name__ == "__main__":
    main() 