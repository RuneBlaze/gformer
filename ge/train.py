import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from accelerate import Accelerator
from ge.encoder import GeneTreeEncoder, QuartetClassifier
from ge.data import GeneTreeDataset

def train(
    pkl_path: str,
    num_species: int = 16,
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
):
    # Initialize accelerator
    accelerator = Accelerator()

    # Initialize dataset and dataloader
    dataset = GeneTreeDataset(pkl_path, m=num_species, seed=42)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )

    # Initialize models
    encoder = GeneTreeEncoder()
    classifier = QuartetClassifier()

    # Initialize optimizer
    optimizer = Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=learning_rate
    )

    # Prepare everything with accelerator
    encoder, classifier, optimizer, dataloader = accelerator.prepare(
        encoder, classifier, optimizer, dataloader
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(dataloader):
            # Get batch data
            distance_matrix = batch["distance_matrix"]
            quartet_queries = batch["quartet_queries"]
            quartet_topologies = batch["quartet_topologies"]

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            gene_repr = encoder(distance_matrix)  # [batch, gene_embed_dim]
            logits = classifier(gene_repr, quartet_queries)  # [batch, num_queries, num_quartet_classes]
            
            # Flatten the output for loss calculation.
            batch_size, num_queries, num_classes = logits.shape
            logits_flat = logits.view(batch_size * num_queries, num_classes)
            targets_flat = quartet_topologies.view(batch_size * num_queries)
            
            # Calculate loss over all quartet queries
            loss = criterion(logits_flat, targets_flat)

            # Backward pass using accelerator
            accelerator.backward(loss)
            optimizer.step()

            # Calculate accuracy
            with torch.no_grad():
                # Get predictions per query.
                predictions = torch.argmax(logits, dim=-1)  # [batch, num_queries]
                correct += (predictions == quartet_topologies).sum().item()
                total += quartet_topologies.numel()
                total_loss += loss.item()

            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                accelerator.print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Batch [{batch_idx+1}/{len(dataloader)}], "
                    f"Loss: {loss.item():.4f}, "
                    f"Accuracy: {100 * correct/total:.2f}%"
                )

        # Print epoch statistics
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        accelerator.print(
            f"Epoch [{epoch+1}/{num_epochs}] - "
            f"Average Loss: {avg_loss:.4f}, "
            f"Accuracy: {accuracy:.2f}%"
        )

    # Unwrap models before returning
    encoder = accelerator.unwrap_model(encoder)
    classifier = accelerator.unwrap_model(classifier)
    return encoder, classifier

if __name__ == "__main__":
    # Replace with your actual path to the pickle file
    PKL_PATH = "/Users/lbq/goof/teedeelee/assets/processed_family.pkl"
    
    encoder, classifier = train(
        pkl_path=PKL_PATH,
        num_species=16,
        batch_size=32,
        num_epochs=10,
    )

    # Optionally save the trained models
    torch.save(encoder.state_dict(), "encoder.pt")
    torch.save(classifier.state_dict(), "classifier.pt") 