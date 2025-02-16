import torch
import torch.nn as nn
import torch.nn.functional as F
import math



# For N = 16 species, there are 16 choose 2 = 120 pairs.
NUM_SPECIES = 16
NUM_PAIRS = (NUM_SPECIES * (NUM_SPECIES - 1)) // 2  # = 120

def generate_pair_indices(n):
    pair_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_indices.append((i, j))
    return pair_indices

class GeneTreeEncoder(nn.Module):
    """
    Encoder-only architecture for a gene tree.
    For ablation purposes, this encoder simply flattens the input and applies an MLP
    projection to produce a gene tree representation.
    
    Input: distance matrix patches of shape [batch, NUM_PAIRS, bits]
    Output: gene tree representation of shape [batch, embed_dim]
    """
    def __init__(self, bits=8, num_pairs=NUM_PAIRS, embed_dim=256):
        super().__init__()
        self.bits = bits
        self.embed_dim = embed_dim
        self.num_pairs = num_pairs
        self.flattened_dim = num_pairs * bits
        
        # Updated MLP: now a two-layer MLP with a SiLU activation
        self.mlp = nn.Sequential(
            nn.Linear(self.flattened_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self._init_weights()

    def _init_weights(self):
        # Update the initialization for all Linear layers in the MLP.
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        """
        x: Tensor of shape [batch, NUM_PAIRS, bits]
        Returns: Tensor of shape [batch, embed_dim]
        
        For ablation, the forward method simply flattens the binary-encoded distance matrix
        and projects it through an MLP.
        """
        batch_size = x.size(0)
        # Flatten the input: [batch, NUM_PAIRS, bits] -> [batch, NUM_PAIRS * bits]
        x_flat = x.view(batch_size, -1)  # shape: [batch, num_pairs * bits]
        output = self.mlp(x_flat)        # shape: [batch, embed_dim]
        return output


class QuartetClassifier(nn.Module):
    """
    Given a gene tree embedding and a query quartet (four species indices),
    predict which of the three possible unrooted quartet topologies is present.
    
    The classifier first looks up embeddings for each species (learned separately)
    and then concatenates them with the gene tree representation.
    """
    def __init__(self, gene_embed_dim=256, species_embed_dim=64,
                 hidden_dim=128, num_quartet_classes=3, num_species=NUM_SPECIES):
        super().__init__()
        # Learn an embedding for each species.
        self.species_embedding = nn.Embedding(num_species, species_embed_dim)
        # Updated MLP in the classifier: simplified to a single linear layer.
        self.mlp = nn.Linear(gene_embed_dim + 4 * species_embed_dim, num_quartet_classes)
    
    def forward(self, gene_repr, quartet_indices):
        """
        gene_repr: Tensor of shape [batch, gene_embed_dim] from the GeneTreeEncoder.
        quartet_indices: Tensor of shape [batch, num_queries, 4] or [batch, 4].
        """
        # If quartet_indices is 2D ([batch, 4]), add a queries dimension.
        if quartet_indices.ndim == 2:
            quartet_indices = quartet_indices.unsqueeze(1)  # Now shape [batch, 1, 4]
        
        batch_size, num_queries, _ = quartet_indices.size()
        
        # Look up species embeddings. Result: [batch, num_queries, 4, species_embed_dim]
        species_embeds = self.species_embedding(quartet_indices)
        # Flatten the species embeddings per query to: [batch, num_queries, 4 * species_embed_dim]
        species_embeds = species_embeds.view(batch_size, num_queries, -1)
        
        # Expand gene_repr to match the number of queries: [batch, num_queries, gene_embed_dim]
        gene_repr_expanded = gene_repr.unsqueeze(1).expand(-1, num_queries, -1)
        
        # Concatenate the gene tree representation with species query embeddings.
        combined = torch.cat([gene_repr_expanded, species_embeds], dim=-1)  # [batch, num_queries, gene_embed_dim + 4*species_embed_dim]
        
        # Flatten out the batch and query dimensions to feed into the MLP.
        combined_flat = combined.view(batch_size * num_queries, -1)
        logits_flat = self.mlp(combined_flat)  # [batch * num_queries, num_quartet_classes]
        
        # Reshape back to [batch, num_queries, num_quartet_classes].
        logits = logits_flat.view(batch_size, num_queries, -1)
        return logits
