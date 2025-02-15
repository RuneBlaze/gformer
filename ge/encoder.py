import torch
import torch.nn as nn
import torch.nn.functional as F



# For N = 16 species, there are 16 choose 2 = 120 pairs.
NUM_SPECIES = 16
NUM_PAIRS = (NUM_SPECIES * (NUM_SPECIES - 1)) // 2  # = 120


class GeneTreeEncoder(nn.Module):
    """
    Encoder-only architecture for a gene tree.
    Input: binary-encoded distance matrix patches of shape [batch, NUM_PAIRS, bits] (bits=8)
    We prepend a <CLS> token, add learned positional embeddings,
    and process with a Transformer encoder.
    The output at the <CLS> token becomes the gene tree representation.
    """
    def __init__(self, bits=8, num_pairs=NUM_PAIRS, embed_dim=256,
                 num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.bits = bits
        self.embed_dim = embed_dim

        # Embed each binary vector (length bits) into an embedding vector.
        self.token_embedding = nn.Linear(bits, embed_dim)

        # Learnable <CLS> token.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learned positional embedding for the sequence tokens (including CLS).
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_pairs + 1, embed_dim))

        # Transformer encoder. (Using PyTorch's transformer encoder layer with batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.xavier_uniform_(self.token_embedding.weight)
        if self.token_embedding.bias is not None:
            nn.init.zeros_(self.token_embedding.bias)
        
    def forward(self, x):
        """
        x: Tensor of shape [batch, NUM_PAIRS, bits]
        """
        batch_size = x.size(0)
        # Map each binary patch (vector of length bits) to embedding.
        token_embeds = self.token_embedding(x)  # [batch, NUM_PAIRS, embed_dim]
        # Expand and prepend the CLS token.
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, embed_dim]
        tokens = torch.cat([cls_tokens, token_embeds], dim=1)  # [batch, NUM_PAIRS+1, embed_dim]
        # Add positional embedding.
        tokens = tokens + self.pos_embedding  # broadcasting over batch
        
        # Transformer encoder.
        encoded = self.transformer(tokens)  # [batch, NUM_PAIRS+1, embed_dim]
        encoded = self.norm(encoded)
        # Return the representation at the CLS token.
        return encoded[:, 0, :]  # shape: [batch, embed_dim]


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
        # MLP that takes the concatenated representation and outputs logits.
        self.mlp = nn.Sequential(
            nn.Linear(gene_embed_dim + 4 * species_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_quartet_classes)
        )
    
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
