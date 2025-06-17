import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np

# -----------------------------
# Dynamic Graph Learning Module
# -----------------------------
class DynamicGraphLearner(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(input_dim, hidden_dim)
        self.k_proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        # x: [B, N, D]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        A = torch.matmul(Q, K.transpose(-1, -2)) / (Q.shape[-1] ** 0.5)
        return torch.softmax(A, dim=-1)  # [B, N, N]

# -----------------------------
# Transformer Encoder Module
# -----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim_model, nhead=4, num_layers=2, hidden_ff=128):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model=dim_model, nhead=nhead, dim_feedforward=hidden_ff, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)  # [B, N, D]

# -----------------------------
# Main ComBrainTF++ Model
# -----------------------------
class ComBrainTFPlus(nn.Module):
    def __init__(self, num_rois=200, input_dim=200, hidden_dim=64, num_clusters=8):
        super().__init__()
        self.num_clusters = num_clusters
        self.hidden_dim = hidden_dim

        # Project raw FC input to embedding space
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Learnable graph
        self.graph_learner = DynamicGraphLearner(hidden_dim, hidden_dim)

        # Community & Global transformers
        self.local_transformer = TransformerBlock(hidden_dim)
        self.global_transformer = TransformerBlock(hidden_dim)

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_clusters, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Binary: ASD vs HC
        )

    def cluster_nodes(self, x, num_clusters=8):
        # x: [B, N, D]
        labels_batch = []
        for i in range(x.size(0)):
            xi = x[i].detach().cpu().numpy()
            km = KMeans(n_clusters=num_clusters, n_init=10)
            labels = km.fit_predict(xi)
            labels_batch.append(labels)
        return labels_batch  # List of length B with N labels

    def forward(self, x):
        """
        x: Functional Connectivity (FC) matrix or ROI features [B, N, N]
        """
        B, N, _ = x.shape

        # Step 1: Feature Projection
        x = self.input_proj(x)  # [B, N, D]

        # Step 2: Dynamic Graph Learning (optional visualization)
        adj = self.graph_learner(x)  # [B, N, N]

        # Step 3: Adaptive Community Discovery
        cluster_labels = self.cluster_nodes(x, self.num_clusters)

        # Step 4: Community-wise Aggregation
        community_reps = torch.zeros(B, self.num_clusters, self.hidden_dim).to(x.device)
        for b in range(B):
            for c in range(self.num_clusters):
                labels_b = cluster_labels[b]  # numpy array
                mask = torch.from_numpy((labels_b == c).astype(np.float32)).to(x.device)
                if mask.sum() > 0:
                    weighted = mask.unsqueeze(-1) * x[b]  # [N, D]
                    community_reps[b, c] = weighted.sum(dim=0) / (mask.sum() + 1e-6)

        # Step 5: Local and Global Transformers
        local_encoded = self.local_transformer(community_reps)  # [B, K, D]
        global_encoded = self.global_transformer(local_encoded)  # [B, K, D]

        # Step 6: Classification
        flat = global_encoded.reshape(B, -1)  # [B, K*D]
        output = self.classifier(flat)  # [B, 2]
        return output
    
    def get_assign_mat(self):
        return None

    def get_attention_weights(self):
        return [None]