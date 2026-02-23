import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math

class SAE(nn.Module):
    def __init__(self, vocab_size, num_topics=50, en_units=200, dropout=0.4,
                 sub_embed_dim=None):
        """
        sub_embed_dim: dimensionality D of your sentence-transformer embeddings.
                       If None, attention is skipped (falls back to original SAE).
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.sub_embed_dim = sub_embed_dim  # D (e.g., 384 for all-MiniLM-L6-v2)

        # ---- prior buffers (unchanged) ----
        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T))
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topics))).T +
                                                  (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T))
        self.mu2.requires_grad = False
        self.var2.requires_grad = False

        # ---- bow encoder (UNCHANGED) ----
        self.fc11 = nn.Linear(vocab_size, en_units)
        self.fc12 = nn.Linear(en_units, sub_embed_dim)
        self.fc21 = nn.Linear(sub_embed_dim, num_topics)
        self.fc22 = nn.Linear(sub_embed_dim, num_topics)

        # batch norms (UNCHANGED)
        self.mean_bn = nn.BatchNorm1d(num_topics, affine=True)
        self.mean_bn.weight.data.copy_(torch.ones(num_topics))
        self.mean_bn.weight.requires_grad = False

        self.logvar_bn = nn.BatchNorm1d(num_topics, affine=True)
        self.logvar_bn.weight.data.copy_(torch.ones(num_topics))
        self.logvar_bn.weight.requires_grad = False

        self.decoder_bn = nn.BatchNorm1d(vocab_size, affine=True)
        self.decoder_bn.weight.data.copy_(torch.ones(vocab_size))
        self.decoder_bn.weight.requires_grad = False

        self.fc1_drop = nn.Dropout(dropout)
        self.theta_drop = nn.Dropout(dropout)

        # ---- decoder (UNCHANGED) ----
        self.fcd1 = nn.Linear(num_topics, vocab_size, bias=False)
        nn.init.xavier_uniform_(self.fcd1.weight)

        # ---- tiny projection ONLY IF dims differ (keeps code simple) ----
        # Query comes from fc12 output (size = en_units). Keys/Values are sub-embeddings (size = sub_embed_dim).
        # If sub_embed_dim is provided and != en_units, map query to D with a single linear.
        # if self.sub_embed_dim is not None and self.sub_embed_dim != en_units:
        #     self.q_proj = nn.Linear(en_units, self.sub_embed_dim)
        # else:
        #     self.q_proj = None  # no projection needed

    def get_beta(self):
        return self.fcd1.weight.T

    # ---------- NEW: sentence-aware attention (per-document) ----------
    @torch.no_grad()
    def _stack_subs(self, sub_list):
        """
        sub_list: list of (D,) tensors on the correct device
        returns: (S, D) tensor; if no subs, returns None
        """
        if len(sub_list) == 0:
            return None
        # They may already be contiguous; just stack:
        return torch.stack(sub_list, dim=0)

    def _attend(self, q_vec, sub_list):
        """
        q_vec: (H,) from fc12 (after optional projection to D)
        sub_list: list of (D,) tensors (variable length)
        returns: doc representation d_i = sum_j alpha_j * h_j  with shape (D,)
        """
        H = self.sub_embed_dim
        S = self._stack_subs(sub_list)  # (S_i, D) or None
        if S is None:
            # no sub-sentences → fall back to using the (projected) q_vec itself
            return q_vec

        # scaled dot-product attention: score = (K @ q) / sqrt(D)
        # Here, K=V=S (we use sub embeddings for both key/value)
        scores = (S @ q_vec) / math.sqrt(H)          # (S_i,)
        alpha = torch.softmax(scores, dim=0)         # (S_i,)
        d_i = (alpha.unsqueeze(1) * S).sum(dim=0)    # (D,)
        return d_i

    # ---------- SAME API but now needs sub lists ----------
    def get_theta(self, x, sub_lists=None):
        """
        x: (B, V) BoW
        sub_lists: list length B; each item is list of (D,) sub-embeddings tensors
                   If None, falls back to original SAE behavior.
        """
        mu, logvar = self.encode(x, sub_lists=sub_lists)
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)
        theta = self.theta_drop(theta)
        if self.training:
            return theta, mu, logvar
        else:
            return theta

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    # ---------- CHANGED: encode now does BOW→fc11→fc12, then attention (if sub_lists given) ----------
    def encode(self, x, sub_lists=None):
        """
        x: (B, V)
        sub_lists: None OR list of length B with variable-length lists of (D,) tensors
        Returns mu, logvar via fc21/fc22 + BN (unchanged heads).
        """
        # original bow MLP trunk
        e1 = F.softplus(self.fc11(x))    # (B, en_units)
        e2 = F.softplus(self.fc12(e1))   # (B, en_units)
        e2 = self.fc1_drop(e2)

        # If no sub embeddings provided, behave exactly like original:
        if (self.sub_embed_dim is None) or (sub_lists is None):
            mu_raw = self.fc21(e2)            # (B, K)
            logvar_raw = self.fc22(e2)        # (B, K)
            return self.mean_bn(mu_raw), self.logvar_bn(logvar_raw)

        # Align query dimension to D (if needed)
        # if self.q_proj is not None:
        #     q = self.q_proj(e2)               # (B, D)
        # else:
        #     # en_units == D
        #     q = e2                            # (B, D)
        q = e2

        # Do attention per-document because sub_lists are variable-length
        B = x.shape[0]
        doc_reps = []
        for i in range(B):
            # q[i]: (D,), sub_lists[i]: list[(D,)]
            d_i = self._attend(q[i], sub_lists[i])   # (D,)
            doc_reps.append(d_i)
        D = self.sub_embed_dim
        d = torch.stack(doc_reps, dim=0)            # (B, D)

        # Heads + BN unchanged
        mu_raw = self.fc21(d)                        # (B, K)
        logvar_raw = self.fc22(d)                    # (B, K)
        return self.mean_bn(mu_raw), self.logvar_bn(logvar_raw)

    def decode(self, theta):
        d1 = F.softmax(self.decoder_bn(self.fcd1(theta)), dim=1)
        return d1

    def forward(self, input, avg_loss=True, epoch_id=None):
        bow = input["data"]                                     # (B, V)
        sub_lists = input.get("sub_contextual_embed", None)     # list of lists (len B)

        theta, mu, logvar = self.get_theta(bow, sub_lists=sub_lists)
        recon_x = self.decode(theta)
        loss = self.loss_function(bow, recon_x, mu, logvar)
        return {'loss': loss}

    def loss_function(self, x, recon_x, mu, logvar):
        # same as yours
        recon_loss = -(x * (recon_x + 1e-10).log()).sum(axis=1)
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - self.num_topics)
        loss = (recon_loss + KLD).mean()
        return loss


# import numpy as np
# import torch
# from torch import nn
# import torch.nn.functional as F
# import math

# class SAE(nn.Module):
#     def __init__(self, vocab_size, num_topics=50, en_units=200, dropout=0.4,
#                  sub_embed_dim=384, eps=1e-6):
#         """
#         SAE-NTM (paper-consistent):
#           - fc11: V -> en_units
#           - fc12: en_units -> D (= sub_embed_dim)  [NO projection layer]
#           - attention uses q = fc12(e1) directly in D space
#           - mu/sigma heads from attention doc rep d_i
#           - decode with explicit φ (topics) over vocab, no decoder BN
#         """
#         super().__init__()
#         assert sub_embed_dim is not None, "sub_embed_dim (D) must be specified to match the paper."
#         self.vocab_size    = vocab_size
#         self.num_topics    = num_topics
#         self.sub_embed_dim = sub_embed_dim  # D (e.g., 384)
#         self.eps = eps

#         # ---- BoW encoder trunk ----
#         self.fc11 = nn.Linear(vocab_size, en_units)                 # V -> H
#         self.fc12 = nn.Linear(en_units, sub_embed_dim)              # H -> D  (fixed to D; NO projection)
#         self.fc1_drop  = nn.Dropout(dropout)
#         self.theta_drop = nn.Dropout(dropout)

#         # ---- heads from doc representation d_i (dimension = D) ----
#         head_in = sub_embed_dim
#         self.mu_head    = nn.Linear(head_in, num_topics)            # D -> K
#         self.sigma_head = nn.Linear(head_in, num_topics)            # D -> K (softplus later)

#         # ---- explicit topic–word parameters φ (K x V), β = softmax(φ) over vocab ----
#         self.phi_logits = nn.Parameter(torch.empty(num_topics, vocab_size))
#         nn.init.xavier_uniform_(self.phi_logits)

#     # ---------- utilities ----------
#     def get_beta(self, return_VxK=True):
#         """β = softmax(φ) over vocab dimension. Default returns (V, K) to match paper."""
#         phi = F.softmax(self.phi_logits, dim=1)  # (K, V), rows are topic dists over vocab
#         return phi.T if return_VxK else phi

#     @torch.no_grad()
#     def _stack_subs(self, sub_list):
#         if (sub_list is None) or (len(sub_list) == 0):
#             return None
#         return torch.stack(sub_list, dim=0)  # (S, D)

#     def _attend(self, q_vec, sub_list):
#         """
#         q_vec: (D,)
#         sub_list: list of (D,) tensors
#         returns: d_i ∈ R^D
#         """
#         if (sub_list is None) or (len(sub_list) == 0):
#             return q_vec
#         S = self._stack_subs(sub_list)  # (S_i, D)
#         D = S.size(1)
#         scores = (S @ q_vec) / math.sqrt(D)
#         alpha  = torch.softmax(scores, dim=0)
#         d_i    = (alpha.unsqueeze(1) * S).sum(0)
#         return d_i

#     # ---------- encode: BoW -> H -> D, attention in D, then μ/σ heads ----------
#     def encode(self, x, sub_lists=None):
#         """
#         x: (B, V)
#         sub_lists: list length B; each item is list of (D,) tensors (or None)
#         Returns: mu, sigma with sigma>0 via softplus
#         """
#         e1 = F.softplus(self.fc11(x))             # (B, H)
#         q  = F.softplus(self.fc12(e1))            # (B, D)  -- fixed to D, no projection
#         q  = self.fc1_drop(q)

#         B = x.size(0)
#         doc_reps = []
#         for i in range(B):
#             d_i = self._attend(q[i], None if sub_lists is None else sub_lists[i])  # (D,)
#             doc_reps.append(d_i)
#         d = torch.stack(doc_reps, dim=0)          # (B, D)

#         mu    = self.mu_head(d)                   # (B, K)
#         sigma = F.softplus(self.sigma_head(d)) + self.eps  # (B, K) positive
#         return mu, sigma

#     # ---------- reparameterize to simplex ----------
#     def reparameterize_to_theta(self, mu, sigma):
#         if self.training:
#             eps = torch.randn_like(sigma)
#             y   = mu + sigma * eps
#         else:
#             y   = mu
#         return torch.softmax(y, dim=1)            # (B, K)

#     # ---------- decode with φ ----------
#     def decode(self, theta):
#         phi = F.softmax(self.phi_logits, dim=1)   # (K, V)
#         logits = theta @ phi.T                    # (B, V)
#         return torch.softmax(logits, dim=1)

#     # ---------- public API ----------
#     def get_theta(self, x, sub_lists=None):
#         mu, sigma = self.encode(x, sub_lists=sub_lists)
#         theta = self.reparameterize_to_theta(mu, sigma)
#         theta = self.theta_drop(theta)
#         if self.training:
#             return theta, mu, sigma
#         else:
#             return theta

#     def forward(self, input, avg_loss=True, epoch_id=None):
#         bow = input["data"]                                 # (B, V)
#         sub_lists = input.get("sub_contextual_embed", None) # list of lists
#         theta, mu, sigma = self.get_theta(bow, sub_lists=sub_lists)
#         recon_x = self.decode(theta)
#         loss = self.loss_function(bow, recon_x, mu, sigma)
#         return {'loss': loss}

#     # ---------- losses ----------
#     def loss_function(self, x, recon_x, mu, sigma):
#         # reconstruction
#         recon_loss = -(x * (recon_x.clamp_min(self.eps)).log()).sum(dim=1)
#         # KL to N(0,I) in logistic-normal space
#         sigma2 = sigma * sigma
#         kld = 0.5 * (mu.pow(2) + sigma2 - sigma2.clamp_min(self.eps).log() - 1.0).sum(dim=1)
#         return (recon_loss + kld).mean()
