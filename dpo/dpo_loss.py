import torch
import torch.nn.functional as F


def dpo_loss(beta_logits, beta_ref_logits, preferences, alpha=1.0, device=None):
    """
    beta_logits: (K, V) current logits
    beta_ref_logits: (K, V) reference logits
    preferences: dict {k: {"w_plus": [idx], "w_minus": [idx]}}
    """
    if device is None:
        device = beta_logits.device

    total_loss = 0.0
    topic_count = 0

    for k, pref in preferences.items():
        w_plus = pref.get("w_plus", [])
        w_minus = pref.get("w_minus", [])
        if not w_plus or not w_minus:
            continue

        w_plus_t = torch.tensor(w_plus, device=device, dtype=torch.long)
        w_minus_t = torch.tensor(w_minus, device=device, dtype=torch.long)

        cur_plus = beta_logits[k, w_plus_t]  # (P,)
        cur_minus = beta_logits[k, w_minus_t]  # (M,)
        ref_plus = beta_ref_logits[k, w_plus_t]
        ref_minus = beta_ref_logits[k, w_minus_t]

        # Pairwise differences: (P, M)
        cur_diff = cur_plus[:, None] - cur_minus[None, :]
        ref_diff = ref_plus[:, None] - ref_minus[None, :]

        logits = alpha * (cur_diff - ref_diff)
        loss_k = -F.logsigmoid(logits).mean()

        total_loss += loss_k
        topic_count += 1

    if topic_count == 0:
        return torch.tensor(0.0, device=device)

    return total_loss / topic_count
