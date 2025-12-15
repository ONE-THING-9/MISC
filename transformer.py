# Pure PyTorch FUNCTIONS (no classes) for:
# 1) self-attention
# 2) multi-head self-attention
# 3) feed-forward network
# 4) layer norm
# 5) residual wrapper
# 6) positional encoding + token/position embeddings
# 7) masked multi-head attention (causal + padding)

import math
import torch
import torch.nn.functional as F


# -------------------------
# 4) LayerNorm (token-wise, over last dim)
# -------------------------
def layer_norm(x, gamma, beta, eps=1e-5):
    """
    x:     (B, T, D)
    gamma: (D,)
    beta:  (D,)
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    xhat = (x - mean) / torch.sqrt(var + eps)
    return xhat * gamma + beta


# -------------------------
# 5) Residual wrapper
# -------------------------
def residual_add(x, sublayer_out, dropout_p=0.0, training=True):
    """
    x:           (B, T, D)
    sublayer_out:(B, T, D)
    """
    if dropout_p > 0.0:
        sublayer_out = F.dropout(sublayer_out, p=dropout_p, training=training)
    return x + sublayer_out


# -------------------------
# 1) Self-Attention (single-head)
# -------------------------
def self_attention(x, Wq, Wk, Wv, Wo=None, attn_mask=None, dropout_p=0.0, training=True):
    """
    x:   (B, T, D)
    Wq:  (D, d_k)
    Wk:  (D, d_k)
    Wv:  (D, d_k)
    Wo:  optional (d_k, D) to project back to D
    attn_mask: broadcastable to (B, T, T)
      - bool: True = masked (blocked)
      - float: additive mask (use -inf for masked)
    returns:
      y:    (B, T, D) if Wo given else (B, T, d_k)
      attn: (B, T, T)
    """
    B, T, _ = x.shape
    q = x @ Wq                    # (B, T, d_k)
    k = x @ Wk                    # (B, T, d_k)
    v = x @ Wv                    # (B, T, d_k)
    d_k = q.size(-1)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)   # (B, T, T)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask

    attn = F.softmax(scores, dim=-1)                      # (B, T, T)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p, training=training)

    y = attn @ v                                          # (B, T, d_k)
    if Wo is not None:
        y = y @ Wo                                        # (B, T, D)
    return y, attn


# -------------------------
# 2) Multi-Head Self-Attention (unmasked)
# -------------------------
def multi_head_self_attention(x, Wq, Wk, Wv, Wo, num_heads,
                              attn_mask=None, dropout_p=0.0, training=True):
    """
    x:   (B, T, D)
    Wq,Wk,Wv: (D, D)  (project to full D, then split heads)
    Wo:       (D, D)
    num_heads: H (D must be divisible by H)
    attn_mask: broadcastable to (B, 1, T, T) or (B, H, T, T) or (B, T, T)
    returns:
      out:  (B, T, D)
      attn: (B, H, T, T)
    """
    B, T, D = x.shape
    H = num_heads
    assert D % H == 0, "D must be divisible by num_heads"
    d_k = D // H

    # Linear projections
    q = x @ Wq   # (B, T, D)
    k = x @ Wk   # (B, T, D)
    v = x @ Wv   # (B, T, D)

    # Split heads: (B, T, D) -> (B, H, T, d_k)
    q = q.view(B, T, H, d_k).transpose(1, 2)
    k = k.view(B, T, H, d_k).transpose(1, 2)
    v = v.view(B, T, H, d_k).transpose(1, 2)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)   # (B, H, T, T)

    if attn_mask is not None:
        # Make mask broadcast-friendly
        if attn_mask.dim() == 3:          # (B, T, T)
            attn_mask = attn_mask.unsqueeze(1)  # (B, 1, T, T)
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask

    attn = F.softmax(scores, dim=-1)                      # (B, H, T, T)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p, training=training)

    out = attn @ v                                        # (B, H, T, d_k)

    # Merge heads: (B, H, T, d_k) -> (B, T, D)
    out = out.transpose(1, 2).contiguous().view(B, T, D)
    out = out @ Wo                                        # (B, T, D)
    return out, attn


# -------------------------
# 3) Feed-Forward Network (FFN)
# -------------------------
def feed_forward(x, W1, b1, W2, b2, activation="gelu", dropout_p=0.0, training=True):
    """
    x:  (B, T, D)
    W1: (D, d_ff)
    b1: (d_ff,)
    W2: (d_ff, D)
    b2: (D,)
    """
    h = x @ W1 + b1                    # (B, T, d_ff)
    if activation == "relu":
        h = F.relu(h)
    elif activation == "gelu":
        h = F.gelu(h)
    elif activation == "silu":
        h = F.silu(h)
    else:
        raise ValueError("activation must be one of: relu, gelu, silu")

    if dropout_p > 0.0:
        h = F.dropout(h, p=dropout_p, training=training)

    y = h @ W2 + b2                    # (B, T, D)
    return y


# -------------------------
# 6a) Sinusoidal positional encoding (absolute)
# -------------------------
def sinusoidal_positional_encoding(T, D, device=None, dtype=torch.float32):
    """
    returns PE: (1, T, D) for easy broadcasting over batch
    """
    device = device or "cpu"
    pe = torch.zeros(T, D, device=device, dtype=dtype)
    position = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)  # (T, 1)
    div_term = torch.exp(torch.arange(0, D, 2, device=device, dtype=dtype) *
                         (-math.log(10000.0) / D))                        # (D/2,)

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, T, D)


# -------------------------
# 6b) Token embeddings + (learned or sinusoidal) position embeddings
# -------------------------
def token_embeddings(input_ids, token_embed_table):
    """
    input_ids: (B, T) LongTensor
    token_embed_table: (vocab_size, D)
    returns: (B, T, D)
    """
    return F.embedding(input_ids, token_embed_table)

def learned_positional_embeddings(T, pos_embed_table, device=None):
    """
    pos_embed_table: (max_len, D)
    returns: (1, T, D)
    """
    device = device or pos_embed_table.device
    pos = torch.arange(T, device=device, dtype=torch.long)
    return F.embedding(pos, pos_embed_table).unsqueeze(0)  # (1, T, D)

def add_position(x_tok, pos_emb):
    """
    x_tok:  (B, T, D)
    pos_emb:(1, T, D) or (B, T, D)
    """
    return x_tok + pos_emb


# -------------------------
# 7) Mask helpers + Masked Multi-Head Attention
# -------------------------
def causal_mask(T, device):
    """
    returns bool mask: (1, 1, T, T)  True = masked (block future)
    """
    m = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
    return m.unsqueeze(0).unsqueeze(0)

def padding_mask(input_ids, pad_id=0):
    """
    input_ids: (B, T)
    returns bool mask: (B, 1, 1, T) True = masked (pad positions)
    """
    return (input_ids == pad_id).unsqueeze(1).unsqueeze(2)

def combine_masks(causal, pad):
    """
    causal: (1, 1, T, T)
    pad:    (B, 1, 1, T)
    returns: (B, 1, T, T) bool
    """
    # pad needs to expand across query positions
    return causal | pad  # broadcasting does the right thing

def masked_multi_head_attention(x, Wq, Wk, Wv, Wo, num_heads,
                                input_ids=None, pad_id=0, use_causal=True,
                                extra_attn_mask=None, dropout_p=0.0, training=True):
    """
    Masked multi-head self-attention.
    x: (B, T, D)
    input_ids: optional (B, T) for padding mask
    use_causal: apply causal mask (decoder / GPT)
    extra_attn_mask: optional additional mask broadcastable to (B, 1, T, T) or (B, H, T, T)
                     (bool True=mask OR float additive)
    returns: out (B, T, D), attn (B, H, T, T)
    """
    B, T, D = x.shape
    device = x.device

    final_mask = None
    if use_causal:
        final_mask = causal_mask(T, device)  # (1,1,T,T)

    if input_ids is not None:
        pm = padding_mask(input_ids, pad_id=pad_id)  # (B,1,1,T)
        if final_mask is None:
            # expand to (B,1,T,T) by broadcasting later in attention fn
            final_mask = pm
        else:
            final_mask = combine_masks(final_mask, pm)  # (B,1,T,T)

    # If user provides extra mask, merge it
    if extra_attn_mask is not None:
        if final_mask is None:
            final_mask = extra_attn_mask
        else:
            # If extra is float additive, convert bool mask to additive (preferred),
            # but simplest: keep bool OR if extra is bool; else add after converting bool->additive
            if extra_attn_mask.dtype == torch.bool and final_mask.dtype == torch.bool:
                final_mask = final_mask | extra_attn_mask
            else:
                # convert bool mask to additive
                additive = (final_mask.to(torch.float32) * float("-inf")) if final_mask.dtype == torch.bool else final_mask
                final_mask = additive + extra_attn_mask

    return multi_head_self_attention(
        x, Wq, Wk, Wv, Wo, num_heads,
        attn_mask=final_mask,
        dropout_p=dropout_p,
        training=training
    )


# -------------------------
# (Optional) Tiny usage sketch (parameters are just tensors)
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D = 2, 5, 64
    H = 8
    d_ff = 256
    vocab = 1000
    pad_id = 0

    # Example "parameters" (you'd usually wrap these in nn.Parameter in real training)
    token_table = torch.randn(vocab, D)
    pos_table = torch.randn(512, D)

    Wq = torch.randn(D, D)
    Wk = torch.randn(D, D)
    Wv = torch.randn(D, D)
    Wo = torch.randn(D, D)

    W1 = torch.randn(D, d_ff)
    b1 = torch.randn(d_ff)
    W2 = torch.randn(d_ff, D)
    b2 = torch.randn(D)

    gamma = torch.ones(D)
    beta = torch.zeros(D)

    input_ids = torch.randint(1, vocab, (B, T))
    input_ids[0, -1] = pad_id  # add a pad example

    x_tok = token_embeddings(input_ids, token_table)
    pe = learned_positional_embeddings(T, pos_table)
    x = add_position(x_tok, pe)

    # Pre-LN attention block example (functional)
    x_norm = layer_norm(x, gamma, beta)
    attn_out, _ = masked_multi_head_attention(x_norm, Wq, Wk, Wv, Wo, H, input_ids=input_ids, pad_id=pad_id, use_causal=True)
    x = residual_add(x, attn_out, dropout_p=0.1, training=True)

    # Pre-LN FFN
    x_norm = layer_norm(x, gamma, beta)
    ffn_out = feed_forward(x_norm, W1, b1, W2, b2, activation="gelu", dropout_p=0.1, training=True)
    x = residual_add(x, ffn_out, dropout_p=0.1, training=True)

    print("final x:", x.shape)  # (B, T, D)
