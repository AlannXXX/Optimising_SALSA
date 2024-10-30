
def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    """
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))




def get_masks(slen, lengths, causal):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]

    # attention mask is the same as mask, or triangular inferior attention (causal)
    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask

    # sanity check
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    return mask, attn_mask


class MultiHeadAttention(nn.Module):

    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, src_dim, dropout, normalized_attention, xav_init=False):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.src_dim = src_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.normalized_attention = normalized_attention
        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(src_dim, dim)
        self.v_lin = nn.Linear(src_dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        if self.normalized_attention:
            self.attention_scale = nn.Parameter(
                torch.tensor(1.0 / math.sqrt(dim // n_heads))
            )
        if xav_init:
            gain = (1 / math.sqrt(2)) if self.src_dim == self.dim else 1.0
            nn.init.xavier_uniform_(self.q_lin.weight, gain=gain)
            nn.init.xavier_uniform_(self.k_lin.weight, gain=gain)
            nn.init.xavier_uniform_(self.v_lin.weight, gain=gain)
            nn.init.xavier_uniform_(self.out_lin.weight)
            nn.init.constant_(self.out_lin.bias, 0.0)

    def forward(self, input, mask, kv=None, use_cache=False, first_loop=True):
        """
        Self-attention (if kv is None)
        or attention over source sentence (provided by kv).
        Input is (bs, qlen, dim)
        Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        """
        assert not (use_cache and self.cache is None)
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if not use_cache else self.cache["slen"] + qlen
        else:
            klen = kv.size(1)
        assert dim == self.dim, "Dimensions do not match: %s input vs %s configured" % (
            dim,
            self.dim,
        )
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return (
                x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
            )

        q = shape(self.q_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = shape(self.k_lin(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        elif not use_cache or self.layer_id not in self.cache:
            k = v = kv
            k = shape(self.k_lin(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(v))  # (bs, n_heads, qlen, dim_per_head)

        if use_cache:
            if self.layer_id in self.cache:
                if kv is None and first_loop:
                    k_, v_ = self.cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
                    v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
                else:
                    k, v = self.cache[self.layer_id]
            self.cache[self.layer_id] = (k, v)
        if self.normalized_attention:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
            q = q * self.attention_scale
        else:
            q = q / math.sqrt(dim_per_head)  # (bs, n_heads, qlen, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, qlen, klen)
        mask = (
            (mask == 0).view(mask_reshape).expand_as(scores)
        )  # (bs, n_heads, qlen, klen)
        scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, qlen, klen)

        weights = F.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (bs, n_heads, qlen, klen)
        weights = F.dropout(
            weights, p=self.dropout, training=self.training
        )  # (bs, n_heads, qlen, klen)
        context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        if TransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs = weights.detach().cpu()

        return self.out_lin(context)


class SwiGLU(nn.Module):
    def __init__(self, in_features):
        super(SwiGLU, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.fc2 = nn.Linear(in_features, in_features)

    def forward(self, x):
        return self.fc1(x) * torch.sigmoid(self.fc2(x))


class TransformerFFN(nn.Module):
    def __init__(self, in_dim, dim_hidden, out_dim, hidden_layers, dropout, num_splits, gelu_activation=False,
                 xav_init=False):
        super().__init__()
        self.dropout = dropout
        self.num_splits = num_splits
        self.dim_per_split = dim_hidden // num_splits
        self.hidden_layers = hidden_layers
        self.act = gelu if gelu_activation else F.relu

        # Create separate linear layers for each split for input and intermediate layers
        self.lin_splits = nn.ModuleList([
            nn.ModuleList(
                [nn.Linear(in_dim // num_splits, self.dim_per_split if i == 0 else self.dim_per_split * 2) for i in
                 range(hidden_layers)])
            for _ in range(num_splits)
        ])

        # Output layer that combines transformed splits
        self.lin_out = nn.Linear(dim_hidden * 2, out_dim)

        # Xavier initialization if specified
        if xav_init:
            self.initialize_weights()

    def initialize_weights(self):
        for split_group in self.lin_splits:
            for lin in split_group:
                nn.init.xavier_uniform_(lin.weight)
                nn.init.constant_(lin.bias, 0.0)
        nn.init.xavier_uniform_(self.lin_out.weight)
        nn.init.constant_(self.lin_out.bias, 0.0)

    def forward(self, x):
        # Split input tensor into multiple segments
        split_inputs = torch.split(x, x.shape[-1] // self.num_splits, dim=-1)

        # Process each split independently
        outputs = []
        for group in self.lin_splits:
            split_output = None
            for i, lin in enumerate(group):
                if i == 0:
                    split_output = self.act(lin(split_inputs[i]))
                else:
                    # Apply non-linear transformation and an element-wise multiplication for interaction
                    split_output = self.act(lin(split_output)) * split_output
                split_output = F.dropout(split_output, p=self.dropout, training=self.training)
            outputs.append(split_output)

        # Concatenate all outputs from splits
        combined_output = torch.cat(outputs, dim=-1)

        # Final linear transformation
        final_output = self.lin_out(combined_output)
        return final_output

