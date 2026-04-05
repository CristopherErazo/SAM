import torch

def generate_dual_task_batch_optimized(num_samples: int,
                                       L: int,
                                       K: int,
                                       distributions: dict[str, torch.Tensor],
                                       trigger_set: torch.Tensor | None = None,
                                       device: str | torch.device = "cuda") -> dict[str, torch.Tensor]:
    """
    Optimized batch generator for the dual task, designed for GPU execution.

    Returns:
        dict with:
            sequence     : (B, L)
            trigger_set  : (B, K)
            output_set   : (B, K)
            counts       : (B, L)
            is_trigg     : (B, L)
    """
    B = num_samples

    # Move distributions to the device
    P_b = distributions['P_b'].to(device)
    P_u = distributions['P_u'].to(device)
    P_o = distributions['P_o'].to(device)
    P_t = distributions['P_t'].to(device)
    V = P_u.shape[0]

    # ---- sample trigger sets ----
    if trigger_set is None:
        trigger_sets = torch.multinomial(P_t, K, replacement=False).unsqueeze(0).repeat(B, 1)
    else:
        trigger_set = trigger_set.to(device)
        assert trigger_set.numel() == K
        trigger_sets = trigger_set.unsqueeze(0).repeat(B, 1)

    # ---- sample output tokens ----
    output_sets = torch.zeros(B, K, dtype=torch.long, device=device)
    for k in range(K):
        output_sets[:, k] = torch.multinomial(P_o[trigger_sets[:, k]], 1).squeeze(-1)

    # ---- build trigger mask ----
    trigger_mask = torch.zeros(B, V, dtype=torch.bool, device=device)
    trigger_mask.scatter_(1, trigger_sets, True)

    # ---- build mapping trigger -> output ----
    mapping = torch.full((B, V), -1, dtype=torch.long, device=device)
    mapping.scatter_(1, trigger_sets, output_sets)

    # ---- initialize sequence ----
    sequence = torch.zeros(B, L + 1, dtype=torch.long, device=device)
    sequence[:, 0] = torch.multinomial(P_u, B, replacement=True)

    # ---- outputs ----
    is_trigg = torch.zeros(B, L + 1, dtype=torch.long, device=device)
    counts = torch.zeros(B, L + 1, dtype=torch.long, device=device)

    # for counting occurrences
    token_counts = torch.zeros(B, V, dtype=torch.long, device=device)

    # ---- main loop over sequence length ----
    for t in range(L):
        current = sequence[:, t]

        # update counts
        token_counts.scatter_add_(
            1,
            current.unsqueeze(1),
            torch.ones(B, 1, dtype=torch.long, device=device),
        )
        counts[:, t] = token_counts[
            torch.arange(B, device=device), current
        ]

        # mark trigger
        is_trigg[:, t] = trigger_mask[
            torch.arange(B, device=device), current
        ].long()

        # ---- next token ----
        prev = current

        # sample bigram transitions
        next_tokens = torch.multinomial(P_b[prev], 1).squeeze(-1)

        # override if trigger
        indices = torch.nonzero(trigger_mask[torch.arange(B, device=device), prev], as_tuple=True)[0]
        next_tokens.index_put_((indices,), mapping[torch.arange(B, device=device), prev][indices])

        sequence[:, t + 1] = next_tokens

    mask = torch.tril(torch.ones((L, L), dtype=torch.bool, device=device))
    batch_mask = mask.unsqueeze(0).expand(B, -1, -1)

    return {
        "sequence": sequence[:, :L],
        "trigger_set": trigger_sets,
        "output_set": output_sets,
        "counts": counts[:, :L],
        "is_trigg": is_trigg[:, :L],
        "mask": batch_mask,
    }