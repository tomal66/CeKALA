import torch
from dassl.data import DataManager
from trainers.mmadapter import load_clip_to_cpu
from clip import clip


def linear_cka(X, Y, eps=1e-12):
    """
    Compute linear Centered Kernel Alignment (CKA) between two
    representation matrices X and Y of shape (N, D).

    Returns a Python float.
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(f"CKA expects 2D tensors, got shapes {tuple(X.shape)} and {tuple(Y.shape)}")

    if X.size(0) != Y.size(0):
        raise ValueError(
            f"CKA expects the same number of samples, got {X.size(0)} and {Y.size(0)}"
        )

    X = X.float()
    Y = Y.float()

    X_c = X - X.mean(dim=0, keepdim=True)
    Y_c = Y - Y.mean(dim=0, keepdim=True)

    hsic = torch.norm(X_c.t() @ Y_c, p="fro") ** 2
    norm_x = torch.norm(X_c.t() @ X_c, p="fro")
    norm_y = torch.norm(Y_c.t() @ Y_c, p="fro")

    denom = norm_x * norm_y
    if not torch.isfinite(denom) or denom.item() <= eps:
        return 0.0

    value = hsic / denom
    if not torch.isfinite(value):
        return 0.0

    return value.item()


def select_top_k_layers(features, k):
    """
    Compute layer scores based on the average of (1 - CKA) to all other layers.

    This matches:
        Score_i = (1 / (L - 1)) * sum_{j != i} (1 - CKA(H_i, H_j))

    Returns top-k layer indices ranked by descending score.
    """
    L = len(features)

    if L == 0:
        return []

    if L == 1:
        return [0]

    k = min(k, L)
    scores = []

    for i in range(L):
        s_i = 0.0
        for j in range(L):
            if i != j:
                s_i += (1.0 - linear_cka(features[i], features[j]))
        scores.append(s_i / (L - 1))

    scores = torch.tensor(scores, dtype=torch.float32)
    top_k_indices = torch.topk(scores, k=k, largest=True).indices.tolist()
    return top_k_indices


def ordered_intersection(a, b):
    """
    Return intersection(a, b) while preserving the order from 'a'.
    """
    b_set = set(b)
    return [x for x in a if x in b_set]


def get_intermediate_features(model, dataloader, classnames, cfg, device="cuda"):
    """
    Collect intermediate layer representations for:
      - image encoder: over the full dataset D (all batches in dataloader)
      - text encoder: over the full class-prompt set used by the classifier

    Notes:
      * Image features use the CLS token from each visual transformer block.
      * Text features use mean pooling over tokens from each text transformer block.
      * Features are moved to CPU immediately to reduce GPU memory pressure.
    """
    model.eval()

    img_features = []
    txt_features = []

    img_hooks = []
    txt_hooks = []

    def get_img_hook(layer_idx):
        def hook(module, input, output):
            while len(img_features) <= layer_idx:
                img_features.append([])

            # CLIP transformer block output is typically (seq_len, batch, dim)
            out = output.permute(1, 0, 2)   # -> (batch, seq_len, dim)
            cls_token = out[:, 0, :]        # CLS token
            img_features[layer_idx].append(cls_token.detach().cpu())

        return hook

    def get_txt_hook(layer_idx):
        def hook(module, input, output):
            while len(txt_features) <= layer_idx:
                txt_features.append([])

            # CLIP transformer block output is typically (seq_len, batch, dim)
            out = output.permute(1, 0, 2)   # -> (batch, seq_len, dim)
            mean_token = out.mean(dim=1)    # mean-pooled token representation
            txt_features[layer_idx].append(mean_token.detach().cpu())

        return hook

    # Register hooks on all visual transformer blocks
    for i, resblock in enumerate(model.visual.transformer.resblocks):
        img_hooks.append(resblock.register_forward_hook(get_img_hook(i)))

    # Register hooks on all text transformer blocks
    for i, resblock in enumerate(model.transformer.resblocks):
        txt_hooks.append(resblock.register_forward_hook(get_txt_hook(i)))

    # Build the full class-prompt set for the text encoder
    classnames = [name.replace("_", " ") for name in classnames]
    prompts = [cfg.TRAINER.MMADAPTER.TEXT_CTX_INIT + " " + name + "." for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

    with torch.no_grad():
        # Use the full image dataset D
        for batch in dataloader:
            image = batch["img"].to(device)
            model.encode_image(image)

        # Use the full class-prompt set once for text
        model.encode_text(tokenized_prompts)

    # Remove hooks
    for h in img_hooks:
        h.remove()
    for h in txt_hooks:
        h.remove()

    # Concatenate all captured features layer-wise
    final_img_features = [torch.cat(feats, dim=0) for feats in img_features]
    final_txt_features = [torch.cat(feats, dim=0) for feats in txt_features]

    return final_img_features, final_txt_features


def select_layers(dataset, shots, cfg, k=4):
    """
    CeKALA algorithm:
      1) Select top-k image layers L_img
      2) Select top-k text layers L_txt
      3) Compute shared multimodal layers L_mm = L_img ∩ L_txt
      4) Remove shared layers from L_img and L_txt
      5) Return L_mm, L_img, L_txt

    Method signature is kept intact.
    """
    print(f"Running CeKALA layer selection for {dataset} with {shots} shots and K={k}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = load_clip_to_cpu(cfg).to(device)

    dm = DataManager(cfg)
    train_loader = dm.train_loader_x
    classnames = dm.dataset.classnames

    img_features, txt_features = get_intermediate_features(
        clip_model, train_loader, classnames, cfg, device=device
    )

    # Step 1 and 2 in the PDF
    L_img_topk = select_top_k_layers(img_features, k)
    L_txt_topk = select_top_k_layers(txt_features, k)

    # Step 3: intersection, preserving ranking order from image side
    L_mm = ordered_intersection(L_img_topk, L_txt_topk)

    # Step 4: remove shared layers from modality-specific selections
    L_mm_set = set(L_mm)
    L_img = [layer for layer in L_img_topk if layer not in L_mm_set]
    L_txt = [layer for layer in L_txt_topk if layer not in L_mm_set]

    print(f"Initial image top-k layers (before removing overlap): {L_img_topk}")
    print(f"Initial text top-k layers  (before removing overlap): {L_txt_topk}")
    print(f"Shared multimodal layers (L_mm): {L_mm}")
    print(f"Image-only layers after removal (L_img): {L_img}")
    print(f"Text-only layers after removal  (L_txt): {L_txt}")

    del clip_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return L_mm, L_img, L_txt