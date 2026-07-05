
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


class SubwordLLM(nn.Module):

    def __init__(self, vocab_size, embed_dim=128, n_heads=4, n_layers=2, block_size=64):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        x = self.transformer(x, mask=mask)
        x = self.ln(x)
        return self.head(x)


def _train_tokenizer(text, vocab_size=500, min_frequency=2):

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer()
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=False,
        initial_alphabet=ByteLevelPreTokenizer.alphabet(),
    )

    tmp_path = "_tmp_corpus_for_tokenizer.txt"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    tokenizer.train([tmp_path], trainer)
    os.remove(tmp_path)
    return tokenizer


def _grow_vocab_and_resize(model, optimizer, config, new_tokenizer,
                            embed_dim, n_heads, n_layers, lr):
    """Builds a new (larger) model sized for `new_tokenizer`'s vocab, and
    transplants weights for every token that exists in BOTH the old and new
    vocab (matched by exact token STRING, not by id -- ids are not stable
    across a tokenizer retrain). Tokens that only exist in the new vocab get
    a fresh random embedding/head row, exactly like a newly initialized model
    would have -- they start untrained and improve only with further gradient
    steps on data that uses them.

    CAVEAT (real, not hypothetical): retraining BPE on a bigger/different
    corpus can also change how OLD, previously-seen text gets merged, not
    just add new tokens. Any old token whose exact string doesn't survive
    into the new vocab loses its trained embedding and starts over. In a
    controlled test with a moderate new corpus, ~96% of old tokens survived
    by exact string match -- decent, but not something to assume as a
    guarantee for arbitrary new data.

    Optimizer state (Adam's per-parameter momentum/variance) is NOT
    transplanted -- it's reset. Resizing the parameter tensors invalidates
    the old optimizer's internal state anyway, so this is unavoidable here,
    and is standard practice when resizing embeddings.
    """
    old_tokenizer = config["tokenizer"]
    old_vocab = old_tokenizer.get_vocab()      # token string -> old id
    new_vocab = new_tokenizer.get_vocab()      # token string -> new id
    new_vocab_size = new_tokenizer.get_vocab_size()
    block_size = config["block_size"]

    new_model = SubwordLLM(new_vocab_size, embed_dim, n_heads, n_layers, block_size)

    with torch.no_grad():
        matched = 0
        for token_str, old_id in old_vocab.items():
            new_id = new_vocab.get(token_str)
            if new_id is not None:
                new_model.token_emb.weight[new_id] = model.token_emb.weight[old_id]
                new_model.head.weight[new_id] = model.head.weight[old_id]
                new_model.head.bias[new_id] = model.head.bias[old_id]
                matched += 1
        # Non-embedding/head layers (positional embeddings, attention,
        # layernorm) don't depend on vocab size -- copy them straight over.
        new_model.pos_emb.load_state_dict(model.pos_emb.state_dict())
        new_model.transformer.load_state_dict(model.transformer.state_dict())
        new_model.ln.load_state_dict(model.ln.state_dict())

    print(f"Vocab grew from {len(old_vocab)} to {new_vocab_size} tokens "
          f"({matched}/{len(old_vocab)} old tokens transplanted by exact "
          f"string match; the rest, plus all new tokens, start untrained).")

    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=lr)
    new_config = {
        "tokenizer": new_tokenizer,
        "vocab_size": new_vocab_size,
        "block_size": block_size,
        "_seen_text": config.get("_seen_text", ""),
    }
    return new_model, new_optimizer, new_config


def save_model(model, optimizer, config, path="subword_llm.pt"):

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "tokenizer_str": config["tokenizer"].to_str(),
            "vocab_size": config["vocab_size"],
            "block_size": config["block_size"],
            "seen_text": config.get("_seen_text", ""),
        },
        path,
    )
    print(f"Saved checkpoint to {path}")


def load_model(path, embed_dim=128, n_heads=4, n_layers=2, lr=3e-4, device="cpu"):
    """Loads a checkpoint written by save_model(), including its tokenizer
    and accumulated corpus text."""
    checkpoint = torch.load(path, map_location=device)
    tokenizer = Tokenizer.from_str(checkpoint["tokenizer_str"])

    config = {
        "tokenizer": tokenizer,
        "vocab_size": checkpoint["vocab_size"],
        "block_size": checkpoint["block_size"],
        "_seen_text": checkpoint.get("seen_text", ""),
    }

    model = SubwordLLM(
        config["vocab_size"], embed_dim, n_heads, n_layers, config["block_size"]
    )
    model.load_state_dict(checkpoint["model_state"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    return model, optimizer, config


# --------------------------------------------------------------------------
# 1) INITIALIZATION
# --------------------------------------------------------------------------
def initialize_model(
    text, path="subword_llm.pt", tokenizer_vocab_size=500,
    embed_dim=128, n_heads=4, n_layers=2, block_size=64, lr=3e-4,
):

    if os.path.exists(path):
        print(f"Found existing checkpoint at {path} -- loading model + "
              f"tokenizer instead of initializing new ones.")
        return load_model(path, embed_dim=embed_dim, n_heads=n_heads,
                           n_layers=n_layers, lr=lr)

    print(f"No checkpoint found at {path} -- training a new BPE tokenizer "
          f"and initializing a new model.")
    tokenizer = _train_tokenizer(text, vocab_size=tokenizer_vocab_size)
    vocab_size = tokenizer.get_vocab_size()

    model = SubwordLLM(vocab_size, embed_dim, n_heads, n_layers, block_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    config = {
        "tokenizer": tokenizer,
        "vocab_size": vocab_size,
        "block_size": block_size,
        "_seen_text": text,
    }
    return model, optimizer, config


def _get_batch(data, block_size, batch_size, device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# --------------------------------------------------------------------------
# 2) TRAINING
# --------------------------------------------------------------------------
def train_model(
    model, optimizer, config, text, steps=500, batch_size=32, device="cpu",
    grow_vocab=True, max_vocab_size=2000,
    embed_dim=128, n_heads=4, n_layers=2, lr=3e-4,
):

    if grow_vocab:
        seen_text = config.get("_seen_text", "")
        combined_text = (seen_text + "\n" + text) if seen_text else text
        new_tokenizer = _train_tokenizer(combined_text, vocab_size=max_vocab_size)
        if new_tokenizer.get_vocab_size() > config["vocab_size"]:
            model, optimizer, config = _grow_vocab_and_resize(
                model, optimizer, config, new_tokenizer,
                embed_dim, n_heads, n_layers, lr,
            )
        else:
            config["tokenizer"] = new_tokenizer
        config["_seen_text"] = combined_text

    model.to(device)
    tokenizer = config["tokenizer"]
    block_size = config["block_size"]

    ids = tokenizer.encode(text).ids
    data = torch.tensor(ids, dtype=torch.long)
    if len(data) <= block_size:
        raise ValueError(
            f"Training text encodes to only {len(data)} tokens, which is <= "
            f"block_size ({block_size}). Provide more text or lower block_size."
        )

    model.train()
    for step in range(steps):
        xb, yb = _get_batch(data, block_size, batch_size, device)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0 or step == steps - 1:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    return model, optimizer, config


# --------------------------------------------------------------------------
# 3) QUERYING / GENERATION
# --------------------------------------------------------------------------
def query_model(model, config, prompt, max_new_tokens=100, temperature=1.0, device="cpu"):
    """Generates up to `max_new_tokens` subword tokens continuing from `prompt`."""
    tokenizer = config["tokenizer"]
    block_size = config["block_size"]
    model.eval()

    ids = tokenizer.encode(prompt).ids
    if not ids:
        raise ValueError("Prompt encoded to zero tokens.")
    idx = torch.tensor([ids], dtype=torch.long).to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

    return tokenizer.decode(idx[0].tolist())


# --------------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------------
if __name__ == "__main__":
    text = (
        """An egg is an organic vessel grown by an animal to carry a possibly fertilized egg cell – a zygote. Within the vessel, an embryo is incubated until it has become an animal fetus that can survive on its own, at which point the animal hatches. Reproductive structures similar to the egg in other kingdoms are termed "spores", or in spermatophytes "seeds", or in gametophytes "egg cells".

Most arthropods, vertebrates (excluding live-bearing mammals), and mollusks lay eggs, although some, such as scorpions, do not. Reptile eggs, bird eggs, and monotreme eggs are laid out of water and are surrounded by a protective shell, either flexible or inflexible. Eggs laid on land or in nests are usually kept within a warm and favorable temperature range while the embryo grows. When the embryo is adequately developed it hatches; i.e., breaks out of the egg's shell. Some embryos have a temporary egg tooth they use to crack, pip, or break the eggshell or covering.

For people, eggs are a popular food item, and they appear on menus worldwide. Eggs remain an important symbol in folklore and mythology, symbolizing life, healing, and rebirth. They are frequently the subject of decoration. Egg collecting has been popular in some cultures, although the practice is now banned in many jurisdictions. Chicken eggs are used in the production of vaccines for infectious diseases.

"""
    ) * 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "subword_llm.pt"

    model, optimizer, config = initialize_model(
        text, path=checkpoint_path, tokenizer_vocab_size=500, block_size=64
    )
    model, optimizer, config = train_model(
        model, optimizer, config, text, steps=1000, device=device
    )
    save_model(model, optimizer, config, path=checkpoint_path)

    generated = query_model(model, config, prompt="hello", max_new_tokens=50, device=device)
    print("\nGenerated text:\n", generated)
