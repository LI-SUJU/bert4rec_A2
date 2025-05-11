import pandas as pd
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

MAX_LEN = 20
MASK_PROB = 0.2
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_and_preprocess_movielens(path="dataset/ratings.dat", min_interactions=5, max_len=MAX_LEN):
    df = pd.read_csv(path, sep="::", names=["userId", "movieId", "rating", "timestamp"], engine="python")
    df = df[df["rating"] >= 4]
    df = df.sort_values(by=["userId", "timestamp"])
    user_seqs = df.groupby("userId")["movieId"].apply(list)
    user_seqs = user_seqs[user_seqs.apply(lambda x: len(x) >= min_interactions)]
    item_set = set(df["movieId"])
    item2id = {item: idx + 1 for idx, item in enumerate(item_set)}
    id2item = {v: k for k, v in item2id.items()}
    encoded_seqs = user_seqs.apply(lambda seq: [item2id[m] for m in seq])
    padded_seqs = encoded_seqs.apply(lambda x: pad_or_truncate(x, max_len))
    df["mappedId"] = df["movieId"].map(item2id)
    item_counts = df["mappedId"].value_counts()
    item_freq = (item_counts / item_counts.sum()).to_dict()
    return padded_seqs, item2id, id2item, item_freq

def pad_or_truncate(seq, max_len):
    if len(seq) >= max_len:
        return seq[-max_len:]
    else:
        return [0] * (max_len - len(seq)) + seq

def split_sequences_by_user(padded_seqs, train_ratio=0.7, val_ratio=0.15):
    train, val, test = [], [], []
    for seq in padded_seqs:
        non_zero = [x for x in seq if x != 0]
        n = len(non_zero)
        if n < 5:
            continue
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        train_seq = pad_or_truncate(non_zero[:train_end], MAX_LEN)
        val_seq = pad_or_truncate(non_zero[train_end:val_end], MAX_LEN)
        test_seq = pad_or_truncate(non_zero[val_end:], MAX_LEN)
        train.append(train_seq)
        val.append(val_seq)
        test.append(test_seq)
    return train, val, test

class BERT4RecDataset(Dataset):
    def __init__(self, sequences, num_classes, mask_token_id, mask_prob=MASK_PROB):
        self.sequences = list(sequences)
        self.num_classes = num_classes
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        tokens = seq.copy()
        labels = [-100] * len(seq)
        for i in range(len(seq)):
            if seq[i] == 0:
                continue
            if random.random() < self.mask_prob:
                labels[i] = tokens[i]
                p = random.random() / self.mask_prob
                if p < 0.8:
                    tokens[i] = self.mask_token_id
                elif p < 0.9:
                    tokens[i] = random.randint(1, self.num_classes - 1)
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

class BERT4RecTestDataset(Dataset):
    def __init__(self, sequences, mask_token_id):
        self.sequences = sequences
        self.mask_token_id = mask_token_id

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        tokens = seq.copy()
        labels = [-100] * len(seq)
        non_zero_indices = [i for i, v in enumerate(seq) if v != 0]
        mask_indices = non_zero_indices
        for i in mask_indices:
            labels[i] = tokens[i]
            tokens[i] = self.mask_token_id
        return torch.tensor(tokens), torch.tensor(labels)

def build_dataloaders(train_seqs, val_seqs, test_seqs,
                      num_classes, mask_token_id,
                      batch_size=128, mask_prob=MASK_PROB):
    train_dataset = BERT4RecDataset(train_seqs, num_classes, mask_token_id, mask_prob)
    val_dataset = BERT4RecDataset(val_seqs, num_classes, mask_token_id, mask_prob)
    #val_dataset = BERT4RecTestDataset(val_seqs, mask_token_id)
    test_dataset  = BERT4RecTestDataset(test_seqs, mask_token_id)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader


class BERT4RecModel(nn.Module):
    def __init__(self, vocab_size, num_classes,
                 hidden_dim=256, num_layers=4,
                 num_heads=8, max_len=MAX_LEN,
                 dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.item_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        item_embed = self.item_embedding(input_ids)
        pos_embed = self.position_embedding(position_ids)
        x = item_embed + pos_embed

        x = self.layer_norm(x)
        x = self.dropout(x)

        attention_mask = (input_ids != 0)
        x = self.transformer_encoder(x, src_key_padding_mask=~attention_mask)

        logits = self.output_layer(x)
        return logits

def compute_loss(logits, labels):
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fn(logits, labels)


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for input_ids, labels in dataloader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        logits = model(input_ids)
        loss = compute_loss(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_l2o_with_loss(model, dataloader, device, item_freq, k=10):
    model.eval()
    recall_list, ndcg_list = [], []
    total_loss = 0
    count = 0

    item_ids = list(item_freq.keys())
    probs = np.array([item_freq[i] for i in item_ids])
    probs /= probs.sum()

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            total_loss += loss.item()
            count += 1

            for i in range(logits.size(0)):
                for j in range(logits.size(1)):
                    true_item = labels[i, j].item()
                    if true_item in [-100, 0]:
                        continue

                    user_history = set(input_ids[i].cpu().numpy().tolist())
                    user_history.discard(0)

                    candidates = [id_ for id_ in item_ids if id_ not in user_history]
                    if true_item not in candidates:
                        candidates.append(true_item)
                    if len(candidates) < 100:
                        candidates += random.choices(candidates, k=100 - len(candidates))

                    negatives = random.sample(candidates, 99)
                    sampled = [true_item] + negatives
                    scores = logits[i, j][sampled]
                    topk = torch.topk(scores, k).indices.tolist()

                    if 0 in topk:
                        rank = topk.index(0) + 1
                        recall_list.append(1)
                        ndcg_list.append(1 / np.log2(rank + 1))
                    else:
                        recall_list.append(0)
                        ndcg_list.append(0)

    avg_loss = total_loss / count
    return float(np.mean(recall_list)), float(np.mean(ndcg_list)), avg_loss


def train_model(model, train_loader, val_loader, device, item_freq,
                epochs=40, lr=1e-4, patience=10,
                weight_decay=1e-4, save_path=None, stage="Pre-train"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    best_ndcg = 0
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        recall, ndcg, val_loss = evaluate_l2o_with_loss(model, val_loader, device, item_freq)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"[{stage}] Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Recall@10: {recall:.4f} | NDCG@10: {ndcg:.4f}")

        scheduler.step()
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            patience_counter = 0
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return train_losses, val_losses


def fine_tune_last_item(model, train_loader, val_loader, device, item_freq,
                        epochs=10, lr=5e-5, patience=3, save_path=None):
    return train_model(model, train_loader, val_loader, device, item_freq,
                       epochs=epochs, lr=lr, patience=patience,
                       weight_decay=0, save_path=save_path, stage="Fine-tune")


def plot_loss_curve(train_losses, val_losses, save_path="loss_curve.png", title="Loss Curve"):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = "dataset/ratings.dat"
    max_len = 20
    mask_prob = 0.2
    batch_size = 128

    hidden_dim = 256
    num_layers = 2
    num_heads = 2
    dropout = 0.1

    pretrain_lr = 5e-4
    pretrain_epochs = 40
    finetune_lr = 1e-6
    finetune_epochs = 40
    patience = 10
    weight_decay = 1e-5

    save_pretrain_path = "checkpoints/pretrained_model.pt"
    save_finetune_path = "checkpoints/finetuned_model.pt"

    padded_seqs, item2id, id2item, item_freq = load_and_preprocess_movielens(path=data_path, max_len=max_len)

    train_seqs, val_seqs, test_seqs = split_sequences_by_user(padded_seqs)

    num_classes = len(item2id) + 1
    mask_token_id = num_classes
    vocab_size = num_classes + 1

    train_loader, val_loader, test_loader = build_dataloaders(
        train_seqs, val_seqs, test_seqs,
        num_classes=num_classes,
        mask_token_id=mask_token_id,
        batch_size=batch_size
    )

    model = BERT4RecModel(
        vocab_size=vocab_size,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_len=max_len,
        dropout=dropout
    ).to(device)

    print("Starting Pretrain...")
    pretrain_train_losses, pretrain_val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        item_freq=item_freq,
        epochs=pretrain_epochs,
        lr=pretrain_lr,
        patience=patience,
        weight_decay=weight_decay,
        save_path=save_pretrain_path,
        stage="Pre-train"
    )

    plot_loss_curve(
        pretrain_train_losses,
        pretrain_val_losses,
        save_path="loss_curve_pretrain.png",
        title="Pre-train Loss Curve"
    )

    print("\nStarting Fine-tuning on Last-Item Prediction...")

    finetune_train_loader = DataLoader(
        BERT4RecTestDataset(train_seqs, mask_token_id),
        batch_size=batch_size,
        shuffle=True
    )

    finetune_val_loader = DataLoader(
        BERT4RecTestDataset(val_seqs, mask_token_id),
        batch_size = batch_size
    )

    finetune_train_losses, finetune_val_losses = fine_tune_last_item(
        model=model,
        train_loader=finetune_train_loader,
        val_loader=finetune_val_loader,
        device=device,
        item_freq=item_freq,
        epochs=finetune_epochs,
        lr=finetune_lr,
        patience=patience,
        save_path=save_finetune_path
    )

    plot_loss_curve(
        finetune_train_losses,
        finetune_val_losses,
        save_path="loss_curve_finetune.png",
        title="Fine-tune Loss Curve"
    )

    model.load_state_dict(torch.load(save_finetune_path, weights_only=True))
    print("[Notice] Evaluation method: Re-ranking with 99 negative samples + 1 positive (as per assignment option).")
    recall, ndcg, _ = evaluate_l2o_with_loss(model, test_loader, device, item_freq)
    print(f"[Final Test] Recall@10 = {recall:.4f}, NDCG@10 = {ndcg:.4f}")




