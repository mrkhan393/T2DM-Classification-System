import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.oversampling import ADASYN

class ContrastiveDataset(Dataset):
    """
    Generates triplets for supervised contrastive learning:
    anchor, positive, negative
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.pos_idx = np.where(y == 1)[0]
        self.neg_idx = np.where(y == 0)[0]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        anchor = self.X[idx]
        label = self.y[idx]

        if label == 1:
            pos = self.X[np.random.choice(self.pos_idx)]
            neg = self.X[np.random.choice(self.neg_idx)]
        else:
            pos = self.X[np.random.choice(self.neg_idx)]
            neg = self.X[np.random.choice(self.pos_idx)]

        return anchor, pos, neg, label

def load_and_preprocess_data(csv_path, features, target_col='final_diabetes'):
    df = pd.read_csv(csv_path)
    X = df[features].values.astype(np.float32)
    y = df[target_col].values.astype(np.int64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reshape NOT yet (important: ADASYN needs 2D)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    
    adasyn = ADASYN(random_state=42)
    X_train, y_train = adasyn.fit_resample(X_train, y_train)
    
    # Now reshape for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    return X_train, X_test, y_train, y_test
    
def split_clients_non_iid(X, y, num_clients=10, min_pos_ratio=0.3, max_pos_ratio=0.7, seed=42):
    rng = np.random.default_rng(seed)
    
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    
    clients_data = []
    pos_ptr, neg_ptr = 0, 0
    
    total_samples = len(y)
    client_sizes = [total_samples // num_clients] * num_clients
    remainder = total_samples - sum(client_sizes)
    for i in range(remainder):
        client_sizes[i] += 1

    for c in range(num_clients):
        client_size = client_sizes[c]
        pos_ratio = rng.uniform(min_pos_ratio, max_pos_ratio)
        pos_count = min(len(pos_idx) - pos_ptr, int(client_size * pos_ratio))
        neg_count = min(len(neg_idx) - neg_ptr, client_size - pos_count)
        
        pos_slice = pos_idx[pos_ptr:pos_ptr + pos_count]
        neg_slice = neg_idx[neg_ptr:neg_ptr + neg_count]
        pos_ptr += pos_count
        neg_ptr += neg_count
        
        idx = np.concatenate([pos_slice, neg_slice])
        rng.shuffle(idx)
        
        clients_data.append((X[idx], y[idx]))
        print(f"Client-{c+1}: Pos={len(pos_slice)}, Neg={len(neg_slice)}, Total={len(idx)}")
    
    total_client_samples = sum([len(c[0]) for c in clients_data])
    print(f"\nTotal samples across all clients: {total_client_samples}")
    print(f"Original train size: {len(y)}")
    
    return clients_data
