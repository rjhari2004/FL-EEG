import os, warnings, numpy as np, pandas as pd, argparse
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import flwr as fl
import random
warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data" # Looking at the local mapped folder
WINDOW_SEC = 6
FS = 256
WINDOW_SAMPLES = WINDOW_SEC * FS
COMPRESSED_DIM = 64
MU = 0.001
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # CPU-Specific Determinism
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1) # Prevents race conditions in multi-threading
    
    # If you ever move back to GPU, these will be ready:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)
# --- CA MATRIX & DYNAMIC SPECTRAL EXTRACTION ---
# (PASTE YOUR generate_hybrid_ca_matrix AND elite_feature_extraction_vectorized HERE EXACTLY AS BEFORE)
def generate_hybrid_ca_matrix(input_dim, output_dim):
    np.random.seed(42)
    state = np.random.randint(0, 2, input_dim)
    matrix_rows = []
    for t in range(output_dim):
        new_state = np.zeros_like(state)
        left = np.roll(state, -1); center = state; right = np.roll(state, 1)
        if t % 2 == 0: new_state = np.bitwise_xor(left, right)
        else: new_state = np.bitwise_xor(np.bitwise_xor(left, center), right)
        matrix_rows.append(new_state); state = new_state
    ca = np.array(matrix_rows).T.astype(np.float32)
    return np.where(ca == 0, -1.0, 1.0) / np.sqrt(input_dim)

CA_MATRIX = generate_hybrid_ca_matrix(115, COMPRESSED_DIM)

def elite_feature_extraction_vectorized(batch_windows):
    fft_vals = np.abs(np.fft.rfft(batch_windows, axis=2))**2
    W = WINDOW_SEC
    bands = [
        np.sum(fft_vals[:, :, int(1*W):int(4*W)], axis=2),
        np.sum(fft_vals[:, :, int(4*W):int(8*W)], axis=2),
        np.sum(fft_vals[:, :, int(8*W):int(13*W)], axis=2),
        np.sum(fft_vals[:, :, int(13*W):int(30*W)], axis=2),
        np.sum(fft_vals[:, :, int(30*W):int(50*W)], axis=2)
    ]
    flat_feats = np.stack(bands, axis=2).reshape(batch_windows.shape[0], -1)
    return np.tanh(np.dot(flat_feats, CA_MATRIX))

# --- DATA LOADER ---
def load_raw_patient_data(pid, split):
    all_files = os.listdir(DATA_DIR)
    s_files = [f for f in all_files if f.startswith(pid) and "_seizures.csv" in f]
    n_files = [f for f in all_files if f.startswith(pid) and "_noseizures.csv" in f]
    if split == 'train': files = s_files[:len(s_files)//2] + n_files[:len(n_files)//2]
    else: files = s_files[len(s_files)//2:] + n_files[len(n_files)//2:]
    X_list, y_list = [], []
    for f in files:
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, f))
            raw = df.values[:, :23].T.astype(np.float32)
            label = 1 if "_seizures" in f else 0
            n_win = raw.shape[1] // WINDOW_SAMPLES
            if n_win == 0: continue
            trunc = raw[:, :n_win*WINDOW_SAMPLES]
            wins = trunc.reshape(23, n_win, WINDOW_SAMPLES).transpose(1, 0, 2)
            feats = elite_feature_extraction_vectorized(wins)
            X_list.append(feats); y_list.append(np.full(n_win, label))
        except: continue
    if not X_list: return None, None
    return np.vstack(X_list), np.concatenate(y_list)

# --- MODEL ---
class Mish(nn.Module):
    def forward(self, x): return x * torch.tanh(nn.functional.softplus(x))

class AttentionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_fc = nn.Linear(COMPRESSED_DIM, 128)
        self.mish = Mish()
        self.attn = nn.Sequential(nn.Linear(128, 32), Mish(), nn.Linear(32, 128), nn.Sigmoid())
        self.res_block = nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), Mish(), nn.Linear(128, 128), nn.BatchNorm1d(128))
        self.head = nn.Sequential(nn.Linear(128, 64), Mish(), nn.Dropout(0.2), nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x):
        x = self.mish(self.input_fc(x))
        x = x * self.attn(x)
        return self.head(self.mish(self.res_block(x) + x))

# --- FEDPROX CLIENT ---
class FlClient(fl.client.NumPyClient):
    def __init__(self, pid):
        self.pid = pid
        self.model = AttentionClassifier().to(DEVICE)
        self.scaler = StandardScaler()

    def get_parameters(self, config): return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    def set_parameters(self, params):
        sd = self.model.state_dict()
        for k, v in zip(sd.keys(), params): sd[k] = torch.tensor(v)
        self.model.load_state_dict(sd)

    def fit(self, params, config):
        self.set_parameters(params)
        global_sd = {k: torch.tensor(v).to(DEVICE) for k, v in zip(self.model.state_dict().keys(), params)}
        
        X_tr_raw, y_tr_raw = load_raw_patient_data(self.pid, "train")
        if X_tr_raw is None: return self.get_parameters({}), 0, {}
        
        X_scaled = self.scaler.fit_transform(X_tr_raw)
        X_t = torch.FloatTensor(X_scaled)
        y_t = torch.FloatTensor(y_tr_raw).unsqueeze(1)
        
        weights = (1. / torch.bincount(y_t.flatten().long()))[y_t.flatten().long()]
        dl = DataLoader(TensorDataset(X_t, y_t), batch_size=128, sampler=WeightedRandomSampler(weights, len(weights)))
        
        opt = optim.AdamW(self.model.parameters(), lr=0.001)
        self.model.train()
        for _ in range(5):
            for bx, by in dl:
                if bx.size(0) <= 1: continue 
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                opt.zero_grad()
                loss = nn.BCELoss()(self.model(bx), by)
                prox = sum((p - global_sd[name]).pow(2).sum() for name, p in self.model.named_parameters())
                (loss + (MU / 2) * prox).backward(); opt.step()
        return self.get_parameters({}), len(X_tr_raw), {}

    def evaluate(self, params, config):
        # This sends metrics to the dashboard!
        self.set_parameters(params)
        X_te_raw, y_te_raw = load_raw_patient_data(self.pid, "test")
        if X_te_raw is None: return float(0), 0, {"accuracy": 0.0, "f1": 0.0}
        
        # Transform using the scaler fit during training
        X_te_t = torch.FloatTensor(self.scaler.transform(X_te_raw)).to(DEVICE)
        y_te_t = torch.FloatTensor(y_te_raw).unsqueeze(1).to(DEVICE)
        
        self.model.eval()
        with torch.no_grad():
            preds_prob = self.model(X_te_t)
            loss = nn.BCELoss()(preds_prob, y_te_t).item()
            preds = (preds_prob.cpu().numpy().flatten() > 0.5).astype(int)
            
        acc = accuracy_score(y_te_raw, preds)
        f1 = f1_score(y_te_raw, preds, zero_division=0)
        return float(loss), len(X_te_raw), {"accuracy": acc, "f1": f1}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=str, required=True)
    parser.add_argument("--server", type=str, default="server:8080")
    args = parser.parse_args()
    
    print(f"Watch Activated for Patient: {args.pid}")
    
    # 1. Initialize the client
    client_instance = FlClient(pid=args.pid)
    
    # 2. Connect to the Server and run the 15 Global FL Rounds
    fl.client.start_client(server_address=args.server, client=client_instance.to_client())
    
    # =================================================================
    # 3. POST-FL EVALUATION (GLOBAL vs FED-PERSONALIZED)
    # =================================================================
    print(f"\n[{args.pid}] Server disconnected. Starting Final Evaluations...")
    
    model = client_instance.model
    scaler = client_instance.scaler # Use the same scaler from training
    
    # Load Test Data for evaluation
    X_te_raw, y_te_raw = load_raw_patient_data(args.pid, "test")
    if X_te_raw is None:
        print(f"[{args.pid}] No test data found. Exiting.")
        exit()
        
    X_te_t = torch.FloatTensor(scaler.transform(X_te_raw)).to(DEVICE)
    y_te = y_te_raw.flatten()

    # --- A. EVALUATE THE GLOBAL MODEL (Before Fine-Tuning) ---
    model.eval()
    with torch.no_grad(): 
        preds_global = (model(X_te_t).cpu().numpy().flatten() > 0.5).astype(int)
        
    acc_global = accuracy_score(y_te, preds_global)
    f1_global = f1_score(y_te, preds_global, zero_division=0)
    
    print(f"\n[{args.pid}] --- GLOBAL MODEL RESULTS (Before Personalization) ---")
    print(f"[{args.pid}] Accuracy: {acc_global*100:.2f}% | F1-Score: {f1_global:.4f}")

    # --- B. LOCAL PERSONALIZATION PHASE (pFL) ---
    print(f"[{args.pid}] Starting Local Personalization (5 epochs)...")
    
    # Load Training Data for Fine-Tuning
    X_tr_raw, y_tr_raw = load_raw_patient_data(args.pid, "train")
    X_tr_t = torch.FloatTensor(scaler.transform(X_tr_raw))
    y_tr_t = torch.FloatTensor(y_tr_raw).unsqueeze(1)
    
    weights = (1. / torch.bincount(y_tr_t.flatten().long()))[y_tr_t.flatten().long()]
    dl_tr = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=64, sampler=WeightedRandomSampler(weights, len(weights)))
    
    # Fine-tune with a very small learning rate
    opt = optim.AdamW(model.parameters(),lr=1e-5, weight_decay=1e-3)

    for epoch in range(5): 
        model.train()
        for bx, by in dl_tr:
            if bx.size(0) <= 1: continue 
            opt.zero_grad()
            nn.BCELoss()(model(bx.to(DEVICE)), by.to(DEVICE)).backward()
            opt.step()
            
    # --- C. EVALUATE THE FED-PERSONALIZED MODEL (After Fine-Tuning) ---
    model.eval()
    with torch.no_grad(): 
        preds_personal = (model(X_te_t).cpu().numpy().flatten() > 0.5).astype(int)
        
    acc_personal = accuracy_score(y_te, preds_personal)
    f1_personal = f1_score(y_te, preds_personal, zero_division=0)
    import json
    from sklearn.metrics import classification_report
    
    # Generate detailed reports as dictionaries
    rep_global = classification_report(y_te, preds_global, zero_division=0, output_dict=True)
    rep_personal = classification_report(y_te, preds_personal, zero_division=0, output_dict=True)
    
    # Save everything to a JSON file for the dashboard
    results = {
        "pid": args.pid,
        "global_acc": acc_global,
        "global_f1": f1_global,
        "personal_acc": acc_personal,
        "personal_f1": f1_personal,
        "report_global": rep_global,
        "report_personal": rep_personal
    }
    
    os.makedirs("./runs", exist_ok=True)
    with open(f"./runs/patient_{args.pid}.json", "w") as f:
        json.dump(results, f)
    # --- D. PRINT FINAL COMPARISON ---
    improvement = (acc_personal - acc_global) * 100
    
    print(f"\n" + "="*60)
    print(f"🎯 FINAL COMPARISON FOR PATIENT {args.pid}")
    print(f"   [Global Model]          Acc: {acc_global*100:.2f}% | F1: {f1_global:.4f}")
    print(f"   [FedPersonalized Model] Acc: {acc_personal*100:.2f}% | F1: {f1_personal:.4f}")
    print(f"   Personalization Boost:  {improvement:+.2f}%")
    print("="*60 + "\n")