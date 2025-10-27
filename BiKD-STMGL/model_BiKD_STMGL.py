import os, time, random, platform
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from embedding import TrafficEmbedding
from STMGL import STFeatureLearner
from teacher_model import TeacherModel, BiKDLoss


def set_seed(seed=2025):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def masked_mae(preds, labels):
    return torch.abs(preds - labels).mean()


def masked_mape(preds, labels):
    numerator = torch.abs(preds - labels)
    denominator = (torch.abs(preds) + torch.abs(labels)).clamp(min=1e-3)
    return (numerator / (denominator / 2)).mean() * 100


def masked_rmse(preds, labels):
    return torch.sqrt(((preds - labels) ** 2).mean())

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if hasattr(v, "dtype") and v.dtype.is_floating_point
        }
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for k, p in model.state_dict().items():
            if not hasattr(p, "dtype") or not p.dtype.is_floating_point:
                continue
            if k not in self.shadow:
                self.shadow[k] = p.detach().clone()
            else:
                self.shadow[k].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_to(self, model):
        self.backup = {}
        sd = model.state_dict()
        for k, v in self.shadow.items():
            if k in sd:
                self.backup[k] = sd[k].detach().clone()
                sd[k].copy_(v)

    def restore(self, model):
        if not self.backup: return
        sd = model.state_dict()
        for k, v in self.backup.items():
            if k in sd: sd[k].copy_(v)
        self.backup = {}


class EarlyStopping:
    def __init__(self, patience=15, delta=1e-3, save_path="best_HuaNan3_final.pth"):
        self.patience, self.delta, self.save_path = patience, delta, save_path
        self.best_metrics = None
        self.counter = 0

    @staticmethod
    def better(mae, mape, rmse, prev):
        mae_prev, mape_prev, rmse_prev = prev
        gain_mae  = (mae_prev - mae)  / (mae_prev + 1e-6)
        gain_mape = (mape_prev - mape)/ (mape_prev + 1e-6)
        gain_rmse = (rmse_prev - rmse)/ (rmse_prev + 1e-6)
        cond1 = (gain_mape > 0.002)
        cond2 = (gain_mae > 0 and gain_rmse > 0)
        cond3 = (gain_mae + gain_mape + gain_rmse) > 0.004
        return cond1 or cond2 or cond3

    def __call__(self, mae, mape, rmse, emb, st):
        if self.best_metrics is None:
            self.best_metrics = (mae, mape, rmse)
            torch.save({"embedder": emb.state_dict(), "stlearner": st.state_dict()}, self.save_path)
            print(f"[INFO] ✅ First save: MAE={mae:.4f} | MAPE={mape:.2f}% | RMSE={rmse:.4f}")
            return False

        if self.better(mae, mape, rmse, self.best_metrics):
            self.best_metrics = (mae, mape, rmse)
            self.counter = 0
            torch.save({"embedder": emb.state_dict(), "stlearner": st.state_dict()}, self.save_path)
            print(f"[INFO] ✅ Improved Val: MAE={mae:.4f} | MAPE={mape:.2f}% | RMSE={rmse:.4f} | Saved.")
            return False

        self.counter += 1
        print(f"[INFO] ⏳ No improvement ({self.counter}/{self.patience}) "
              f"| Best MAE={self.best_metrics[0]:.4f} MAPE={self.best_metrics[1]:.2f}% RMSE={self.best_metrics[2]:.4f}")
        return self.counter >= self.patience

def main():
    set_seed(2025)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    data_dir = r"/data/ZhangChao/wanganna/HuaNan"
    A = pd.read_csv(os.path.join(data_dir, "Huanan_adj.csv"), index_col=0).values.astype(np.float32)

    def load_npz(name):
        d = np.load(os.path.join(data_dir, name))
        return d["x"].astype(np.float32), d["y"].astype(np.float32)

    x_train, y_train = load_npz("train.npz")
    x_val, y_val     = load_npz("val.npz")
    x_test, y_test   = load_npz("test.npz")
    print(f"[INFO] Dataset: Train={x_train.shape}, Val={x_val.shape}, Test={x_test.shape}")

    x_mean, x_std = x_train.mean(), x_train.std()
    x_train = (x_train - x_mean) / (x_std + 1e-6)
    x_val   = (x_val   - x_mean) / (x_std + 1e-6)
    x_test  = (x_test  - x_mean) / (x_std + 1e-6)

    BATCH, EPOCHS, LR = 16, 250, 2e-4
    d_model, C_in, C_out = 64, 1, 1
    N = x_train.shape[2]
    T_out = y_train.shape[1]
    #T_out= 3
    #y_train = y_train[:, :T_out, :, :]
    #y_val = y_val[:, :T_out, :, :]
    #y_test = y_test[:, :T_out, :, :]

    embedder  = TrafficEmbedding(C_in, d_model, M=2, cheb_K=4).to(device)
    stlearner = STFeatureLearner(N, d_model, H_future=T_out, out_dim=C_out).to(device)
    teacher   = TeacherModel(model_name="deepseek-chat", temperature=0.25, timeout_s=30)

    loss_fn = BiKDLoss(alpha=0.9, beta=0.04, gamma=0.02, delta=0.28,
                       lambda_smooth=0.06, dtw_window=4)

    optimizer = optim.AdamW(
        list(embedder.parameters()) + list(stlearner.parameters()),
        lr=LR, betas=(0.9, 0.98), weight_decay=5e-5
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1)
    A_torch   = torch.tensor(A, dtype=torch.float32, device=device)

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ema_e, ema_s = EMA(embedder), EMA(stlearner)
    early = EarlyStopping(patience=15, delta=1e-3, save_path=os.path.join(data_dir, "best_HuaNan3_final.pth"))

    num_workers = 0 if platform.system().lower().startswith("win") else 4
    pin_memory  = (device.type == "cuda")

    def make_loader(x, y, bs, shuffle):
        ds = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=num_workers,
                          pin_memory=pin_memory, drop_last=shuffle)

    train_loader = make_loader(x_train, y_train, BATCH, True)
    val_loader   = make_loader(x_val,   y_val,   BATCH, False)
    test_loader  = make_loader(x_test,  y_test,  BATCH, False)

    print("[INFO] Start BiKD-FINAL training ...")

    llm_warmup_epochs = 4
    ema_teacher = 0.6
    ema_alpha   = 0.85
    score_margin = 0.01

    for epoch in range(1, EPOCHS + 1):
        embedder.train(); stlearner.train()
        total_loss = 0.0; total_mae = 0.0
        t0 = time.time()

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=100):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                X_te, X_sp = embedder(xb, A_torch)
                y_pred, _  = stlearner(X_te, X_sp, A_torch)

                diff = torch.abs(y_pred - yb)
                with torch.no_grad():
                    eps = torch.quantile(torch.abs(yb), 0.10)
                relerr = (diff / torch.clamp(torch.abs(yb), min=eps)).mean(dim=(1,2,3))
                S_temporal = diff.mean(dim=(1, 2, 3))
                S_distribution = torch.std(y_pred - yb, dim=(1, 2, 3))
                Q_seq = torch.sigmoid(1.0 - 0.4*S_temporal - 0.4*S_distribution - 0.2*relerr)

                use_score = 0.5 if epoch <= llm_warmup_epochs else ema_teacher
                loss = loss_fn(y_pred, yb, use_score, Q_seq)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(list(embedder.parameters()) + list(stlearner.parameters()), 5.0)
            scaler.step(optimizer)
            scaler.update()

            ema_e.update(embedder); ema_s.update(stlearner)

            total_loss += float(loss.detach().cpu())
            total_mae  += float(masked_mae(y_pred.detach(), yb.detach()).cpu())

        scheduler.step()

        if (epoch % 2 == 0) or (epoch == 1):
            with torch.no_grad():
                preds, trues = [], []
                for i, (xv, yv) in enumerate(val_loader):
                    if i >= 10: break
                    xv, yv = xv.to(device), yv.to(device)
                    X_te, X_sp = embedder(xv, A_torch)
                    yp, _ = stlearner(X_te, X_sp, A_torch)
                    preds.append(yp.mean(dim=2)[:, :, 0].mean(dim=0))
                    trues.append(yv.mean(dim=2)[:, :, 0].mean(dim=0))
                if len(preds) > 0:
                    yp_seq = torch.stack(preds).mean(dim=0)
                    yt_seq = torch.stack(trues).mean(dim=0)
                    znorm = lambda z: (z - z.mean()) / (z.std() + 1e-6)
                    yp_n, yt_n = znorm(yp_seq), znorm(yt_seq)
                    try:
                        raw_score, explanation = teacher.evaluate(yp_n, yt_n)
                    except Exception as e:
                        print(f"[LLM] evaluate error: {e}")
                        raw_score, explanation = 0.5, "回退：弱监督"
                    if abs(raw_score - ema_teacher) > score_margin:
                        ema_teacher = ema_alpha * ema_teacher + (1 - ema_alpha) * raw_score

        embedder.eval(); stlearner.eval()
        ema_e.apply_to(embedder); ema_s.apply_to(stlearner)
        val_mae = val_mape = val_rmse = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                X_te, X_sp = embedder(xb, A_torch)
                y_pred, _ = stlearner(X_te, X_sp, A_torch)
                val_mae  += float(masked_mae(y_pred, yb).cpu())
                val_mape += float(masked_mape(y_pred, yb).cpu())
                val_rmse += float(masked_rmse(y_pred, yb).cpu())
        ema_e.restore(embedder); ema_s.restore(stlearner)

        n_val = max(1, len(val_loader))
        val_mae  /= n_val; val_mape /= n_val; val_rmse /= n_val

        print(f"[EPOCH {epoch:03d}] Loss={total_loss/max(1,len(train_loader)):.4f} | "
              f"Val MAE={val_mae:.4f} | MAPE={val_mape:.2f}% | RMSE={val_rmse:.4f} | "
              f"LLM={ema_teacher:.3f} | Time={time.time()-t0:.1f}s")

        if early(val_mae, val_mape, val_rmse, embedder, stlearner):
            break

    print("[INFO] ✅ Training completed.")

    ckpt = torch.load(os.path.join(data_dir, "best_HuaNan3_final.pth"), map_location=device)
    embedder.load_state_dict(ckpt["embedder"]); stlearner.load_state_dict(ckpt["stlearner"])
    ema_e.apply_to(embedder); ema_s.apply_to(stlearner)
    embedder.eval(); stlearner.eval()

    with torch.no_grad():
        mae = mape = rmse = 0.0
        for xb, yb in test_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            X_te, X_sp = embedder(xb, A_torch)
            y_pred, _  = stlearner(X_te, X_sp, A_torch)
            mae  += float(masked_mae(y_pred, yb).cpu())
            mape += float(masked_mape(y_pred, yb).cpu())
            rmse += float(masked_rmse(y_pred, yb).cpu())
        mae  /= len(test_loader); mape /= len(test_loader); rmse /= len(test_loader)

    print(f"[TEST RESULT] MAE={mae:.4f} | MAPE={mape:.2f}% | RMSE={rmse:.4f}")
    print("[INFO] ✅ Best model (EMA) evaluated successfully.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
