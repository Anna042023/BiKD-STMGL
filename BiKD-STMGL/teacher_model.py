import torch
import torch.nn as nn
import torch.nn.functional as F
from openai import OpenAI
import json, numpy as np, re, time

client = OpenAI(
    api_key='sk-070007d5096342d59986e5e26302bb4f',
    base_url='https://api.deepseek.com'
)
AVAILABLE_MODELS = ["deepseek-chat", "deepseek-reasoner", "deepseek-coder"]


class TeacherModel:
    def __init__(self, model_name="deepseek-chat", temperature: float = 0.25, timeout_s: int = 30):
        self.model_name = model_name if model_name in AVAILABLE_MODELS else "deepseek-chat"
        self.temperature = float(temperature)
        self.timeout_s = int(timeout_s)

    def evaluate(self, y_pred_seq: torch.Tensor, y_true_seq: torch.Tensor):
        def safe_json_parse(text: str):
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                raise ValueError("No JSON object found")
            obj = json.loads(match.group(0))
            score = float(np.clip(obj.get("score", 0.5), 0, 1))
            comment = obj.get("explanation", obj.get("comment", "无说明"))
            return score, comment

        try:
            pred = y_pred_seq.detach().flatten().cpu().numpy()
            true = y_true_seq.detach().flatten().cpu().numpy()
            T = len(pred)
            pred, true = pred[:min(T, 36)], true[:min(T, 36)]

            pz = (pred - pred.mean()) / (pred.std() + 1e-6)
            tz = (true - true.mean()) / (true.std() + 1e-6)


            r = np.corrcoef(pz, tz)[0, 1]
            dir_acc = np.mean(np.sign(np.diff(pz)) == np.sign(np.diff(tz)))
            trend_score = 0.5 * (r + 1) + 0.5 * dir_acc


            seg_num = 4
            seg_len = max(3, len(pz)//seg_num)
            seg_scores = []
            for i in range(0, len(pz), seg_len):
                p_seg = pz[i:i+seg_len]; t_seg = tz[i:i+seg_len]
                if len(p_seg) < 3: continue
                r_s = np.corrcoef(p_seg, t_seg)[0, 1]
                d_s = np.mean(np.sign(np.diff(p_seg)) == np.sign(np.diff(t_seg)))
                seg_scores.append(0.5 * (r_s + 1) + 0.5 * d_s)
            seg_score = np.mean(seg_scores) if seg_scores else trend_score

            mae = np.mean(np.abs(pred - true))
            rmse = np.sqrt(np.mean((pred - true) ** 2))
            mae_norm = np.exp(-mae / (np.abs(true).mean() + 1e-3))
            rmse_norm = np.exp(-rmse / (np.std(true) + 1e-3))
            mag_score = 0.5 * mae_norm + 0.5 * rmse_norm

            S_stat = np.clip(0.45 * seg_score + 0.55 * mag_score, 0, 1)
            prompt = f"""
            你是一名精通 **车道级时空交通流预测（Lane-level Spatio-Temporal Traffic Flow Prediction）** 的智能教师模型，当前处于一个学术研究中双向知识蒸馏的教师模型提示词学习阶段。
            该阶段构建了一种基于大语言模型的双向知识蒸馏机制。知识蒸馏是一种典型的模型压缩与知识迁移方法，通常通过让轻量级的学生模型模仿复杂教师模型的输出，实现性能提升。
            在该研究中，将上述传统知识蒸馏进一步扩展为双向交互式蒸馏：学生模型负责生成交通流预测结果，教师模型则基于提示词对学生模型的预测进行语义层面的评估，计算预测与场景间的一致性得分，并借助一致性损失反向优化学生模型，从而构建起“预测—评估—反馈”的闭环学习系统。

            目前学生模型的预测性能为：
            - MAE：5.7806
            - MAPE：28.69%
            - RMSE：8.2390

            学习目标：
            - MAE≈4.0、MAPE≈20%、RMSE≈7.0，并保证三者**同时**下降。

            【Remark｜指标定义（记号：ŷ_t 为预测，y_t 为真实）】
            - MAE = (1/T) ∑ |ŷ_t - y_t|
            - RMSE = √[(1/T) ∑ (ŷ_t - y_t)^2]
            - MAPE = (100/T) ∑ (|ŷ_t - y_t| / max(|y_t|, 1e-3))
            若结果非最优，请基于趋势一致性、时间对齐与幅值偏差的平衡，促使三指标**同时下降**。

            【预测序列】{pred.tolist()}
            【真实序列】{true.tolist()}

            请仅输出 JSON，如：
            {{"score": 0.83, "explanation": "趋势吻合良好，略有偏低。"}}
            """
            llm_score, comment = 0.5, "无说明"
            for attempt in range(2):
                try:
                    resp = client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=180,
                        timeout=self.timeout_s
                    )
                    text = resp.choices[0].message.content.strip()
                    llm_score, comment = safe_json_parse(text)
                    break
                except Exception as e:
                    print(f"[LLM RETRY {attempt+1}/2] {e}")
                    time.sleep(0.6)

            # 融合语义与统计得分（统计为主）
            final_score = float(np.clip(0.6 * S_stat + 0.4 * llm_score, 0, 1))
            if comment in ["无说明", "", None]:
                if seg_score > 0.8:   comment = "趋势吻合良好"
                elif seg_score > 0.6: comment = "趋势较接近，局部偏差"
                else:                 comment = "趋势偏弱或幅值偏离"

            return final_score, comment

        except Exception as e:
            print(f"[TeacherModel ERROR] {e}")
            return 0.5, "Teacher fallback"

class BatchedDTW(nn.Module):
    def __init__(self, window: int = 4):
        super().__init__()
        self.window = int(window)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        assert X.shape == Y.shape, "X and Y shapes must match"
        B, T, N, C = X.shape
        device = X.device
        w = self.window
        total = 0.0

        for n in range(N):
            x = X[:, :, n, :].detach()
            y = Y[:, :, n, :].detach()
            A = x.unsqueeze(2)
            B_ = y.unsqueeze(1)
            D = torch.norm(A - B_, dim=-1)
            D = torch.nan_to_num(D, nan=0.0, posinf=1e3, neginf=1e3)

            inf = torch.tensor(float('inf'), device=device, dtype=D.dtype)
            R = torch.full((B, T + 1, T + 1), inf, device=device, dtype=D.dtype)
            R[:, 0, 0] = 0.0

            for i in range(1, T + 1):
                j_lo = max(1, i - w)
                j_hi = min(T, i + w)
                left_up = R[:, i - 1, j_lo - 1: j_hi]
                left    = R[:, i,     j_lo - 1: j_hi]
                up      = R[:, i - 1, j_lo:     j_hi + 1]
                min_prev = torch.minimum(torch.minimum(left_up, left), up)
                R[:, i, j_lo: j_hi + 1] = D[:, i - 1, j_lo - 1: j_hi] + min_prev

            dist = R[:, T, T]
            total += torch.nan_to_num(dist.mean(), nan=0.0, posinf=1e3, neginf=1e3)

        return total / N

class BiKDLoss(nn.Module):
    def __init__(self,
                 alpha=0.9, beta=0.04, gamma=0.02, delta=0.28,
                 lambda_smooth=0.06, dtw_window=4):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.delta = alpha, beta, gamma, delta
        self.lambda_smooth = lambda_smooth
        self.dtw = BatchedDTW(window=dtw_window)

    def spatial_correlation(self, Y: torch.Tensor) -> torch.Tensor:
        B, T, N, C = Y.shape
        Yb = Y.mean(dim=1).squeeze(-1)
        Yb = Yb - Yb.mean(dim=0, keepdim=True)
        std = Yb.std(dim=0, keepdim=True) + 1e-6
        Yb = Yb / std
        cov = (Yb.t() @ Yb) / (B - 1 + 1e-6)
        d = torch.sqrt(torch.clamp(torch.diag(cov), min=1e-6))
        denom = torch.ger(d, d) + 1e-6
        corr = cov / denom
        return torch.nan_to_num(corr, nan=0.0)

    def forward(self, y_pred, y_true, llm_score, Q_seq):
        L_pred = F.smooth_l1_loss(y_pred, y_true)
        L_temporal = self.dtw(y_pred, y_true)
        Gamma_p = self.spatial_correlation(y_pred)
        Gamma_t = self.spatial_correlation(y_true)
        L_spatial = torch.norm(Gamma_p - Gamma_t, p='fro') / Gamma_p.numel()
        L_smooth  = torch.mean(torch.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]))


        with torch.no_grad():
            eps = torch.quantile(torch.abs(y_true), 0.10)
        rel = torch.abs(y_pred - y_true) / torch.clamp(torch.abs(y_true), min=eps)
        L_rel = rel.mean()
        bias = ((y_pred - y_true) / torch.clamp(torch.abs(y_true), min=eps)).mean()
        L_bias = torch.abs(bias)


        llm_score = float(np.clip(llm_score, 0.0, 1.0))
        adaptive_delta = self.delta * (0.5 + 0.5 * llm_score)
        L_consistency = torch.abs(torch.tensor(llm_score, device=y_pred.device, dtype=y_pred.dtype)
                                  - Q_seq.mean())

        loss = (self.alpha * L_pred +
                self.beta  * L_temporal +
                self.gamma * (L_spatial + self.lambda_smooth * L_smooth) +
                adaptive_delta * L_consistency +
                0.12 * L_rel +
                0.02 * L_bias)
        return torch.nan_to_num(loss, nan=0.0, posinf=1e3, neginf=1e3)
