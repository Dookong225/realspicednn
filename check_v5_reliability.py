import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Device: {device} | AI vs Physics Final Showdown")

# =========================================================
# 1. 데이터셋 & 모델 로드 (V5)
# =========================================================
class Spice706Dataset_V5(Dataset):
    def __init__(self, filename='madem_paper_spice_14x14.pt'):
        print(f"📖 데이터 로딩... ({filename})")
        data = torch.load(filename, weights_only=False)
        self.inputs = data['inputs'].float()
        self.targets = data['targets'].float()
        gs_list = [g.to_dense() for g in data['Gs']]
        self.Gs = torch.stack(gs_list).float()
        self.g_row_sum = self.Gs.sum(dim=2) * 100
        self.g_col_sum = self.Gs.sum(dim=1) * 100

    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx): 
        return self.inputs[idx], self.Gs[idx], self.g_row_sum[idx], self.g_col_sum[idx], self.targets[idx]

class ShapePreservingSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.N = 706
        self.G_sense = 1e-3
        input_dim = 510 + 706 + 706 + 196
        self.scaler_net = nn.Sequential(
            nn.Linear(input_dim, 2048), nn.BatchNorm1d(2048), nn.GELU(),
            nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.GELU(),
            nn.Linear(1024, 510), nn.Sigmoid()
        )
    
    def solve_physics_approx(self, img_bin, G_matrix):
        vin = torch.where(img_bin > 0, 0.2, 0.0)
        G_in_hid = G_matrix[:, 0:196, 196:696]
        I_hid = torch.bmm(vin.unsqueeze(1), G_in_hid).squeeze(1)
        G_sum_hid = torch.sum(G_matrix[:, 196:696, :], dim=2) + self.G_sense
        v_hid = I_hid / (G_sum_hid + 1e-9)
        G_hid_out = G_matrix[:, 196:696, 696:706]
        I_out = torch.bmm(v_hid.unsqueeze(1), G_hid_out).squeeze(1)
        G_sum_out = torch.sum(G_matrix[:, 696:706, :], dim=2) + self.G_sense
        v_out = I_out / (G_sum_out + 1e-9)
        return torch.cat([v_hid, v_out], dim=1)

    def forward(self, img_bin, G_matrix, g_row, g_col):
        v_approx = self.solve_physics_approx(img_bin, G_matrix)
        net_input = torch.cat([img_bin, v_approx, g_row, g_col], dim=1)
        raw_scale = self.scaler_net(net_input)
        return v_approx * (raw_scale * 0.7 + 0.5)

# R2 Score 수동 계산 함수
def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

if __name__ == "__main__":
    dataset = Spice706Dataset_V5("madem_paper_spice_14x14.pt")
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    model = ShapePreservingSolver().to(device)
    try:
        model.load_state_dict(torch.load("fast_solver_v5.pth"))
        model.eval()
    except:
        print("❌ 모델 없음")
        exit()
        
    all_targets = []
    all_phys_preds = [] # Physics Only
    all_ai_preds = []   # AI V5
    
    print("📊 비교 분석 시작...")
    with torch.no_grad():
        for imgs, Gs, g_rows, g_cols, targets in loader:
            imgs, Gs = imgs.to(device), Gs.to(device)
            g_rows, g_cols = g_rows.to(device), g_cols.to(device)
            targets = targets.to(device)
            
            real_target = targets[:, 196:]
            
            # 1. Physics Only (Base)
            phys_pred = model.solve_physics_approx(imgs, Gs)
            
            # 2. AI Solver (Ours)
            ai_pred = model(imgs, Gs, g_rows, g_cols)
            
            all_targets.append(real_target.cpu().numpy().flatten())
            all_phys_preds.append(phys_pred.cpu().numpy().flatten())
            all_ai_preds.append(ai_pred.cpu().numpy().flatten())
            
    y_true = np.concatenate(all_targets)
    y_phys = np.concatenate(all_phys_preds)
    y_ai = np.concatenate(all_ai_preds)
    
    # 점수 계산
    r2_phys = calculate_r2(y_true, y_phys)
    r2_ai = calculate_r2(y_true, y_ai)
    
    print(f"\n🥊 [Final Scoreboard]")
    print(f"   🔵 Physics Only (Ideal): R² = {r2_phys:.4f}")
    print(f"   🔴 AI Solver V5 (Real):  R² = {r2_ai:.4f}")
    print(f"   ---------------------------------------")
    print(f"   🚀 Performance Boost: +{(r2_ai - r2_phys)*100:.2f} points")

    # 시각화 (Scatter Plot Overlay)
    plt.figure(figsize=(10, 8))
    
    # 데이터 샘플링 (너무 많아서 5000개만)
    idx = np.random.choice(len(y_true), 5000, replace=False)
    
    # Physics 산점도 (파란색)
    plt.scatter(y_true[idx]*1000, y_phys[idx]*1000, 
                alpha=0.2, s=5, color='blue', label=f'Physics Only (R²={r2_phys:.2f})')
    
    # AI 산점도 (빨간색)
    plt.scatter(y_true[idx]*1000, y_ai[idx]*1000, 
                alpha=0.2, s=5, color='red', label=f'AI Solver V5 (R²={r2_ai:.2f})')
    
    # 기준선
    min_val, max_val = y_true.min()*1000, y_true.max()*1000
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect (y=x)')
    
    plt.title("Physics Engine vs. AI Solver (Accuracy Comparison)")
    plt.xlabel("Real SPICE Voltage (mV)")
    plt.ylabel("Predicted Voltage (mV)")
    plt.legend()
    plt.grid()
    plt.show()