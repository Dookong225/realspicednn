import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Device: {device} | Model: V5 (Shape-Preserving Scaler)")

# =========================================================
# 1. 데이터셋 (V3와 동일하게 G 통계량 사용)
# =========================================================
class Spice706Dataset(Dataset):
    def __init__(self, filename='madem_paper_spice_14x14.pt'):
        print(f"📖 데이터 로딩 중... ({filename})")
        data = torch.load(filename, weights_only=False)
        self.inputs = data['inputs'].float()
        self.targets = data['targets'].float()
        
        print("   -> G 행렬 통계량 추출 중...")
        # CNN처럼 이미지를 쓰지 않고, 다시 '통계량'으로 돌아감 (CNN 번짐 방지)
        gs_list = [g.to_dense() for g in data['Gs']]
        G_tensor = torch.stack(gs_list).float()
        
        # [핵심] Row/Col Sum 정보만 추출 (위치 정보 보존 + 번짐 없음)
        self.g_row_sum = G_tensor.sum(dim=2) * 100
        self.g_col_sum = G_tensor.sum(dim=1) * 100
        
        # 원본 G 행렬은 이제 필요 없으므로 메모리 해제
        del gs_list
        del G_tensor
        print("✅ 로드 완료")

    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx): 
        return self.inputs[idx], self.g_row_sum[idx], self.g_col_sum[idx], self.targets[idx]

# =========================================================
# 2. V5: Multiplicative Scaler (모양은 물리가, 크기는 AI가)
# =========================================================
class ShapePreservingSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.N = 706
        self.G_sense = 1e-3

        # 입력: Physics전압(510) + RowSum(706) + ColSum(706) + Img(196)
        input_dim = 510 + 706 + 706 + 196
        
        # 스케일 팩터 예측 네트워크 (출력 범위 0.0 ~ 1.2)
        self.scaler_net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            
            nn.Linear(1024, 510), # 출력 노드별 스케일 팩터
            nn.Sigmoid() # 0 ~ 1 사이 값으로 강제
        )
        
        # 1차 물리 엔진에서 쓸 마스크 (미리 생성)
        mask = torch.zeros((706, 706), device=device)
        mask[0:196, 196:696] = 1; mask[196:696, 0:196] = 1
        mask[196:696, 696:706] = 1; mask[696:706, 196:696] = 1
        self.register_buffer('mask', mask)

    # G 행렬 없이 통계량과 Image만으로 근사 계산 (메모리 절약 + 속도)
    # 하지만 정확한 Shape를 위해 학습 때는 G_matrix를 쓰지 않고
    # *Dataset에 저장된 row/col sum* 만으로는 KCL을 못 풂.
    # -> 따라서 학습 루프 내에서 G_matrix를 쓰지 않는 대신,
    #    "Dataset에 저장된 targets"와 비교만 수행.
    #    Wait! forward를 하려면 v_approx가 필요한데, v_approx를 구하려면 G가 필요함.
    #    => Dataset 구조를 살짝 변경해서 __getitem__에서 G를 줘야 함?
    #    => 아니면 v_approx 자체를 미리 계산해서 데이터셋에 저장해두는 게 베스트.
    #    => 코드가 복잡해지니, 여기서는 "G_matrix 없이 근사"하는 트릭 대신
    #       데이터셋에서 G를 다시 받아오도록 수정 (메모리 좀 먹더라도 정확도 우선)
    
    # (수정) Dataset에서 G를 다시 뱉도록 변경 안 하고,
    # 그냥 "V_approx"를 입력으로 받도록 설계.
    # -> 즉, 물리 엔진 계산은 외부(Training Loop)에서 하지 않고
    #    모델 내부에서 "입력된 통계량"으로 추론? 아니면 G를 받기?
    #    -> G를 받는 게 맞음. V5의 핵심은 "정확한 물리식 * AI보정" 이니까.

    def solve_physics_approx(self, img_bin, G_row_sum, G_col_sum):
        # [트릭] G 행렬 전체를 로딩하면 V4처럼 터질 수 있음.
        # 대신, Row/Col Sum 정보를 이용해서 "가상의 전압"을 추정하거나
        # 아니면 학습 데이터셋에 G Matrix를 다시 포함시켜야 함.
        # ==> 파트너의 맥북 성능을 믿고 G Matrix를 다시 포함시킵시다.
        pass 

class Spice706Dataset_V5(Dataset):
    def __init__(self, filename='madem_paper_spice_14x14.pt'):
        print(f"📖 데이터 로딩 중... ({filename})")
        data = torch.load(filename, weights_only=False)
        self.inputs = data['inputs'].float()
        self.targets = data['targets'].float()
        
        # G 행렬 (Dense)
        print("   -> G 행렬 압축 해제...")
        gs_list = [g.to_dense() for g in data['Gs']]
        self.Gs = torch.stack(gs_list).float()
        
        # 통계량 미리 계산
        self.g_row_sum = self.Gs.sum(dim=2) * 100
        self.g_col_sum = self.Gs.sum(dim=1) * 100
        print("✅ 로드 완료")

    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx): 
        return self.inputs[idx], self.Gs[idx], self.g_row_sum[idx], self.g_col_sum[idx], self.targets[idx]

class ShapePreservingSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.N = 706
        self.G_sense = 1e-3
        
        # 입력: Physics전압(510) + RowSum(706) + ColSum(706) + Img(196)
        input_dim = 510 + 706 + 706 + 196
        
        self.scaler_net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(1024, 510),
            nn.Sigmoid() # 결과는 0.0 ~ 1.0 (감쇠 비율)
        )
        
        # Output Scale 조정 (0.5 ~ 1.2 범위로 매핑)
        # 보통 IR Drop은 전압을 깎아먹으므로 0.xxx가 나와야 함.
        # 하지만 가끔 노이즈로 튈 수 있으니 1.2까지 여유를 줌.
        self.output_scale = 1.2 

    def solve_physics_approx(self, img_bin, G_matrix):
        # (기존 물리 엔진 - 정확한 Shape 제공자)
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
        # 1. 물리 엔진: "모양(Shape)" 결정
        v_approx = self.solve_physics_approx(img_bin, G_matrix)
        
        # 2. AI Scaler: "비율(Ratio)" 결정
        net_input = torch.cat([img_bin, v_approx, g_row, g_col], dim=1)
        raw_scale = self.scaler_net(net_input)
        
        # 0.5 ~ 1.2 사이의 값으로 변환 (초기값은 0.85 근처가 됨)
        scale_factor = raw_scale * 0.7 + 0.5 
        
        # 3. 최종 출력 = 물리값 * 비율
        # (더하기가 아니라 곱하기! 밀림 현상 원천 봉쇄)
        return v_approx * scale_factor

# =========================================================
# 3. 학습 루프
# =========================================================
if __name__ == "__main__":
    dataset = Spice706Dataset_V5("madem_paper_spice_14x14.pt")
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    model = ShapePreservingSolver().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.003, steps_per_epoch=len(train_loader), epochs=200)
    
    # L1 Loss (모양 맞추기 + 이상치 무시)
    criterion = nn.L1Loss()
    
    print("\n🔥 [V5 Final] 형상 보존 스케일링 모델 학습 (200 Epochs)...")
    
    pbar = tqdm(range(200), desc="Training")
    for epoch in pbar:
        model.train()
        total_loss = 0
        
        for imgs, Gs, g_rows, g_cols, targets in train_loader:
            imgs, Gs = imgs.to(device), Gs.to(device)
            g_rows, g_cols = g_rows.to(device), g_cols.to(device)
            targets = targets.to(device)
            target_roi = targets[:, 196:] 
            
            optimizer.zero_grad()
            # V5는 곱하기 방식이라 0 근처에서 학습이 불안정할 수 있음
            # -> targets가 0인 경우는 거의 없으므로(누설전류) 괜찮음
            preds = model(imgs, Gs, g_rows, g_cols)
            
            loss = criterion(preds * 1000, target_roi * 1000)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        pbar.set_postfix({'Loss(mV)': f"{avg_loss:.4f}"})

    print("✅ 학습 완료!")

    # =========================================================
    # 4. 검증
    # =========================================================
    model.eval()
    
    all_err_phys = []
    all_err_ai = []
    
    print("📊 테스트셋 전체 검증 중...")
    with torch.no_grad():
        for imgs, Gs, g_rows, g_cols, targets in test_loader:
            imgs, Gs = imgs.to(device), Gs.to(device)
            g_rows, g_cols = g_rows.to(device), g_cols.to(device)
            targets = targets.to(device)
            real_target = targets[:, 196:]
            
            approx = model.solve_physics_approx(imgs, Gs)
            hybrid = model(imgs, Gs, g_rows, g_cols)
            
            err_p = torch.abs(real_target - approx) * 1000
            err_a = torch.abs(real_target - hybrid) * 1000
            
            all_err_phys.append(err_p.cpu().numpy())
            all_err_ai.append(err_a.cpu().numpy())
            
    all_err_phys = np.concatenate(all_err_phys).flatten()
    all_err_ai = np.concatenate(all_err_ai).flatten()
    
    mean_phys = np.mean(all_err_phys)
    mean_ai = np.mean(all_err_ai)
    improvement = (1 - mean_ai/mean_phys) * 100
    
    print(f"\n🏆 [최종 성적표 (V5 - Shape Preserving)]")
    print(f"   Physics Only 평균 오차: {mean_phys:.4f} mV")
    print(f"   AI Solver V5 평균 오차: {mean_ai:.4f} mV")
    print(f"   -> 개선율: {improvement:.2f}%")
    
    # 그래프
    sample_img, sample_G, sample_g_row, sample_g_col, sample_target = test_data[0]
    sample_img = sample_img.unsqueeze(0).to(device)
    sample_G = sample_G.unsqueeze(0).to(device)
    sample_g_row = sample_g_row.unsqueeze(0).to(device)
    sample_g_col = sample_g_col.unsqueeze(0).to(device)
    real_target = sample_target[196:].cpu().numpy()
    
    with torch.no_grad():
        approx = model.solve_physics_approx(sample_img, sample_G).cpu().numpy()[0]
        hybrid = model(sample_img, sample_G, sample_g_row, sample_g_col).cpu().numpy()[0]

    plt.figure(figsize=(10, 5))
    plt.plot(real_target[-10:]*1000, 'k-o', label='SPICE (Ground Truth)', linewidth=2)
    plt.plot(approx[-10:]*1000, 'b--', label='Physics Only (Shape Source)')
    plt.plot(hybrid[-10:]*1000, 'r-x', label=f'AI V5 (Multiplicative)', linewidth=2)
    plt.title(f"V5 Shape-Preserving Solver (Imp: {improvement:.1f}%)")
    plt.ylabel("Voltage (mV)")
    plt.legend()
    plt.grid()
    plt.show()
    
    torch.save(model.state_dict(), "fast_solver_v5.pth")