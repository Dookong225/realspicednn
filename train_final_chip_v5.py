import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import sys
import numpy as np 

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Final Mission: Fine-Tuning to 95% on {device}")

# =========================================================
# 1. Solver (동일)
# =========================================================
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

# =========================================================
# 2. Chip (동일)
# =========================================================
class DualMemristorChip(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_pos = nn.Parameter(torch.randn(706, 706) * 0.1 + 0.5) 
        self.w_neg = nn.Parameter(torch.randn(706, 706) * 0.1 + 0.5) 
        
        mask = torch.zeros((706, 706), device=device)
        mask[0:196, 196:696] = 1; mask[196:696, 0:196] = 1
        mask[196:696, 696:706] = 1; mask[696:706, 196:696] = 1
        self.register_buffer('mask', mask)
        self.G_min = 2e-6
        self.G_max = 100e-6

    def enforce_symmetry(self):
        with torch.no_grad():
            self.w_pos.data = (self.w_pos.data + self.w_pos.data.t()) / 2
            self.w_neg.data = (self.w_neg.data + self.w_neg.data.t()) / 2

    def get_G_pos(self):
        return (torch.sigmoid(self.w_pos) * (self.G_max - self.G_min) + self.G_min) * self.mask

    def get_G_neg(self):
        return (torch.sigmoid(self.w_neg) * (self.G_max - self.G_min) + self.G_min) * self.mask

# =========================================================
# 3. Training Loop (Fixed)
# =========================================================
if __name__ == "__main__":
    solver = ShapePreservingSolver().to(device)
    try:
        # map_location 추가하여 안전하게 로드
        solver.load_state_dict(torch.load("fast_solver_v5.pth", map_location=device))
        solver.eval()
    except:
        print("❌ fast_solver_v5.pth 파일 없음!")
        sys.exit()

    chip = DualMemristorChip().to(device)
    
    train_loader = DataLoader(
        datasets.MNIST('./data', train=True, transform=transforms.Compose([
            transforms.Resize((14, 14)), transforms.ToTensor()
        ])), batch_size=64, shuffle=True
    )

    # 1. Optimizer
    optimizer = optim.AdamW(chip.parameters(), lr=0.01)
    
    # [수정됨] verbose=True 제거
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    print("\n🔥 Starting Fine-Tuning Process...")
    
    best_acc = 0.0
    
    for epoch in range(30): 
        chip.train()
        total_correct = 0
        total_samples = 0
        
        # Beta Annealing
        current_beta = max(0.2, 0.5 * (0.95 ** epoch))
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1} (Beta={current_beta:.2f})")
        
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            B = imgs.shape[0]
            img_bin = torch.where(imgs.view(B, -1) > 0.1, 1.0, -1.0)
            
            # --- Forward ---
            G_pos = chip.get_G_pos().unsqueeze(0).expand(B, -1, -1)
            G_neg = chip.get_G_neg().unsqueeze(0).expand(B, -1, -1)
            
            out_pos = solver(img_bin, G_pos, G_pos.sum(2)*100, G_pos.sum(1)*100)[:, -10:] * 1000
            out_neg = solver(img_bin, G_neg, G_neg.sum(2)*100, G_neg.sum(1)*100)[:, -10:] * 1000
            
            output_free = out_pos - out_neg
            
            # --- Contrastive Nudging ---
            target_vals = torch.full_like(output_free, -5.0)
            target_vals[range(B), labels] = 10.0
            
            output_nudge = (1 - current_beta) * output_free + current_beta * target_vals
            
            # --- Update ---
            loss = nn.MSELoss()(output_free, output_nudge.detach())
            
            optimizer.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                chip.w_pos.grad *= chip.mask
                chip.w_neg.grad *= chip.mask
            
            optimizer.step()
            chip.enforce_symmetry()
            
            # Accuracy
            preds = torch.argmax(output_free, dim=1)
            acc = (preds == labels).sum().item()
            total_correct += acc
            total_samples += B
            
            curr_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'Acc': f"{total_correct/total_samples*100:.1f}%", 'LR': f"{curr_lr:.5f}"})
            
        epoch_acc = total_correct/total_samples*100
        
        scheduler.step(epoch_acc)
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(chip.state_dict(), "best_chip_sota.pth")
            print(f"🌟 New Best: {best_acc:.2f}% (Saved!)")
            
    print(f"🏆 Final Accuracy: {best_acc:.2f}%")