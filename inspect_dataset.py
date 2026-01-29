import torch
import matplotlib.pyplot as plt
import numpy as np

# 데이터셋 파일 경로 (아까 생성한 파일)
FILENAME = 'madem_paper_spice_14x14.pt'

print(f"🔍 데이터셋 정밀 검사 시작: {FILENAME}")

try:
    data = torch.load(FILENAME, weights_only=False)
except FileNotFoundError:
    print("❌ 파일이 없습니다. 경로를 확인해주세요.")
    exit()

inputs = data['inputs']   # (N, 196)
targets = data['targets'] # (N, 706)
Gs_sparse = data['Gs']    # List of Sparse Tensors

print(f"✅ 데이터 로드 완료")
print(f"   - 총 샘플 수: {len(inputs)}")
print(f"   - Input Shape: {inputs.shape}")
print(f"   - Target Shape: {targets.shape}")

# ---------------------------------------------------------
# 1. 결측치 및 에러 검사 (Sanity Check)
# ---------------------------------------------------------
print("\n[1] 결측치 및 에러 검사")

# NaN / Inf 검사
if torch.isnan(targets).any() or torch.isinf(targets).any():
    print("   ⚠️ 경고: 데이터에 NaN 또는 Inf가 포함되어 있습니다!")
else:
    print("   OK: NaN/Inf 없음.")

# All Zeros 검사 (SPICE 실패 의심)
# 출력 전압이 모두 0인 샘플 개수 확인
zero_samples = (targets.abs().sum(dim=1) == 0).sum().item()
if zero_samples > 0:
    print(f"   ⚠️ 경고: 출력이 모두 0인 샘플이 {zero_samples}개 발견되었습니다. (SPICE 오류 가능성)")
else:
    print("   OK: 데드 샘플(All Zero) 없음.")

# ---------------------------------------------------------
# 2. 값의 분포 확인 (Distribution)
# ---------------------------------------------------------
print("\n[2] 값의 분포 확인")

# G값 샘플 하나 꺼내서 확인
sample_idx = 0
G_sample = Gs_sparse[sample_idx].to_dense()
g_min = G_sample.min().item()
g_max = G_sample.max().item()
g_mean = G_sample.mean().item()

print(f"   - G값 범위 (Sample 0): Min={g_min:.2e}, Max={g_max:.2e}, Mean={g_mean:.2e}")
if g_max < 1e-7:
    print("     ⚠️ 경고: G값이 너무 작습니다. (단위 확인 필요)")

# Target 전압 분포 (Hidden/Output 노드만)
# 입력 노드(0~195)는 제외하고 실제 계산된 노드(196~705)만 확인
target_voltages = targets[:, 196:].numpy().flatten() * 1000 # mV 단위
v_min = target_voltages.min()
v_max = target_voltages.max()
v_mean = target_voltages.mean()

print(f"   - 출력 전압 범위: Min={v_min:.2f}mV, Max={v_max:.2f}mV, Mean={v_mean:.2f}mV")

if v_max < 0.01:
    print("     🚨 비상: 전압이 거의 0mV입니다. 회로가 끊겨 있거나 입력이 안 들어갔습니다.")
elif v_max > 1000:
    print("     🚨 비상: 전압이 너무 높습니다. (발산)")
else:
    print("   OK: 전압 범위가 상식적입니다.")

# ---------------------------------------------------------
# 3. 시각화 (눈으로 확인)
# ---------------------------------------------------------
print("\n[3] 샘플 시각화 (Sample 0)")
plt.figure(figsize=(15, 5))

# A. 입력 이미지
plt.subplot(1, 3, 1)
img = inputs[sample_idx].view(14, 14).numpy()
plt.imshow(img, cmap='gray')
plt.title("Input Image")
plt.colorbar()

# B. G 행렬 (Log Scale로 보기)
plt.subplot(1, 3, 2)
# 0이 있으면 log가 안되므로 아주 작은 값 더함
plt.imshow(np.log10(G_sample.numpy() + 1e-12), cmap='inferno')
plt.title("G Matrix (Log Scale)")
plt.colorbar()

# C. 출력 전압 분포 (Histogram)
plt.subplot(1, 3, 3)
plt.hist(target_voltages, bins=50, color='blue', alpha=0.7)
plt.title("Output Voltage Distribution (mV)")
plt.xlabel("Voltage (mV)")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

print("\n🔍 점검 완료. 그래프와 경고 메시지를 확인하세요.")
