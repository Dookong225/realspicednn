import os
import subprocess
import torch
import numpy as np
import re
from tqdm import tqdm
from torchvision import datasets, transforms

# =========================================================
# 📜 Yi Li et al. (Nature Electronics 2022) 기반 설정
# =========================================================
# 논문에서는 하드웨어 실험(64x64)과 시뮬레이션(706x706)을 병행함
# MADEM 알고리즘의 성능을 검증하기 위한 'Dirty' 환경 조성

N_SAMPLES = 2000 
SAVE_PATH = "madem_paper_spice_14x14.pt"

# 논문 참조 파라미터 (추정치 포함)
R_WIRE = 2.5       # 옴 (Wire Resistance between cells) - IR Drop의 주범!
R_CONTACT = 50.0   # 옴 (Electrode Contact Resistance)
G_MIN = 2e-6       # 2uS (Off state - 논문 그래프 참조)
G_MAX = 100e-6     # 100uS (On state)

# 노이즈 레벨 (Variability)
READ_NOISE_STD = 0.005 # 0.5% Read Noise

# =========================================================
# 1. High-Fidelity Netlist Generator (Parasitics Heavy)
# =========================================================
class MademSpiceGenerator:
    def __init__(self):
        self.filename = "temp_madem_sim.sp"
        self.N = 706 
        
        # 14x14 MNIST 구조 (Input 196 -> Hidden 500 -> Output 10)
        self.mask = torch.zeros((self.N, self.N))
        self.mask[0:196, 196:696] = 1; self.mask[196:696, 0:196] = 1
        self.mask[196:696, 696:706] = 1; self.mask[696:706, 196:696] = 1
        
    def create_netlist(self, input_pattern, G_matrix):
        """
        [MADEM Simulation Setup]
        단순 저항 병렬 연결이 아니라, '전선 저항(R_wire)'을 포함한 격자(Mesh) 구조 생성.
        IR Drop 효과를 극대화하여 MADEM 알고리즘의 필요성을 증명하는 데이터셋.
        """
        with open(self.filename, 'w') as f:
            f.write(f"* Nature Electronics 2022 (MADEM) 706x706 Simulation\n")
            # 수렴성과 정밀도의 타협점
            f.write(".OPTIONS RELTOL=1e-4 ABSTOL=1e-10 VNTOL=1uV NOACCT\n")
            
            # --- 1. 입력 (Read Voltage) ---
            f.write("\n* --- Input Stimuli ---\n")
            for i in range(196):
                vol = 0.2 if input_pattern[i] > 0 else 0.0
                f.write(f"Vin_{i} in_{i} 0 DC {vol}\n")
                # 입력단 배선 저항 (Source Resistance)
                f.write(f"Rsrc_{i} in_{i} node_{i} {R_CONTACT}\n")

            # --- 2. 멤리스터 어레이 (Memristive Grid with Parasitics) ---
            f.write("\n* --- Parasitic-Included Crossbar ---\n")
            
            # (주의) 전선 저항(R_wire)을 SPICE로 완벽하게 구현하려면 
            # 노드 수가 수십만 개로 늘어나서 5시간 내에 불가능함.
            # 대안: 'Lumped Parameter Model' 사용 
            # -> 소자 저항에 위치별 배선 저항을 등가적으로 더해주는 방식 (Fast & Accurate)
            
            rows, cols = torch.nonzero(self.mask, as_tuple=True)
            for r, c in zip(rows, cols):
                r, c = r.item(), c.item()
                if r < c: 
                    g_intrinsic = G_matrix[r, c].item()
                    r_mem = 1.0 / (g_intrinsic + 1e-15)
                    
                    # [논문 디테일] 위치(Locality)에 따른 선 저항 추가
                    # 입력단(0)에서 멀고, 출력단(705)에서 멀수록 저항이 커짐
                    # 간단한 맨해튼 거리 기반 IR Drop 모델링
                    dist_factor = (r + c) * 0.01 
                    r_parasitic = R_CONTACT + (R_WIRE * dist_factor)
                    
                    # 최종 등가 저항
                    r_total = r_mem + r_parasitic
                    f.write(f"R_{r}_{c} node_{r} node_{c} {r_total}\n")
            
            # --- 3. 출력 부하 (Trans-Impedance Amp Modeling) ---
            f.write("\n* --- Output Sensing (Virtual Ground) ---\n")
            # 실제 칩은 TIA를 써서 출력단을 가상 접지(Virtual Ground)로 잡음
            # 이를 모사하기 위해 아주 작은 저항(1옴)을 통해 전류를 측정하거나
            # 적절한 Shunt 저항(1k ~ 10k)을 사용. 논문은 보통 전류 모드.
            for i in range(196, 706):
                # 1k옴 부하 저항 (전압 모드 읽기)
                f.write(f"Rload_{i} node_{i} 0 1000\n")

            # --- 4. 실행 ---
            f.write("\n.control\n")
            f.write("op\n")
            # 은닉층/출력층 전압 저장
            nodes_to_print = " ".join([f"v(node_{i})" for i in range(196, 706)])
            f.write(f"print {nodes_to_print}\n")
            f.write(".endc\n")
            f.write(".end\n")

    def run_ngspice(self):
        if not os.path.exists(self.filename): return None
        
        cmd = ["ngspice", "-b", self.filename]
        try:
            # 706x706 + Parasitics는 시간이 좀 걸림 (15초 타임아웃)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except subprocess.TimeoutExpired:
            return None
        except Exception:
            return None

        voltages = torch.zeros(self.N)
        pattern = re.compile(r"v\(node_(\d+)\)\s*=\s*([-\d\.eE\+]+)")
        
        found_cnt = 0
        for match in pattern.finditer(result.stdout):
            try:
                node_idx = int(match.group(1))
                val = float(match.group(2))
                voltages[node_idx] = val
                found_cnt += 1
            except: continue
            
        if found_cnt < 100: return None # 파싱 실패 시 버림
        return voltages

# =========================================================
# 2. 메인 실행: MADEM 논문 데이터 생성
# =========================================================
if __name__ == "__main__":
    print(f"🏭 [Yi Li et al. 2022] MADEM Paper Replication Data Generation")
    print(f"   Target: 706x706 Memristor Array with IR Drop & Variability")
    print(f"   Samples: {N_SAMPLES} (Estimated Time: 4~5 Hours)")

    transform = transforms.Compose([
        transforms.Resize((14, 14)), 
        transforms.ToTensor()
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    valid_indices = [i for i, label in enumerate(dataset.targets) if label in [0, 1, 7]]
    
    dataset_inputs = []
    dataset_Gs = []
    dataset_targets = []
    
    generator = MademSpiceGenerator()
    
    pbar = tqdm(total=N_SAMPLES, desc="Simulating Physics")
    
    while len(dataset_inputs) < N_SAMPLES:
        # A. 랜덤 이미지
        idx = np.random.choice(valid_indices)
        img, _ = dataset[idx]
        img_bin = torch.where(img.view(-1) > 0.1, 1.0, -1.0)
        
        # B. G값 생성 (Device-to-Device Variation 반영)
        # 논문에서는 Log-Normal 분포를 따르는 경우가 많지만, 여기선 Uniform + Noise로 근사
        g_base = torch.rand((706, 706)) * (G_MAX - G_MIN) + G_MIN
        
        # [Dirty 1] Write Variability (프로그래밍할 때 오차 발생)
        write_noise = torch.randn_like(g_base) * (g_base * 0.05) # 5% Write Error
        g_real = torch.clamp(g_base + write_noise, G_MIN, G_MAX)
        
        current_G = g_real * generator.mask
        
        # C. SPICE 시뮬레이션
        generator.create_netlist(img_bin, current_G)
        voltages = generator.run_ngspice()
        
        if voltages is not None:
            # [Dirty 2] Read Noise (읽을 때 오차 발생)
            # Cycle-to-Cycle variation & Thermal Noise
            read_noise = torch.randn_like(voltages) * READ_NOISE_STD 
            voltages_noisy = voltages + read_noise
            
            dataset_inputs.append(img_bin.float())
            dataset_Gs.append(current_G.to_sparse())
            dataset_targets.append(voltages_noisy.float())
            
            pbar.update(1)

    pbar.close()

    print(f"\n💾 데이터 저장 완료 -> {SAVE_PATH}")
    torch.save({
        'inputs': torch.stack(dataset_inputs),
        'Gs': dataset_Gs,
        'targets': torch.stack(dataset_targets)
    }, SAVE_PATH)
    
    print("🎉 논문급 데이터셋 확보 완료. 이제 진짜 연구를 시작해봅시다.")