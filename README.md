# FastSpiceDnn

This repository contains the implementation of a Fast Solver / DNN-based approach to approximate SPICE-based analog crossbar simulations. It focuses on replicating the simulation environment of *Yi Li et al. (Nature Electronics 2022)* for a 706x706 memristor array, specifically addressing the trade-off between simulation accuracy (accounting for IR drop and parasitics) and computational speed.

## Project Overview

### Research Problem
Simulating large-scale analog crossbar arrays (e.g., 706x706 for 14x14 MNIST) with high fidelity is computationally expensive. Standard SPICE simulations must account for:
- **Parasitic Resistances:** Wire resistance ($R_{wire}$) and electrode contact resistance ($R_{contact}$) causing significant IR drop.
- **Device Variability:** Write noise (programming errors) and Read noise (cycle-to-cycle variation).
- **Non-ideal behavior:** KCL/KVL solution over a massive resistive mesh.

### Approach: Shape-Preserving Fast Solver
We propose a hybrid "Fast Solver" (V5) that combines:
1.  **Physics Approximation:** An analytical solver that calculates voltages assuming an ideal mesh (fast but inaccurate due to ignoring parasitics).
2.  **AI Scaler (DNN):** A neural network that predicts a multiplicative scaling factor to correct the physics approximation, learning the non-linear IR drop patterns from SPICE ground truth.

This approach achieves high correlation ($R^2$) with actual SPICE simulations while being orders of magnitude faster, enabling rapid "chip-in-the-loop" training simulations.

## Method

### 1. High-Fidelity Data Generation
*   **Script:** `generate_spice_14x14.py`
*   **Engine:** `ngspice`
*   **Details:**
    *   Generates a 706x706 resistive mesh netlist representing a 14x14 MNIST neural network (Input 196 $\to$ Hidden 500 $\to$ Output 10).
    *   Injects parasitic resistances ($R_{wire}=2.5\Omega$, $R_{contact}=50\Omega$) based on Manhattan distance to model position-dependent IR drop.
    *   Adds random device conductance variations (Write Noise) and measurement noise (Read Noise).
    *   Saves input patterns, conductance matrices ($G$), and SPICE-simulated node voltages to `madem_paper_spice_14x14.pt`.

### 2. Fast Solver Training
*   **Script:** `train_fast_solver_14x14.py`
*   **Model:** `ShapePreservingSolver` (V5)
*   **Architecture:**
    *   Takes input image, approximate physics solution, and Row/Col conductance sums.
    *   Predicts a scalar correction factor in the range [0.5, 1.2].
    *   Final output = $V_{approx} \times Scale_{pred}$.
*   **Training:** Optimizes L1 Loss between the solver prediction and SPICE ground truth.

### 3. Reliability Verification
*   **Script:** `check_v5_reliability.py`
*   **Metric:** $R^2$ Score & Mean Absolute Error (MAE).
*   **Comparison:** Evaluates "Physics Only" (Ideal assumption) vs. "AI Solver V5" (Ours) against the SPICE ground truth.
*   **Output:** Generates scatter plots showing the alignment with real SPICE voltages.

### 4. Application: Chip Training
*   **Script:** `train_final_chip_v5.py`
*   **Purpose:** Uses the trained Fast Solver as a differentiable surrogate environment to train the weights of the memristor chip for MNIST classification.
*   **Technique:** Contrastive Nudging with symmetric conductance updates.

## Repository Structure

| File | Description |
|------|-------------|
| `generate_spice_14x14.py` | Generates SPICE netlists and runs `ngspice` to create the ground truth dataset. |
| `train_fast_solver_14x14.py` | Trains the hybrid AI-Physics solver model (`fast_solver_v5.pth`). |
| `check_v5_reliability.py` | Validates the solver's accuracy against the test set. |
| `train_final_chip_v5.py` | Demonstrates downstream task training (MNIST) using the Fast Solver. |
| `data/` | Directory for MNIST dataset (downloaded automatically). |

## Usage

### Prerequisites
*   Python 3.8+
*   PyTorch, torchvision
*   `ngspice` (Required only for `generate_spice_14x14.py`)
*   `numpy`, `matplotlib`, `tqdm`

### Execution Steps

1.  **Generate Data** (Requires `ngspice` installed):
    ```bash
    python generate_spice_14x14.py
    # Output: madem_paper_spice_14x14.pt
    ```

2.  **Train Fast Solver**:
    ```bash
    python train_fast_solver_14x14.py
    # Output: fast_solver_v5.pth
    ```

3.  **Verify Reliability**:
    ```bash
    python check_v5_reliability.py
    # Output: R2 score logs and scatter plots
    ```

4.  **(Optional) Train Chip on MNIST**:
    ```bash
    python train_final_chip_v5.py
    # Output: best_chip_sota.pth
    ```

## Notes
*   **Dataset:** The generated dataset (`madem_paper_spice_14x14.pt`) is **not** included in the repository due to size. You must run the generation script locally.
*   **Performance:** The data generation process simulates 2000 samples and may take several hours depending on CPU performance.
*   **Device:** Scripts automatically detect Apple Silicon (MPS) or CUDA if available, falling back to CPU.

## Citation
This code implements concepts from:
> Yi Li et al., "In situ training of multilayer perceptron networks on a memristor crossbar array," *Nature Electronics*, 2022.
