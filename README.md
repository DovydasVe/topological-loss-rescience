# [Re] A Topological Loss Function for Image Segmentation

Replication of **Clough et al. (2022)**, *"A Topological Loss Function for Deep-Learning Based Image Segmentation Using Persistent Homology"*, IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(12), 8766–8778.

This repository reproduces the test-time topological post-processing framework across **Experiment 1** (MNIST Image Denoising) and **Experiment 3** (ACDC Cardiac MRI Segmentation).

---

## Repository Structure

```text
├── reproduction_topological_loss.ipynb   # Main end-to-end replication notebook
├── loss_functions.py                     # Topological loss & post-processing framework
├── neural_nets_mnist.py                  # Experiment 1: MNIST denoiser, classifier & training
├── neural_nets_acdc.py                   # Experiment 3: ACDC U-Net, training & 3D Dice evaluation
├── article/                              # Replication paper and related Latex files
├── figures/                              # Paper figures & barcode generation notebook (figures.ipynb)
├── requirements.txt                      # Python dependencies
└── LICENSE                               # MIT License
```

---

## Setup & Dependencies

Requires **Python $\ge$ 3.10** and a CUDA-capable GPU (recommended).

**Install standard dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

**Install `TopologyLayer`:**
   `TopologyLayer` (Brüel-Gabrielsson et al., 2020) was written for PyTorch 1.x and requires patching legacy C++ macros (`AT_CHECK` $\to$ `TORCH_CHECK`) and compiling with `-std=c++17` on modern PyTorch ($\ge$ 2.0).
   - **Google Colab / Kaggle / Local Linux:** Handled automatically in **Cell 2** of `reproduction_topological_loss.ipynb`.
   - **Manual Linux Build:** Run the build commands in Cell 2 of the notebook.

---

## Datasets

Both datasets download and extract automatically inside `reproduction_topological_loss.ipynb`:
- **MNIST:** Downloaded via `torchvision.datasets.MNIST` into `./mnist_data`.
- **ACDC:** Downloaded via `huggingface_hub` from a public mirror of the Automated Cardiac Diagnosis Challenge into `./acdc_data`.

---

## Running the Experiments

Open and execute **`reproduction_topological_loss.ipynb`** sequentially:

**Experiment 1**
   - Trains baseline denoising autoencoders across corruption levels $m \in \{2, 4, 6, 8, 10, 12\}$.
   - Applies sample-wise topological post-processing ($\lambda = 0.02$, $\eta = 10^{-4}$, $N_{\text{iter}} = 50$) on $N = 1000$ test images and evaluates MSE (Wilcoxon signed-rank test) and accuracy (McNemar's test).

**Experiment 3**
   - Trains a 4-level 2D U-Net with early stopping on patient-level 3D Dice.
   - Applies slice-wise topological post-processing ($\lambda = 0.01$, $\eta = 10^{-5}$, $N_{\text{iter}} = 5$) across all 1,076 test slices (50 patients) and evaluates volumetric 3D Dice for End-Diastole (ED) and End-Systole (ES).

**Execution Flags**
   - `run_experiments = True` executes the main experiments.
   - `run_grids = False` (default) skips the hyperparameter sensitivity grid searches to save compute time; set to `True` to regenerate the heatmaps.

---

## Hardware & Reproducibility Notes

- **Primary Environment:** Google Colab (NVIDIA L4 GPU, PyTorch 2.x).
- **Alternative Environment:** Kaggle (2 $\times$ NVIDIA Tesla T4 GPUs).
- Random seeds (`seed = 42`) and deterministic CuDNN flags are enforced across all scripts. Minor floating-point variations may occur across different GPU architectures due to non-deterministic CUDA reductions.

---

## License

This project is licensed under the [MIT License](LICENSE).
