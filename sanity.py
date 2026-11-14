"""
sanity.py: checks to run before the full PCA vs autoencoder experiment.

Part A checks the from-scratch PCA against sklearn and against theory.
Part B checks that the linear AE, trained with the EXACT experiment settings,
converges to the PCA solution (Eckart-Young / Baldi & Hornik). If it does not,
any linear-AE vs PCA gap in the full results is an optimisation artefact.

Settings are imported from main.py so they cannot drift from the experiment.
Set SMOKE = False in main.py before running:   python sanity.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA as SkPCA

import main as experiment
from data import load_dataset
from PCA import PCA
from autoencoder import Autoencoder
from train import train_ae
from metrics import mse, r2

if experiment.SMOKE:
    raise RuntimeError("Set SMOKE = False in main.py: the sanity check must use the full-run settings.")

CONFIG = experiment.CONFIG
device = experiment.device
SANITY_DIMS = [2, 16, 64]   # small, medium, and the hardest case in the sweep
SEED = CONFIG['seeds'][0]
A_DIM = 16                  # latent dim used for the sklearn comparison in Part A


def verdict(ok, warn=False):
    return "PASS" if ok else ("WARN" if warn else "FAIL")


def ae_reconstruct(model, X, batch=8192):
    """Full-dataset reconstruction in batches; returns CPU numpy."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def effective_decoder(model, d):
    """
    The linear decoder is affine: x_hat = W z + b. decode(I_d) has rows W[:, i] + b,
    and decode(0) = b, so their difference is W^T. Returns W as (784, d) float64.
    """
    model.eval()
    with torch.no_grad():
        eye = torch.eye(d, device=device)
        zero = torch.zeros(1, d, device=device)
        W_T = model.decode(eye) - model.decode(zero)
    return W_T.T.cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------- Part A
def part_a(X_train, X_test):
    print("\n=== PART A: PCA implementation ===")
    p = X_train.shape[1]

    # A1: with all components, transform -> inverse_transform must be the identity
    pca_full = PCA(n_components=p).fit(X_train)
    a1 = np.max(np.abs(pca_full.inverse_transform(pca_full.transform(X_test)) - X_test))
    print(f"A1 full-rank round trip, max |error|      : {a1:.2e}  {verdict(a1 < 1e-4)}")

    # A2: agreement with sklearn (exact full SVD, so no randomised-solver noise)
    d = A_DIM
    pca = PCA(n_components=d).fit(X_train)
    sk = SkPCA(n_components=d, svd_solver='full').fit(X_train)
    ours_evr = pca.eigenvalues[:d] / pca.eigenvalues.sum()
    evr_diff = np.max(np.abs(ours_evr - sk.explained_variance_ratio_))
    our_mse = mse(X_test, pca.inverse_transform(pca.transform(X_test)))
    sk_mse = mse(X_test, sk.inverse_transform(sk.transform(X_test)))
    mse_rel = abs(our_mse - sk_mse) / sk_mse
    print(f"A2 vs sklearn (d={d}), max |EVR diff|       : {evr_diff:.2e}  {verdict(evr_diff < 1e-5)}")
    print(f"A2 vs sklearn (d={d}), relative MSE diff    : {mse_rel:.2e}  {verdict(mse_rel < 1e-4)}")

    # A3: the loadings must be orthonormal
    V = pca.principal_components
    a3 = np.max(np.abs(V.T @ V - np.eye(d)))
    print(f"A3 orthonormality, max |V^T V - I|        : {a3:.2e}  {verdict(a3 < 1e-8)}")

    # A4: on the fitting data, variance-weighted R^2 equals the explained-variance ratio
    r2_train = r2(X_train, pca.inverse_transform(pca.transform(X_train)))
    a4 = abs(r2_train - pca.ratio)
    print(f"A4 train R^2 vs explained-variance ratio  : {r2_train:.6f} vs {pca.ratio:.6f}"
          f"  (diff {a4:.2e})  {verdict(a4 < 1e-5)}")


# ---------------------------------------------------------------- Part B
def part_b(X_train, X_val, X_test, Y_train, Y_val):
    print("\n=== PART B: linear AE vs PCA (experiment settings) ===")
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)),
                              batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)),
                            batch_size=CONFIG['batch_size'], shuffle=False)

    rows = []
    for d in SANITY_DIMS:
        print(f"\n--- d = {d} ---")
        pca = PCA(n_components=d).fit(X_train)
        pca_train = mse(X_train, pca.inverse_transform(pca.transform(X_train)))
        pca_test = mse(X_test, pca.inverse_transform(pca.transform(X_test)))

        experiment.set_seed(SEED)
        model = Autoencoder(CONFIG['input_dims'], d, CONFIG['h1'], CONFIG['h2'], activation=False).to(device)
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
        model, best_val, epochs_run = train_ae(model, train_loader, val_loader, nn.MSELoss(),
                                               optimizer, device, epochs=CONFIG['epochs'])

        ae_train = mse(X_train, ae_reconstruct(model, X_train))
        ae_test = mse(X_test, ae_reconstruct(model, X_test))

        # B1: Eckart-Young makes PCA optimal on the data it was fit on, so this gap
        # cannot be negative beyond float error. Thresholds (2% / 10%) are judgement calls.
        gap_train = (ae_train - pca_train) / pca_train
        gap_test = (ae_test - pca_test) / pca_test
        if gap_train < -1e-4:
            b1 = "FAIL (negative: data/metric mismatch, i.e. a bug)"
        elif gap_train <= 0.02:
            b1 = "PASS"
        elif gap_train <= 0.10:
            b1 = "WARN (slightly under-converged)"
        else:
            b1 = "FAIL (under-converged: check epochs/patience)"

        # B3: angles between PCA's subspace and the decoder's column space.
        # Trailing angles converge slowest when lambda_d / lambda_{d+1} is close to 1,
        # so a large last angle with a small MSE gap is harmless.
        W = effective_decoder(model, d)
        angles = np.sort(np.degrees(subspace_angles(pca.principal_components, W)))
        eigengap = pca.eigenvalues[d - 1] / pca.eigenvalues[d]

        stopped_early = epochs_run < CONFIG['epochs']
        print(f"B1 train MSE gap  : {gap_train:+.2%}   (PCA {pca_train:.5f}, AE {ae_train:.5f})  {b1}")
        print(f"B2 test  MSE gap  : {gap_test:+.2%}   (PCA {pca_test:.5f}, AE {ae_test:.5f})")
        print(f"B3 subspace angles: mean {angles.mean():.2f} deg, max {angles.max():.2f} deg"
              f"   (eigengap lambda_d/lambda_d+1 = {eigengap:.3f})")
        print(f"B4 epochs run     : {epochs_run}/{CONFIG['epochs']}"
              f"   ({'early stopping fired' if stopped_early else 'hit the epoch cap'})")

        rows.append({'d': d, 'pca_train_mse': pca_train, 'ae_train_mse': ae_train, 'gap_train': gap_train,
                     'pca_test_mse': pca_test, 'ae_test_mse': ae_test, 'gap_test': gap_test,
                     'angle_mean_deg': angles.mean(), 'angle_max_deg': angles.max(),
                     'eigengap': eigengap, 'epochs_run': epochs_run, 'best_val_loss': best_val,
                     'b1_verdict': b1})
    return rows


def main():
    os.makedirs("results", exist_ok=True)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_dataset(
        CONFIG['train_size'], CONFIG['val_size'], CONFIG['SPLIT_SEED'])
    print(f"device: {device} | train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")

    part_a(X_train, X_test)
    rows = part_b(X_train, X_val, X_test, Y_train, Y_val)

    pd.DataFrame(rows).to_csv("results/sanity.csv", index=False)
    print("\nPart B summary saved to results/sanity.csv")


if __name__ == "__main__":
    main()