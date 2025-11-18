"""
LDA (Local Density Approximation) functionals for DFT
LDA（局所密度近似）汎関数

This module implements LDA exchange functional (Slater exchange) for both
spin-unpolarized (RKS) and spin-polarized (UKS) calculations.
Correlation part is currently neglected.

このモジュールはスピン非分極（RKS）とスピン分極（UKS）計算の両方に対応した
LDA交換汎関数（Slater交換）を実装しています。
相関部分は現在無視されています。
"""
import numpy as np


def lda_kernel(electron_density_at_grids):
  """
  Calculate LDA exchange kernel (for TDDFT)
  LDA交換カーネルを計算（TDDFT用）

  The kernel is used in linear response TDDFT calculations.
  カーネルは線形応答TDDFT計算で使用されます。

  Args:
      electron_density_at_grids: Electron density at grid points / グリッド点での電子密度

  Returns:
      Exchange kernel / 交換カーネル
  """
  # Exchange kernel: f_x = -(3/4) * (3/π)^(1/3) * ρ^(1/3)
  # 交換カーネル: f_x = -(3/4) * (3/π)^(1/3) * ρ^(1/3)
  exchange_coeff = (-3.0 / 4.0) * ((3.0 / np.pi) ** (1.0 / 3.0))
  exchange_kernel = exchange_coeff * (electron_density_at_grids ** (1.0 / 3.0))

  # Correlation part is neglected
  # 相関部分は無視

  return exchange_kernel


def lda_potential(electron_density_at_grids):
  """
  Calculate LDA exchange potential
  LDA交換ポテンシャルを計算

  The exchange potential is the functional derivative of exchange energy.
  交換ポテンシャルは交換エネルギーの汎関数微分です。

  Args:
      electron_density_at_grids: Electron density at grid points / グリッド点での電子密度

  Returns:
      Exchange potential / 交換ポテンシャル
  """
  # Slater exchange potential: V_x = -(3/π)^(1/3) * ρ^(1/3)
  # Slater交換ポテンシャル: V_x = -(3/π)^(1/3) * ρ^(1/3)
  # This is the local density approximation for exchange
  # これは交換相互作用の局所密度近似です
  exchange_coeff = -(3.0 / np.pi) ** (1.0 / 3.0)
  exchange_potential = exchange_coeff * (electron_density_at_grids ** (1.0 / 3.0))

  # Correlation part is neglected
  # 相関部分は無視

  return exchange_potential


def lda_energy(electron_density_at_grids, grid_weights):
  """
  Calculate LDA exchange energy
  LDA交換エネルギーを計算

  Integrates the exchange energy density over all space.
  交換エネルギー密度を全空間で積分します。

  Args:
      electron_density_at_grids: Electron density at grid points / グリッド点での電子密度
      grid_weights: Integration weights at grid points / グリッド点での積分重み

  Returns:
      Exchange energy in Hartree / 交換エネルギー（Hartree単位）
  """
  # Slater exchange energy: E_x = -(3/4) * (3/π)^(1/3) * ∫ ρ^(4/3) dr
  # Slater交換エネルギー: E_x = -(3/4) * (3/π)^(1/3) * ∫ ρ^(4/3) dr
  exchange_coeff = (-3.0 / 4.0) * ((3.0 / np.pi) ** (1.0 / 3.0))

  # Numerical integration: ∫ ρ^(4/3) dr ≈ Σ_n w_n * ρ_n^(4/3)
  # 数値積分: ∫ ρ^(4/3) dr ≈ Σ_n w_n * ρ_n^(4/3)
  exchange_energy = np.sum(grid_weights * electron_density_at_grids ** (4.0 / 3.0))
  exchange_energy *= exchange_coeff

  # Correlation part is neglected
  # 相関部分は無視

  return exchange_energy


def lda_potential_spinpol(rho_alpha, rho_beta):
  """
  Calculate spin-polarized LDA exchange potential (for UKS)
  スピン分極LDA交換ポテンシャルを計算（UKS用）

  For UKS, we use spin density functional theory (SDFT).
  Each spin component has its own exchange potential.
  UKSではスピン密度汎関数理論（SDFT）を使用します。
  各スピン成分は独自の交換ポテンシャルを持ちます。

  Args:
      rho_alpha: Alpha spin electron density at grid points / グリッド点でのα電子密度
      rho_beta: Beta spin electron density at grid points / グリッド点でのβ電子密度

  Returns:
      V_x_alpha: Exchange potential for alpha spin / α交換ポテンシャル
      V_x_beta: Exchange potential for beta spin / β交換ポテンシャル
  """
  # Spin-polarized Slater exchange (Local Spin Density Approximation, LSDA)
  # スピン分極Slater交換（局所スピン密度近似、LSDA）
  # V_x^σ = -2^(1/3) * (3/π)^(1/3) * ρ_σ^(1/3)
  # where σ = α or β / ここでσ = α または β
  # The factor 2^(1/3) comes from spin scaling
  # 因子2^(1/3)はスピンスケーリングから生じる

  exchange_coeff = -(2.0 ** (1.0 / 3.0)) * (3.0 / np.pi) ** (1.0 / 3.0)

  # Alpha spin potential / αスピンポテンシャル
  V_x_alpha = exchange_coeff * (rho_alpha ** (1.0 / 3.0))

  # Beta spin potential / βスピンポテンシャル
  V_x_beta = exchange_coeff * (rho_beta ** (1.0 / 3.0))

  # Correlation part is neglected
  # 相関部分は無視

  return V_x_alpha, V_x_beta


def lda_energy_spinpol(rho_alpha, rho_beta, grid_weights):
  """
  Calculate spin-polarized LDA exchange energy (for UKS)
  スピン分極LDA交換エネルギーを計算（UKS用）

  The exchange energy is calculated separately for alpha and beta spins,
  then summed to get the total exchange energy.
  交換エネルギーはαとβスピンに対して別々に計算され、
  その後合計されて全交換エネルギーが得られます。

  Args:
      rho_alpha: Alpha spin electron density at grid points / グリッド点でのα電子密度
      rho_beta: Beta spin electron density at grid points / グリッド点でのβ電子密度
      grid_weights: Integration weights at grid points / グリッド点での積分重み

  Returns:
      Exchange energy in Hartree / 交換エネルギー（Hartree単位）
  """
  # Spin-polarized Slater exchange energy
  # スピン分極Slater交換エネルギー
  # E_x = -(3/4) * (3/π)^(1/3) * 2^(1/3) * ∫ (ρ_α^(4/3) + ρ_β^(4/3)) dr
  # The factor 2^(1/3) comes from spin scaling
  # 因子2^(1/3)はスピンスケーリングから生じる

  exchange_coeff = (-3.0 / 4.0) * ((3.0 / np.pi) ** (1.0 / 3.0)) * (2.0 ** (1.0 / 3.0))

  # Numerical integration for alpha spin: ∫ ρ_α^(4/3) dr ≈ Σ_n w_n * ρ_α,n^(4/3)
  # α電子の数値積分: ∫ ρ_α^(4/3) dr ≈ Σ_n w_n * ρ_α,n^(4/3)
  E_x_alpha = np.sum(grid_weights * rho_alpha ** (4.0 / 3.0))

  # Numerical integration for beta spin: ∫ ρ_β^(4/3) dr ≈ Σ_n w_n * ρ_β,n^(4/3)
  # β電子の数値積分: ∫ ρ_β^(4/3) dr ≈ Σ_n w_n * ρ_β,n^(4/3)
  E_x_beta = np.sum(grid_weights * rho_beta ** (4.0 / 3.0))

  # Total exchange energy / 全交換エネルギー
  exchange_energy = exchange_coeff * (E_x_alpha + E_x_beta)

  # Correlation part is neglected
  # 相関部分は無視

  return exchange_energy


def lda_kernel_spinpol(rho_alpha, rho_beta):
  """
  Calculate spin-polarized LDA exchange kernel (for UKS-TDDFT)
  スピン分極LDA交換カーネルを計算（UKS-TDDFT用）

  The kernel is used in linear response TDDFT calculations for open-shell systems.
  カーネルは開殻系の線形応答TDDFT計算で使用されます。

  Args:
      rho_alpha: Alpha spin electron density at grid points / グリッド点でのα電子密度
      rho_beta: Beta spin electron density at grid points / グリッド点でのβ電子密度

  Returns:
      f_x_alpha: Exchange kernel for alpha spin / α交換カーネル
      f_x_beta: Exchange kernel for beta spin / β交換カーネル
  """
  # Spin-polarized exchange kernel: f_x^σ = -(3/4) * 2^(1/3) * (3/π)^(1/3) * ρ_σ^(1/3)
  # スピン分極交換カーネル: f_x^σ = -(3/4) * 2^(1/3) * (3/π)^(1/3) * ρ_σ^(1/3)
  exchange_coeff = (-3.0 / 4.0) * (2.0 ** (1.0 / 3.0)) * ((3.0 / np.pi) ** (1.0 / 3.0))

  # Alpha spin kernel / αスピンカーネル
  f_x_alpha = exchange_coeff * (rho_alpha ** (1.0 / 3.0))

  # Beta spin kernel / βスピンカーネル
  f_x_beta = exchange_coeff * (rho_beta ** (1.0 / 3.0))

  # Correlation part is neglected
  # 相関部分は無視

  return f_x_alpha, f_x_beta
