"""
CIS/TDHF/TDA-TDDFT (Configuration Interaction Singles) calculator class
CIS/TDHF/TDA-TDDFT（単一励起配置相互作用）計算クラス

This module implements CIS for calculating excited states of molecules.
CIS is the simplest method for excited state calculations, considering
only single excitations from the ground state.
このモジュールは分子の励起状態を計算するCISを実装します。
CISは励起状態計算の最も単純な手法で、基底状態からの単一励起のみを考慮します。
"""
import numpy as np


class Calculator():
  """
  CIS calculator for excited state calculations
  励起状態計算のためのCIS計算器
  """
  def __init__(self, scf_object):
    """
    Initialize CIS calculator with SCF results
    SCF計算結果を使ってCIS計算器を初期化

    Args:
        scf_object: Converged SCF calculation object / 収束したSCF計算オブジェクト
    """
    self.scf = scf_object

  def cis(self):
    """
    Calculate CIS excited states
    CIS励起状態を計算

    This implementation prioritizes clarity over efficiency for educational purposes.
    CIS constructs and diagonalizes the Hamiltonian matrix in the space of
    single excitations from occupied to virtual orbitals.
    この実装は教育目的のため、効率よりも明確さを優先しています。
    CISは占有軌道から仮想軌道への単一励起空間でハミルトニアン行列を
    構築し、対角化します。

    Returns:
        excitation_energies: Array of excitation energies in Hartree / 励起エネルギー配列（Hartree単位）
    """
    # Check for closed-shell restriction
    # 閉殻系の制限を確認
    if self.scf.spin_multiplicity != 1:
      raise NotImplementedError("Current CIS only supports the singlet state.")

    # Set parameters from SCF results
    # SCF結果からパラメータを設定
    # Number of occupied orbitals (doubly occupied in closed-shell)
    # 占有軌道数（閉殻系では二重占有）
    num_occupied_mo = int(self.scf.num_electrons / 2)
    # Number of virtual (unoccupied) orbitals
    # 仮想（非占有）軌道数
    num_virtual_mo = self.scf.mo_coefficients.shape[1] - num_occupied_mo

    # Extract occupied and virtual MO coefficients and energies
    # 占有軌道と仮想軌道のMO係数とエネルギーを抽出
    # MO coefficients are organized as (AO index, MO index)
    # MO係数は（AO添字、MO添字）として整理されています
    occupied_mo_coeff = self.scf.mo_coefficients[:, :num_occupied_mo]
    virtual_mo_coeff = self.scf.mo_coefficients[:, num_occupied_mo:]
    occupied_mo_energies = self.scf.mo_energies[:num_occupied_mo]
    virtual_mo_energies = self.scf.mo_energies[num_occupied_mo:]

    # Dimension of CIS Hamiltonian matrix
    # CISハミルトニアン行列の次元
    # Each single excitation i->a is a basis state
    # 各単一励起 i->a が基底状態です
    dim_cis_hamiltonian = num_occupied_mo * num_virtual_mo

    # Check problem size for current implementation
    # 現在の実装の問題サイズを確認
    if dim_cis_hamiltonian > 1000:
      raise NotImplementedError(
          "The number of atomic orbitals is too much for the current CIS implementation.")

    ### Calculate CIS Hamiltonian matrix in the molecular orbital (MO) basis
    ### 分子軌道（MO）基底でCISハミルトニアン行列を計算
    # Index convention / 添字の規約:
    # i, j: Occupied MOs / 占有MO
    # a, b: Virtual MOs / 仮想MO
    # p, q, r, s: Atomic orbitals (AOs) / 原子軌道（AO）
    # MO coefficients: C(p, i) means AO p in MO i
    # MO係数: C(p, i)はMO iにおけるAO p

    # Calculate orbital energy contributions (diagonal part)
    # 軌道エネルギー寄与を計算（対角部分）
    # The diagonal elements are: H_ia,ia = ε_a - ε_i
    # 対角要素は: H_ia,ia = ε_a - ε_i
    print("Calculating orbital energy contributions...")
    print("軌道エネルギー寄与を計算中...")
    orbital_energy_contributions = np.zeros(
        (dim_cis_hamiltonian, dim_cis_hamiltonian))
    for i in range(num_occupied_mo):
      for j in range(num_occupied_mo):
        for a in range(num_virtual_mo):
          for b in range(num_virtual_mo):
            if i == j and a == b:
              # Map 2D index (i,a) to 1D index
              # 2次元添字(i,a)を1次元添字に変換
              idx_ia = i * num_virtual_mo + a
              idx_jb = j * num_virtual_mo + b
              # Diagonal: energy difference between virtual and occupied
              # 対角: 仮想軌道と占有軌道のエネルギー差
              orbital_energy_contributions[idx_ia, idx_jb] = \
                  virtual_mo_energies[a] - occupied_mo_energies[i]

    # Transform electron repulsion integrals from AO to MO basis
    # 電子反発積分をAO基底からMO基底へ変換
    # AO electron repulsion integrals are obtained from SCF calculation
    # AO電子反発積分はSCF計算から取得されます
    # self.scf.ao_electron_repulsion_integral[p,q,r,s] = (pq|rs)

    num_ao = self.scf.ao_electron_repulsion_integral.shape[0]

    # Efficient transformation for (ia|jb) integrals
    # (ia|jb)積分の効率的な変換
    # (ia|jb) = ∫∫ φ_i(1) φ_a(1) (1/r_12) φ_j(2) φ_b(2) dr_1 dr_2
    print("Transforming AO integrals to MO basis for (ia|jb)...")
    print("AO積分を(ia|jb)のMO基底に変換中...")

    # Use 4-step transformation: (pq|rs) -> (iq|rs) -> (iq|js) -> (ia|js) -> (ia|jb)
    # 4段階変換を使用: (pq|rs) -> (iq|rs) -> (iq|js) -> (ia|js) -> (ia|jb)
    print("  Step 1/4: Transforming first index...")
    print("  ステップ1/4: 最初の添字を変換中...")
    temp1 = np.zeros((num_occupied_mo, num_ao, num_ao, num_ao))
    for i in range(num_occupied_mo):
      for q in range(num_ao):
        for r in range(num_ao):
          for s in range(num_ao):
            temp1[i, q, r, s] = np.dot(occupied_mo_coeff[:, i],
                                       self.scf.ao_electron_repulsion_integral[:, q, r, s])

    print("  Step 2/4: Transforming third index...")
    print("  ステップ2/4: 3番目の添字を変換中...")
    temp2 = np.zeros((num_occupied_mo, num_ao, num_occupied_mo, num_ao))
    for i in range(num_occupied_mo):
      for q in range(num_ao):
        for j in range(num_occupied_mo):
          for s in range(num_ao):
            temp2[i, q, j, s] = np.dot(occupied_mo_coeff[:, j],
                                       temp1[i, q, :, s])

    print("  Step 3/4: Transforming second index...")
    print("  ステップ3/4: 2番目の添字を変換中...")
    temp3 = np.zeros((num_occupied_mo, num_virtual_mo, num_occupied_mo, num_ao))
    for i in range(num_occupied_mo):
      for a in range(num_virtual_mo):
        for j in range(num_occupied_mo):
          for s in range(num_ao):
            temp3[i, a, j, s] = np.dot(virtual_mo_coeff[:, a],
                                       temp2[i, :, j, s])

    print("  Step 4/4: Transforming fourth index...")
    print("  ステップ4/4: 4番目の添字を変換中...")
    mo_integral_iajb = np.zeros((num_occupied_mo, num_virtual_mo,
                                  num_occupied_mo, num_virtual_mo))
    for i in range(num_occupied_mo):
      for a in range(num_virtual_mo):
        for j in range(num_occupied_mo):
          for b in range(num_virtual_mo):
            mo_integral_iajb[i, a, j, b] = np.dot(virtual_mo_coeff[:, b],
                                                  temp3[i, a, j, :])
    print("  Completed (ia|jb) transformation!")
    print("  (ia|jb)変換完了！")

    # Efficient transformation for (ij|ab) integrals
    # (ij|ab)積分の効率的な変換
    # (ij|ab) = ∫∫ φ_i(1) φ_j(1) (1/r_12) φ_a(2) φ_b(2) dr_1 dr_2
    print("Transforming AO integrals to MO basis for (ij|ab)...")
    print("AO積分を(ij|ab)のMO基底に変換中...")

    # Use 4-step transformation: (pq|rs) -> (iq|rs) -> (iq|as) -> (iq|ab) -> (ij|ab)
    # 4段階変換を使用: (pq|rs) -> (iq|rs) -> (iq|as) -> (iq|ab) -> (ij|ab)
    # Note: We can reuse temp1 from above since it's (iq|rs)
    # 注: 上で計算したtemp1を再利用できます（(iq|rs)なので）
    print("  Step 1/4: Reusing previous (iq|rs)...")
    print("  ステップ1/4: 前の(iq|rs)を再利用中...")

    print("  Step 2/4: Transforming third index...")
    print("  ステップ2/4: 3番目の添字を変換中...")
    temp2_2 = np.zeros((num_occupied_mo, num_ao, num_virtual_mo, num_ao))
    for i in range(num_occupied_mo):
      for q in range(num_ao):
        for a in range(num_virtual_mo):
          for s in range(num_ao):
            temp2_2[i, q, a, s] = np.dot(virtual_mo_coeff[:, a],
                                         temp1[i, q, :, s])

    print("  Step 3/4: Transforming fourth index...")
    print("  ステップ3/4: 4番目の添字を変換中...")
    temp3_2 = np.zeros((num_occupied_mo, num_ao, num_virtual_mo, num_virtual_mo))
    for i in range(num_occupied_mo):
      for q in range(num_ao):
        for a in range(num_virtual_mo):
          for b in range(num_virtual_mo):
            temp3_2[i, q, a, b] = np.dot(virtual_mo_coeff[:, b],
                                         temp2_2[i, q, a, :])

    print("  Step 4/4: Transforming second index...")
    print("  ステップ4/4: 2番目の添字を変換中...")
    mo_integral_ijab = np.zeros((num_occupied_mo, num_occupied_mo,
                                  num_virtual_mo, num_virtual_mo))
    for i in range(num_occupied_mo):
      for j in range(num_occupied_mo):
        for a in range(num_virtual_mo):
          for b in range(num_virtual_mo):
            mo_integral_ijab[i, j, a, b] = np.dot(occupied_mo_coeff[:, j],
                                                  temp3_2[i, :, a, b])
    print("  Completed (ij|ab) transformation!")
    print("  (ij|ab)変換完了！")

    # Build CIS Hamiltonian matrix
    # CISハミルトニアン行列を構築
    # H_ia,jb = δ_ij δ_ab (ε_a - ε_i) + 2(ia|jb) - (ij|ab)
    # where δ is Kronecker delta
    # ここでδはクロネッカーのデルタ
    print("Building CIS Hamiltonian matrix...")
    print("CISハミルトニアン行列を構築中...")
    cis_hamiltonian = np.zeros((dim_cis_hamiltonian, dim_cis_hamiltonian))

    # Calculate CIS Hamiltonian matrix elements
    # CISハミルトニアン行列要素を計算
    # The Hamiltonian includes:
    # ハミルトニアンは以下を含みます:
    # 1. Orbital energy differences (diagonal)
    #    軌道エネルギー差（対角）
    # 2. Coulomb terms: 2(ia|jb)
    #    クーロン項: 2(ia|jb)
    # 3. Exchange terms: -(ij|ab)
    #    交換項: -(ij|ab)
    for i in range(num_occupied_mo):
      for j in range(num_occupied_mo):
        for a in range(num_virtual_mo):
          for b in range(num_virtual_mo):
            # Map 2D indices to 1D
            # 2次元添字を1次元に変換
            idx_ia = i * num_virtual_mo + a
            idx_jb = j * num_virtual_mo + b
            # CIS Hamiltonian = orbital energy + Coulomb - Exchange
            # CISハミルトニアン = 軌道エネルギー + クーロン - 交換
            cis_hamiltonian[idx_ia, idx_jb] = \
              orbital_energy_contributions[idx_ia, idx_jb] + \
                2.0 * mo_integral_iajb[i, a, j, b] - mo_integral_ijab[i, j, a, b]

    # Diagonalize the CIS Hamiltonian matrix
    # CISハミルトニアン行列を対角化
    # Eigenvalues are excitation energies
    # 固有値が励起エネルギーです
    # Eigenvectors are CI coefficients for excited states
    # 固有ベクトルが励起状態のCI係数です
    print("Diagonalizing CIS Hamiltonian matrix...")
    print("CISハミルトニアン行列を対角化中...")
    excitation_energies, eigen_vectors = np.linalg.eigh(cis_hamiltonian)

    # Print results / 結果を出力
    print("")
    print("The number of excited states is %s." % len(excitation_energies))
    au_to_ev = 27.21162  # Conversion factor from Hartree to eV / HartreeからeVへの換算係数
    print("CIS excitation energies (eV):")
    print(*excitation_energies * au_to_ev)

    return excitation_energies
