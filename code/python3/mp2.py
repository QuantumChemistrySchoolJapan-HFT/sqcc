"""
MP2 (Second-order Møller-Plesset Perturbation Theory) calculator class
MP2（2次Møller-Plesset摂動論）計算クラス

This module implements MP2 correlation energy calculations for closed-shell systems.
MP2 is a post-Hartree-Fock method that includes electron correlation effects
through second-order perturbation theory.
このモジュールは閉殻系のMP2相関エネルギー計算を実装します。
MP2はハートリー・フォック法の後処理手法で、2次摂動論を通じて
電子相関効果を取り入れます。
"""
import numpy as np


class Calculator():
  """
  MP2 calculator for post-HF correlation energy
  ハートリー・フォック後の相関エネルギーを計算するMP2計算器
  """
  def __init__(self, scf_object):
    """
    Initialize MP2 calculator with SCF results
    SCF計算結果を使ってMP2計算器を初期化

    Args:
        scf_object: Converged SCF calculation object / 収束したSCF計算オブジェクト
    """
    self.scf = scf_object

  def mp2(self):
    """
    Calculate MP2 correlation energy
    MP2相関エネルギーを計算

    This implementation prioritizes clarity over efficiency for educational purposes.
    The algorithm explicitly shows all steps in MP2 theory.
    この実装は教育目的のため、効率よりも明確さを優先しています。
    アルゴリズムはMP2理論のすべてのステップを明示的に示します。

    Returns:
        mp2_energy: Total MP2 energy in Hartree / 全MP2エネルギー（Hartree単位）
    """
    # Check for closed-shell restriction
    # 閉殻系の制限を確認
    if self.scf.spin_multiplicity != 1:
      raise NotImplementedError("Current MP2 only supports the closed-shell singlet state.")

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

    # Dimension of MP2 problem
    # MP2問題の次元
    dim_mp2 = num_occupied_mo * num_virtual_mo

    # Check problem size for current implementation
    # 現在の実装の問題サイズを確認
    if dim_mp2 > 1000:
      raise NotImplementedError(
          "The number of atomic orbitals is too much for the current MP2 implementation.")

    ### Calculate MP2 energy in the molecular orbital (MO) basis
    ### 分子軌道（MO）基底でMP2エネルギーを計算
    # Index convention / 添字の規約:
    # i, j: Occupied MOs / 占有MO
    # a, b: Virtual MOs / 仮想MO
    # p, q, r, s: Atomic orbitals (AOs) / 原子軌道（AO）
    # MO coefficients: C(p, i) means AO p in MO i
    # MO係数: C(p, i)はMO iにおけるAO p

    # Calculate orbital energy denominators: ε_i + ε_j - ε_a - ε_b
    # 軌道エネルギー分母を計算: ε_i + ε_j - ε_a - ε_b
    # This appears in the MP2 formula
    # これはMP2の式に現れます
    orbital_energy_contributions = np.zeros(
        (num_occupied_mo, num_occupied_mo, num_virtual_mo, num_virtual_mo))
    for i in range(num_occupied_mo):
      for j in range(num_occupied_mo):
        for a in range(num_virtual_mo):
          for b in range(num_virtual_mo):
            # Denominator: ε_i + ε_j - ε_a - ε_b
            # 分母: ε_i + ε_j - ε_a - ε_b
            orbital_energy_contributions[i, j, a, b] = \
              occupied_mo_energies[i] + occupied_mo_energies[j] \
                - virtual_mo_energies[a] - virtual_mo_energies[b]

    # Transform electron repulsion integrals from AO to MO basis
    # 電子反発積分をAO基底からMO基底へ変換
    # AO electron repulsion integrals are obtained from SCF calculation
    # AO電子反発積分はSCF計算から取得されます
    # self.scf.ao_electron_repulsion_integral[p,q,r,s] = (pq|rs)

    # Efficient 4-index transformation using intermediate arrays
    # 中間配列を使った効率的な4添字変換
    # This reduces complexity from O(N^8) to O(N^5)
    # これにより計算量がO(N^8)からO(N^5)に削減されます
    # Step-by-step transformation: (pq|rs) -> (iq|rs) -> (iq|js) -> (ia|js) -> (ia|jb)
    # 段階的な変換: (pq|rs) -> (iq|rs) -> (iq|js) -> (ia|js) -> (ia|jb)

    num_ao = self.scf.ao_electron_repulsion_integral.shape[0]

    print("Transforming AO integrals to MO basis (efficient 4-step method)...")
    print("AO積分をMO基底に変換中（効率的な4段階法）...")

    # Step 1: (pq|rs) -> (iq|rs) by transforming first index p to occupied i
    # ステップ1: 最初の添字pを占有軌道iに変換 (pq|rs) -> (iq|rs)
    print("  Step 1/4: Transforming first index...")
    print("  ステップ1/4: 最初の添字を変換中...")
    temp1 = np.zeros((num_occupied_mo, num_ao, num_ao, num_ao))
    for i in range(num_occupied_mo):
      for q in range(num_ao):
        for r in range(num_ao):
          for s in range(num_ao):
            # temp1[i,q,r,s] = Σ_p C[p,i] * (pq|rs)
            temp1[i, q, r, s] = np.dot(occupied_mo_coeff[:, i],
                                       self.scf.ao_electron_repulsion_integral[:, q, r, s])

    # Step 2: (iq|rs) -> (iq|js) by transforming third index r to occupied j
    # ステップ2: 3番目の添字rを占有軌道jに変換 (iq|rs) -> (iq|js)
    print("  Step 2/4: Transforming third index...")
    print("  ステップ2/4: 3番目の添字を変換中...")
    temp2 = np.zeros((num_occupied_mo, num_ao, num_occupied_mo, num_ao))
    for i in range(num_occupied_mo):
      for q in range(num_ao):
        for j in range(num_occupied_mo):
          for s in range(num_ao):
            # temp2[i,q,j,s] = Σ_r C[r,j] * temp1[i,q,r,s]
            temp2[i, q, j, s] = np.dot(occupied_mo_coeff[:, j],
                                       temp1[i, q, :, s])

    # Step 3: (iq|js) -> (ia|js) by transforming second index q to virtual a
    # ステップ3: 2番目の添字qを仮想軌道aに変換 (iq|js) -> (ia|js)
    print("  Step 3/4: Transforming second index...")
    print("  ステップ3/4: 2番目の添字を変換中...")
    temp3 = np.zeros((num_occupied_mo, num_virtual_mo, num_occupied_mo, num_ao))
    for i in range(num_occupied_mo):
      for a in range(num_virtual_mo):
        for j in range(num_occupied_mo):
          for s in range(num_ao):
            # temp3[i,a,j,s] = Σ_q C[q,a] * temp2[i,q,j,s]
            temp3[i, a, j, s] = np.dot(virtual_mo_coeff[:, a],
                                       temp2[i, :, j, s])

    # Step 4: (ia|js) -> (ia|jb) by transforming fourth index s to virtual b
    # ステップ4: 4番目の添字sを仮想軌道bに変換 (ia|js) -> (ia|jb)
    print("  Step 4/4: Transforming fourth index...")
    print("  ステップ4/4: 4番目の添字を変換中...")
    mo_integral_iajb = np.zeros((num_occupied_mo, num_virtual_mo,
                                  num_occupied_mo, num_virtual_mo))
    for i in range(num_occupied_mo):
      for a in range(num_virtual_mo):
        for j in range(num_occupied_mo):
          for b in range(num_virtual_mo):
            # mo_integral_iajb[i,a,j,b] = Σ_s C[s,b] * temp3[i,a,j,s]
            mo_integral_iajb[i, a, j, b] = np.dot(virtual_mo_coeff[:, b],
                                                  temp3[i, a, j, :])

    print("  Completed (ia|jb) transformation!")
    print("  (ia|jb)変換完了！")

    # Calculate (ib|ja) integrals using the same efficient method
    # 同じ効率的な方法で(ib|ja)積分を計算
    # (ib|ja) = ∫∫ φ_i(1) φ_b(1) (1/r_12) φ_j(2) φ_a(2) dr_1 dr_2
    print("Transforming AO integrals to MO basis for (ib|ja)...")
    print("AO積分を(ib|ja)のMO基底に変換中...")

    # Step 1: (pq|rs) -> (iq|rs)
    # ステップ1: (pq|rs) -> (iq|rs)
    print("  Step 1/4: Transforming first index...")
    print("  ステップ1/4: 最初の添字を変換中...")
    temp1_2 = np.zeros((num_occupied_mo, num_ao, num_ao, num_ao))
    for i in range(num_occupied_mo):
      for q in range(num_ao):
        for r in range(num_ao):
          for s in range(num_ao):
            temp1_2[i, q, r, s] = np.dot(occupied_mo_coeff[:, i],
                                         self.scf.ao_electron_repulsion_integral[:, q, r, s])

    # Step 2: (iq|rs) -> (iq|js)
    # ステップ2: (iq|rs) -> (iq|js)
    print("  Step 2/4: Transforming third index...")
    print("  ステップ2/4: 3番目の添字を変換中...")
    temp2_2 = np.zeros((num_occupied_mo, num_ao, num_occupied_mo, num_ao))
    for i in range(num_occupied_mo):
      for q in range(num_ao):
        for j in range(num_occupied_mo):
          for s in range(num_ao):
            temp2_2[i, q, j, s] = np.dot(occupied_mo_coeff[:, j],
                                         temp1_2[i, q, :, s])

    # Step 3: (iq|js) -> (ib|js) by transforming second index q to virtual b
    # ステップ3: 2番目の添字qを仮想軌道bに変換 (iq|js) -> (ib|js)
    print("  Step 3/4: Transforming second index...")
    print("  ステップ3/4: 2番目の添字を変換中...")
    temp3_2 = np.zeros((num_occupied_mo, num_virtual_mo, num_occupied_mo, num_ao))
    for i in range(num_occupied_mo):
      for b in range(num_virtual_mo):
        for j in range(num_occupied_mo):
          for s in range(num_ao):
            temp3_2[i, b, j, s] = np.dot(virtual_mo_coeff[:, b],
                                         temp2_2[i, :, j, s])

    # Step 4: (ib|js) -> (ib|ja) by transforming fourth index s to virtual a
    # ステップ4: 4番目の添字sを仮想軌道aに変換 (ib|js) -> (ib|ja)
    print("  Step 4/4: Transforming fourth index...")
    print("  ステップ4/4: 4番目の添字を変換中...")
    mo_integral_ibja = np.zeros((num_occupied_mo, num_virtual_mo,
                                  num_occupied_mo, num_virtual_mo))
    for i in range(num_occupied_mo):
      for b in range(num_virtual_mo):
        for j in range(num_occupied_mo):
          for a in range(num_virtual_mo):
            mo_integral_ibja[i, b, j, a] = np.dot(virtual_mo_coeff[:, a],
                                                  temp3_2[i, b, j, :])

    print("  Completed (ib|ja) transformation!")
    print("  (ib|ja)変換完了！")

    # Calculate MP2 correlation energy using the MP2 formula
    # MP2の式を使って相関エネルギーを計算
    # E_corr^MP2 = Σ_ijab [2(ia|jb)² - (ia|jb)(ib|ja)] / (ε_i + ε_j - ε_a - ε_b)
    # The factor 2 comes from spin integration
    # 係数2はスピン積分から生じます
    print("Calculating MP2 correlation energy...")
    print("MP2相関エネルギーを計算中...")
    mp2_correlation_energy = 0.0
    for i in range(num_occupied_mo):
      for j in range(num_occupied_mo):
        for a in range(num_virtual_mo):
          for b in range(num_virtual_mo):
            # Direct term: 2 * (ia|jb)²
            # 直接項: 2 * (ia|jb)²
            direct_term = 2.0 * mo_integral_iajb[i, a, j, b] * mo_integral_iajb[i, a, j, b]
            # Exchange term: -(ia|jb)(ib|ja)
            # 交換項: -(ia|jb)(ib|ja)
            exchange_term = -mo_integral_iajb[i, a, j, b] * mo_integral_ibja[i, b, j, a]
            # Add contribution divided by energy denominator
            # エネルギー分母で割った寄与を加える
            mp2_correlation_energy += (direct_term + exchange_term) / \
                                       orbital_energy_contributions[i, j, a, b]

    # Calculate total MP2 energy = SCF energy + correlation energy
    # 全MP2エネルギーを計算 = SCFエネルギー + 相関エネルギー
    mp2_energy = self.scf.scf_energy + mp2_correlation_energy

    # Print results / 結果を出力
    print("")
    print("MP2 energy (Hartree):", mp2_energy)
    print("MP2 correlation energy (Hartree):", mp2_correlation_energy)

    return mp2_energy
