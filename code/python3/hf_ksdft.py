"""
Hartree-Fock and Kohn-Sham DFT calculator class
ハートリー・フォック法およびコーン・シャムDFT計算クラス

This module implements self-consistent field (SCF) calculations
for both Hartree-Fock and Kohn-Sham DFT methods.
このモジュールはハートリー・フォック法とコーン・シャムDFT法の
自己無撞着場(SCF)計算を実装します。
"""
import numpy as np
import interface_psi4 as ipsi4
import ksdft_functional as kdf


class Calculator():
  """
  SCF calculator for Hartree-Fock and Kohn-Sham DFT
  ハートリー・フォック法およびコーン・シャムDFTのためのSCF計算クラス
  """
  def __init__(self, mol_xyz, nuclear_numbers, geom_coordinates,
               basis_set_name, ksdft_functional_name,
               molecular_charge, spin_multiplicity):
    """
    Initialize SCF calculator with molecular information
    分子情報を使ってSCF計算器を初期化

    Args:
        mol_xyz: XYZ format molecular geometry / XYZ形式の分子構造
        nuclear_numbers: Array of atomic numbers / 原子番号の配列
        geom_coordinates: Atomic coordinates / 原子座標
        basis_set_name: Name of basis set / 基底関数セット名
        ksdft_functional_name: DFT functional name (None for HF) / DFT汎関数名（HFの場合None）
        molecular_charge: Molecular charge / 分子の電荷
        spin_multiplicity: Spin multiplicity (2S+1) / スピン多重度（2S+1）
    """
    # Store molecular information
    # 分子情報を保存
    self.mol_xyz = mol_xyz
    self.nuclear_numbers = nuclear_numbers
    self.geom_coordinates = geom_coordinates
    self.basis_set_name = basis_set_name
    self.ksdft_functional_name = ksdft_functional_name
    # Calculate total number of electrons
    # 総電子数を計算
    self.num_electrons = np.sum(nuclear_numbers) - molecular_charge
    self.spin_multiplicity = spin_multiplicity

    # Check if DFT functional is requested
    # DFT汎関数が要求されているか確認
    if self.ksdft_functional_name is None or str(self.ksdft_functional_name).lower() == 'none':
      self.flag_ksdft = False
    elif str(self.ksdft_functional_name).lower() == 'lda':
      self.flag_ksdft = True
    else:
      raise NotImplementedError(
        "Only the LDA exchange functional is implemented.")

    # Note: Unrestricted KS-DFT with LDA is now supported using spin-polarized functionals
    # 注意: LDAを使った非制限KS-DFTがスピン分極汎関数を使用してサポートされるようになりました

  @staticmethod
  def solve_one_electron_problem(orthogonalizer, fock_matrix):
    """
    Solve the one-electron eigenvalue problem F'C' = SC'E
    1電子固有値問題 F'C' = SC'E を解く

    Transform Fock matrix to orthogonalized basis and diagonalize
    Fock行列を直交化基底に変換して対角化

    Args:
        orthogonalizer: Orthogonalization matrix S^(-1/2) / 直交化行列 S^(-1/2)
        fock_matrix: Fock matrix in AO basis / AO基底でのFock行列

    Returns:
        orbital_energies: Orbital energies / 軌道エネルギー
        mo_coefficients: MO coefficients in orthogonalized basis / 直交化基底でのMO係数
    """
    # Transform Fock matrix: F' = S^(-1/2)^T * F * S^(-1/2)
    # Fock行列を変換: F' = S^(-1/2)^T * F * S^(-1/2)
    fock_matrix_in_orthogonalize_basis = np.matmul(
        np.transpose(orthogonalizer.conjugate()), fock_matrix)
    fock_matrix_in_orthogonalize_basis = np.matmul(
        fock_matrix_in_orthogonalize_basis, orthogonalizer)

    # Solve eigenvalue problem: F'C' = C'E
    # 固有値問題を解く: F'C' = C'E
    return np.linalg.eigh(fock_matrix_in_orthogonalize_basis)


  def calc_density_matrix_in_ao_basis(self, mo_coefficients):
    """
    Calculate density matrix from MO coefficients
    MO係数から密度行列を計算

    Args:
        mo_coefficients: Molecular orbital coefficients / 分子軌道係数

    Returns:
        Density matrix in AO basis / AO基底での密度行列
    """
    # Restricted calculation (closed-shell)
    # 制限計算（閉殻系）
    if self.spin_multiplicity == 1:
      # Extract occupied MO coefficients
      # 占有軌道係数を抽出
      occupied_mo_coefficients = mo_coefficients[:, :int(self.num_electrons / 2)]

      # Compute density matrix: P = 2 * C_occ * C_occ^T
      # 密度行列を計算: P = 2 * C_occ * C_occ^T
      # Factor of 2 accounts for spin (alpha and beta electrons)
      # 係数2はスピン（αおよびβ電子）を考慮
      return 2.0 * np.matmul(occupied_mo_coefficients, np.transpose(occupied_mo_coefficients))

    # Unrestricted calculation (open-shell)
    # 非制限計算（開殻系）
    else:
      # Calculate the number of alpha and beta electrons
      # αおよびβ電子の数を計算
      diff_num_alpha_and_beta_electrons = int(self.spin_multiplicity - 1)
      num_base_electrons = int((self.num_electrons - diff_num_alpha_and_beta_electrons) / 2)
      num_alpha_electrons = num_base_electrons + diff_num_alpha_and_beta_electrons
      num_beta_electrons = num_base_electrons

      # Extract alpha and beta occupied MO coefficients
      # αおよびβ占有軌道係数を抽出
      alpha_occupied_mo_coefficients = mo_coefficients[0, :, :num_alpha_electrons]
      beta_occupied_mo_coefficients = mo_coefficients[1, :, :num_beta_electrons]

      # Initialize density matrix array for alpha and beta
      # αおよびβの密度行列配列を初期化
      num_ao = mo_coefficients.shape[1]
      density_matrix = np.zeros((2, num_ao, num_ao))
      # Compute alpha density matrix: P_alpha = C_alpha * C_alpha^T
      # α密度行列を計算: P_alpha = C_alpha * C_alpha^T
      density_matrix[0] = np.matmul(alpha_occupied_mo_coefficients, \
        np.transpose(alpha_occupied_mo_coefficients))
      # Compute beta density matrix: P_beta = C_beta * C_beta^T
      # β密度行列を計算: P_beta = C_beta * C_beta^T
      density_matrix[1] = np.matmul(beta_occupied_mo_coefficients, \
        np.transpose(beta_occupied_mo_coefficients))

      return density_matrix

  @staticmethod
  def calc_nuclei_repulsion_energy(coordinates, charges):
    """
    Calculate nuclear repulsion energy
    核間反発エネルギーを計算

    Computes the classical Coulomb repulsion between nuclei
    原子核間の古典的クーロン反発を計算

    Args:
        coordinates: Nuclear coordinates in Angstrom / 原子核座標（オングストローム単位）
        charges: Nuclear charges (atomic numbers) / 核電荷（原子番号）

    Returns:
        Nuclear repulsion energy in Hartree / 核間反発エネルギー（Hartree単位）
    """
    # Conversion factor from Angstrom to Bohr
    # オングストロームからボーアへの変換係数
    ang_to_bohr = 1 / 0.52917721067
    natoms = len(coordinates)
    ret = 0.0
    # Sum over all unique nuclear pairs
    # 全ての原子核ペアについて和を取る
    for i in range(natoms):
      for j in range(i + 1, natoms):
        # Calculate internuclear distance in Bohr
        # 原子核間距離をボーア単位で計算
        d = np.linalg.norm((coordinates[i] - coordinates[j]) * ang_to_bohr)
        # Add Coulomb repulsion: Z_i * Z_j / r_ij
        # クーロン反発を加算: Z_i * Z_j / r_ij
        ret += charges[i] * charges[j] / d
    return ret


  def scf(self):
    """
    Perform self-consistent field (SCF) calculation
    自己無撞着場(SCF)計算を実行

    Iteratively solves the Hartree-Fock or Kohn-Sham equations
    until self-consistency is achieved.
    ハートリー・フォックまたはコーン・シャム方程式を
    自己無撞着性が達成されるまで反復的に解きます。
    """
    # Maximum number of SCF iterations
    # SCF反復の最大回数
    num_max_scf_iter = 1000

    ### Preprocessing for SCF
    ### SCFの前処理

    # Initialize Psi4 interface for AO integrals
    # AO積分のためのPsi4インターフェースを初期化
    proc_ao_integral = ipsi4.Psi4Interface(
      self.mol_xyz, self.nuclear_numbers,
      self.geom_coordinates, self.basis_set_name,
      self.ksdft_functional_name)

    # Compute all requisite analytical AO integrals
    # 必要な解析的AO積分を全て計算
    # Kinetic energy integrals T
    # 運動エネルギー積分 T
    ao_kinetic_integral = proc_ao_integral.ao_kinetic_integral()
    # Nuclear attraction integrals V
    # 核引力積分 V
    ao_nuclear_attraction_integral = proc_ao_integral.ao_nuclear_attraction_integral()
    # Electron repulsion integrals (ERIs)
    # 電子反発積分 (ERI)
    ao_electron_repulsion_integral = proc_ao_integral.ao_electron_repulsion_integral()
    # Overlap integrals S
    # 重なり積分 S
    ao_overlap_integral = proc_ao_integral.ao_overlap_integral()

    # For KS-DFT: generate numerical integration grids
    # KS-DFTの場合: 数値積分グリッドを生成
    if self.flag_ksdft:
      # In practice, grids should be generated on-the-fly to save memory
      # 実際には、メモリを節約するためにグリッドをその場で生成すべき
      num_ao = len(ao_overlap_integral)
      if num_ao > 120:
        raise NotImplementedError(
          "The number of AOs is too large for the current implementation.")
      # Generate 3D grid points and integration weights
      # 3Dグリッド点と積分重みを生成
      real_space_grids, weights_grids = \
        proc_ao_integral.generate_numerical_integration_grids_and_weights()
      num_grids = len(real_space_grids)
      ao_values_at_grids = np.zeros((num_grids, num_ao))
      # Calculate AO values at each grid point
      # 各グリッド点でのAO値を計算
      ao_values_at_grids = proc_ao_integral.generate_ao_values_at_grids(real_space_grids)

    # Compute core Hamiltonian H_core = T + V
    # コアハミルトニアンを計算 H_core = T + V
    core_hamiltonian = ao_kinetic_integral + ao_nuclear_attraction_integral

    # Prepare the orthogonalizer S^(-1/2) for solving the eigenvalue problem
    # 固有値問題を解くための直交化行列 S^(-1/2) を準備
    overlap_eigen_value, overlap_eigen_function = np.linalg.eigh(
      ao_overlap_integral)

    # Get the number of atomic orbitals (basis functions)
    # 原子軌道（基底関数）の数を取得
    num_ao = len(overlap_eigen_value)
    # Construct S^(-1/2) = U * s^(-1/2) * U^T
    # S^(-1/2) を構築: S^(-1/2) = U * s^(-1/2) * U^T
    # where U are eigenvectors and s are eigenvalues of S
    # Uは固有ベクトル、sはSの固有値
    for i in range(len(overlap_eigen_value)):
      overlap_eigen_value[i] = overlap_eigen_value[i] ** (-0.5)
    half_inverse_overlap_eigen_value = np.diag(overlap_eigen_value)
    orthogonalizer = np.matmul(
      overlap_eigen_function, half_inverse_overlap_eigen_value)
    orthogonalizer = np.matmul(orthogonalizer, np.transpose(
      overlap_eigen_function.conjugate()))

    # Calculate initial guess by solving H_core in orthogonalized basis
    # 直交化基底でH_coreを解いて初期推測を計算
    # This neglects two-electron terms in the first iteration
    # 最初の反復では2電子項を無視
    orbital_energies, temp_mo_coefficients = Calculator.solve_one_electron_problem(
      orthogonalizer, core_hamiltonian)

    # Transform MO coefficients back to original AO basis
    # MO係数を元のAO基底に変換
    if self.spin_multiplicity == 1:
      mo_coefficients = np.matmul(orthogonalizer, temp_mo_coefficients)
    else:
      mo_coefficients = np.zeros((2, num_ao, num_ao))
      mo_coefficients[0] = np.matmul(orthogonalizer, temp_mo_coefficients)
      mo_coefficients[1] = np.matmul(orthogonalizer, temp_mo_coefficients)

    # Calculate initial density matrix in AO basis
    # AO基底での初期密度行列を計算
    density_matrix_in_ao_basis = Calculator.calc_density_matrix_in_ao_basis(
      self, mo_coefficients)


    # Calculate nuclear repulsion energy (constant throughout SCF)
    # 核間反発エネルギーを計算（SCF中は一定）
    nuclear_repulsion_energy = Calculator.calc_nuclei_repulsion_energy(
      self.geom_coordinates, self.nuclear_numbers)

    # For KS-DFT: calculate initial electron density at grid points
    # KS-DFTの場合: グリッド点での初期電子密度を計算
    if self.flag_ksdft:
      if self.spin_multiplicity == 1:
        # RKS: Calculate total electron density: rho(r) = sum_pq P_pq * phi_p(r) * phi_q(r)
        # RKS: 総電子密度を計算: rho(r) = sum_pq P_pq * phi_p(r) * phi_q(r)
        electron_density_at_grids = np.zeros(num_grids)
        # rho[n] = sum_pq P[p,q] * phi[n,p] * phi[n,q]
        # First compute phi * P, then element-wise multiply with phi and sum
        # まず phi * P を計算し、次に phi と要素ごとに掛けて和を取る
        for n in range(num_grids):
          temp = np.matmul(ao_values_at_grids[n, :], density_matrix_in_ao_basis)
          electron_density_at_grids[n] = np.dot(temp, ao_values_at_grids[n, :])
      else:
        # UKS: Calculate alpha and beta spin densities separately
        # UKS: αとβスピン密度を別々に計算
        alpha_density_at_grids = np.zeros(num_grids)
        beta_density_at_grids = np.zeros(num_grids)
        for n in range(num_grids):
          # Alpha spin density / αスピン密度
          temp_alpha = np.matmul(ao_values_at_grids[n, :], density_matrix_in_ao_basis[0])
          alpha_density_at_grids[n] = np.dot(temp_alpha, ao_values_at_grids[n, :])
          # Beta spin density / βスピン密度
          temp_beta = np.matmul(ao_values_at_grids[n, :], density_matrix_in_ao_basis[1])
          beta_density_at_grids[n] = np.dot(temp_beta, ao_values_at_grids[n, :])

    ### Start SCF iterations
    ### SCF反復を開始
    for idx_scf in range(num_max_scf_iter):

      # For KS-DFT: compute exchange-correlation potential and energy
      # KS-DFTの場合: 交換相関ポテンシャルとエネルギーを計算
      if self.flag_ksdft:
        if self.spin_multiplicity == 1:
          # RKS: Compute V_xc using spin-unpolarized LDA functional
          # RKS: スピン非分極LDA汎関数を使ってV_xcを計算
          exchange_correlation_potential = kdf.lda_potential(electron_density_at_grids)

          # Integrate V_xc into Fock matrix: V_xc[p,q] = sum_n w_n * phi_p(n) * V_xc(n) * phi_q(n)
          # V_xcをFock行列に積分: V_xc[p,q] = sum_n w_n * phi_p(n) * V_xc(n) * phi_q(n)
          exchange_correlation_potential_in_Fock_matrix = np.zeros((num_ao, num_ao))
          for p in range(num_ao):
            for q in range(num_ao):
              exchange_correlation_potential_in_Fock_matrix[p, q] = np.sum(
                weights_grids * ao_values_at_grids[:, p] * \
                exchange_correlation_potential * ao_values_at_grids[:, q])
        else:
          # UKS: Compute V_xc using spin-polarized LDA functional
          # UKS: スピン分極LDA汎関数を使ってV_xcを計算
          V_xc_alpha, V_xc_beta = kdf.lda_potential_spinpol(alpha_density_at_grids, beta_density_at_grids)

          # Integrate V_xc into alpha and beta Fock matrices
          # V_xcをαおよびβFock行列に積分
          alpha_xc_matrix = np.zeros((num_ao, num_ao))
          beta_xc_matrix = np.zeros((num_ao, num_ao))
          for p in range(num_ao):
            for q in range(num_ao):
              alpha_xc_matrix[p, q] = np.sum(
                weights_grids * ao_values_at_grids[:, p] * V_xc_alpha * ao_values_at_grids[:, q])
              beta_xc_matrix[p, q] = np.sum(
                weights_grids * ao_values_at_grids[:, p] * V_xc_beta * ao_values_at_grids[:, q])

      # Build Fock matrix in AO basis
      # AO基底でFock行列を構築
      if self.spin_multiplicity == 1:
        # Compute Coulomb term J: J[p,q] = sum_rs (pq|rs) * P[r,s]
        # クーロン項Jを計算: J[p,q] = sum_rs (pq|rs) * P[r,s]
        electron_repulsion_in_Fock_matrix = np.zeros((num_ao, num_ao))
        for p in range(num_ao):
          for q in range(num_ao):
            electron_repulsion_in_Fock_matrix[p, q] = np.sum(
              ao_electron_repulsion_integral[p, q, :, :] * density_matrix_in_ao_basis)
        if not self.flag_ksdft:
          # Compute exchange term K: K[p,q] = sum_rs (pr|qs) * P[r,s]
          # 交換項Kを計算: K[p,q] = sum_rs (pr|qs) * P[r,s]
          exchange_in_Fock_matrix = np.zeros((num_ao, num_ao))
          for p in range(num_ao):
            for q in range(num_ao):
              exchange_in_Fock_matrix[p, q] = np.sum(
                ao_electron_repulsion_integral[p, :, q, :] * density_matrix_in_ao_basis)
          # Build Hartree-Fock Fock matrix: F = H_core + J - 0.5*K
          # ハートリー・フォックFock行列を構築: F = H_core + J - 0.5*K
          fock_matrix = core_hamiltonian + electron_repulsion_in_Fock_matrix \
            - 0.5 * exchange_in_Fock_matrix
        else:
          # Build KS-DFT Fock matrix: F = H_core + J + V_xc
          # KS-DFT Fock行列を構築: F = H_core + J + V_xc
          fock_matrix = core_hamiltonian + electron_repulsion_in_Fock_matrix \
            + exchange_correlation_potential_in_Fock_matrix
      else:
        # Unrestricted calculation: build separate alpha and beta Fock matrices
        # 非制限計算: αおよびβFock行列を別々に構築
        # Total density for Coulomb term
        # クーロン項のための全密度
        total_density = density_matrix_in_ao_basis[0] + density_matrix_in_ao_basis[1]
        electron_repulsion_in_Fock_matrix = np.zeros((num_ao, num_ao))
        for p in range(num_ao):
          for q in range(num_ao):
            electron_repulsion_in_Fock_matrix[p, q] = np.sum(
              ao_electron_repulsion_integral[p, q, :, :] * total_density)

        if not self.flag_ksdft:
          # UHF: Build Fock matrices with exchange terms
          # UHF: 交換項を含むFock行列を構築
          # Alpha-spin exchange term
          # αスピン交換項
          alpha_exchange_in_Fock_matrix = np.zeros((num_ao, num_ao))
          for p in range(num_ao):
            for q in range(num_ao):
              alpha_exchange_in_Fock_matrix[p, q] = np.sum(
                ao_electron_repulsion_integral[p, :, q, :] * density_matrix_in_ao_basis[0])
          alpha_fock_matrix = core_hamiltonian + electron_repulsion_in_Fock_matrix \
            - alpha_exchange_in_Fock_matrix
          # Beta-spin exchange term
          # βスピン交換項
          beta_exchange_in_Fock_matrix = np.zeros((num_ao, num_ao))
          for p in range(num_ao):
            for q in range(num_ao):
              beta_exchange_in_Fock_matrix[p, q] = np.sum(
                ao_electron_repulsion_integral[p, :, q, :] * density_matrix_in_ao_basis[1])
          beta_fock_matrix = core_hamiltonian + electron_repulsion_in_Fock_matrix \
            - beta_exchange_in_Fock_matrix
        else:
          # UKS: Build Fock matrices with XC potentials
          # UKS: XCポテンシャルを含むFock行列を構築
          # Alpha Fock matrix: F_α = H_core + J + V_xc^α
          # αFock行列: F_α = H_core + J + V_xc^α
          alpha_fock_matrix = core_hamiltonian + electron_repulsion_in_Fock_matrix \
            + alpha_xc_matrix
          # Beta Fock matrix: F_β = H_core + J + V_xc^β
          # βFock行列: F_β = H_core + J + V_xc^β
          beta_fock_matrix = core_hamiltonian + electron_repulsion_in_Fock_matrix \
            + beta_xc_matrix

      # Calculate the electronic energy (without nuclear repulsion)
      # 電子エネルギーを計算（核間反発を除く）
      if self.spin_multiplicity == 1:
        if not self.flag_ksdft:
          # HF energy: E_elec = 0.5 * Tr[P * (H_core + F)]
          # HFエネルギー: E_elec = 0.5 * Tr[P * (H_core + F)]
          electronic_energy = 0.5 * np.sum(
            density_matrix_in_ao_basis * (core_hamiltonian + fock_matrix))
        else:
          # KS-DFT energy includes exchange-correlation contribution
          # KS-DFTエネルギーは交換相関寄与を含む
          exchange_correlation_energy = kdf.lda_energy(
            electron_density_at_grids, weights_grids)
          electronic_energy = 0.5 * np.sum(
            density_matrix_in_ao_basis * (2.0 * core_hamiltonian + electron_repulsion_in_Fock_matrix))
          electronic_energy += exchange_correlation_energy
      else:
        # Unrestricted energy calculation
        # 非制限エネルギー計算
        if not self.flag_ksdft:
          # UHF energy: E_elec = 0.5 * sum_σ Tr[P_σ * (H_core + F_σ)]
          # UHFエネルギー: E_elec = 0.5 * sum_σ Tr[P_σ * (H_core + F_σ)]
          electronic_energy = 0.5 * np.sum(
            (density_matrix_in_ao_basis[0] + density_matrix_in_ao_basis[1]) * core_hamiltonian)
          electronic_energy += 0.5 * np.sum(
            density_matrix_in_ao_basis[0] * alpha_fock_matrix)
          electronic_energy += 0.5 * np.sum(
            density_matrix_in_ao_basis[1] * beta_fock_matrix)
        else:
          # UKS energy includes spin-polarized exchange-correlation contribution
          # UKSエネルギーはスピン分極交換相関寄与を含む
          exchange_correlation_energy = kdf.lda_energy_spinpol(
            alpha_density_at_grids, beta_density_at_grids, weights_grids)
          # E = 0.5 * sum_σ Tr[P_σ * (2*H_core + J)] + E_xc
          electronic_energy = 0.5 * np.sum(
            (density_matrix_in_ao_basis[0] + density_matrix_in_ao_basis[1]) * \
            (2.0 * core_hamiltonian + electron_repulsion_in_Fock_matrix))
          electronic_energy += exchange_correlation_energy

      # Total energy = electronic energy + nuclear repulsion
      # 全エネルギー = 電子エネルギー + 核間反発
      total_energy = electronic_energy + nuclear_repulsion_energy
      print("SCF step %s: " % str(idx_scf + 1), total_energy, "Hartree")
      # Check for convergence
      # 収束を確認
      if idx_scf > 0:
        if abs(old_total_energy - total_energy) < 1.e-9:
          print("SCF converged!")
          print("SCFが収束しました!")
          break
      old_total_energy = total_energy

      # Solve the one-electron eigenvalue problem to get new MOs
      # 新しいMOを得るために1電子固有値問題を解く
      if self.spin_multiplicity == 1:
        orbital_energies, mo_coefficients = Calculator.solve_one_electron_problem(
          orthogonalizer, fock_matrix)
      else:
        alpha_orbital_energies, alpha_mo_coefficients = Calculator.solve_one_electron_problem(
          orthogonalizer, alpha_fock_matrix)
        beta_orbital_energies, beta_mo_coefficients = Calculator.solve_one_electron_problem(
          orthogonalizer, beta_fock_matrix)

      # Transform MO coefficients back to original AO basis
      # MO係数を元のAO基底に変換
      if self.spin_multiplicity == 1:
        mo_coefficients = np.matmul(orthogonalizer, mo_coefficients)
      else:
        mo_coefficients = np.zeros((2, alpha_fock_matrix.shape[0],
                                    alpha_fock_matrix.shape[1]))
        mo_coefficients[0] = np.matmul(orthogonalizer, alpha_mo_coefficients)
        mo_coefficients[1] = np.matmul(orthogonalizer, beta_mo_coefficients)

      # Update density matrix for next iteration
      # 次の反復のために密度行列を更新
      density_matrix_in_ao_basis = Calculator.calc_density_matrix_in_ao_basis(
        self, mo_coefficients)

      # Update electron density at grids for KS-DFT
      # KS-DFTの場合、グリッドでの電子密度を更新
      if self.flag_ksdft:
        if self.spin_multiplicity == 1:
          # RKS: Update total electron density
          # RKS: 総電子密度を更新
          for n in range(num_grids):
            temp = np.matmul(ao_values_at_grids[n, :], density_matrix_in_ao_basis)
            electron_density_at_grids[n] = np.dot(temp, ao_values_at_grids[n, :])
        else:
          # UKS: Update alpha and beta spin densities separately
          # UKS: αとβスピン密度を別々に更新
          for n in range(num_grids):
            # Alpha spin density / αスピン密度
            temp_alpha = np.matmul(ao_values_at_grids[n, :], density_matrix_in_ao_basis[0])
            alpha_density_at_grids[n] = np.dot(temp_alpha, ao_values_at_grids[n, :])
            # Beta spin density / βスピン密度
            temp_beta = np.matmul(ao_values_at_grids[n, :], density_matrix_in_ao_basis[1])
            beta_density_at_grids[n] = np.dot(temp_beta, ao_values_at_grids[n, :])

    ### Save SCF results for post-Hartree-Fock calculations
    ### ポストハートリー・フォック計算のためにSCF結果を保存
    self.density_matrix_in_ao_basis = density_matrix_in_ao_basis
    self.num_ao = num_ao
    self.mo_energies = orbital_energies
    self.mo_coefficients = mo_coefficients
    self.scf_energy = total_energy
    self.ao_electron_repulsion_integral = ao_electron_repulsion_integral

    self.basis_set_object = proc_ao_integral.psi4_basis_set


    def calc_mulliken_atomic_charges(density_matrix_in_ao_basis, ao_overlap_integral):
      """
      Calculate and print Mulliken atomic charges
      Mulliken原子電荷を計算して出力
      """
      # Get which atom each basis function belongs to
      # 各基底関数がどの原子に属するかを取得
      ao_atomic_affiliation = proc_ao_integral.get_basis_atomic_affiliation()
      if self.spin_multiplicity == 1:
        # Compute charge matrix: Q = P * S
        # 電荷行列を計算: Q = P * S
        charge_matrix = np.matmul(density_matrix_in_ao_basis, ao_overlap_integral)
      else:
        # For unrestricted: use total density for charge
        # 非制限の場合: 電荷には全密度を使用
        charge_matrix = np.matmul(density_matrix_in_ao_basis[0] + density_matrix_in_ao_basis[1],
                                  ao_overlap_integral)
        # Spin density difference for spin charges
        # スピン電荷にはスピン密度差を使用
        spin_diff_charge_matrix = np.matmul(density_matrix_in_ao_basis[0] - \
                                  density_matrix_in_ao_basis[1],
                                  ao_overlap_integral)

      # Calculate Mulliken charges for each atom
      # 各原子のMulliken電荷を計算
      num_atom = len(self.nuclear_numbers)
      mulliken_atomic_charges = np.zeros(num_atom)
      # Sum contributions from all basis functions on each atom
      # 各原子上の全基底関数からの寄与を合計
      for i in range(len(ao_atomic_affiliation)):
        mulliken_atomic_charges[ao_atomic_affiliation[i]] += -charge_matrix[i, i]
      # Add nuclear charges to get net atomic charges
      # 核電荷を加えて正味の原子電荷を取得
      mulliken_atomic_charges += self.nuclear_numbers

      if self.spin_multiplicity != 1:
        # Calculate spin charges for unrestricted calculation
        # 非制限計算のスピン電荷を計算
        spin_diff_mulliken_atomic_charges = np.zeros(num_atom)
        for i in range(len(ao_atomic_affiliation)):
          spin_diff_mulliken_atomic_charges[ao_atomic_affiliation[i]
                                            ] += spin_diff_charge_matrix[i, i]

      # Print Mulliken charges
      # Mulliken電荷を出力
      print("Mulliken atomic charges (|e|):", *mulliken_atomic_charges)
      if self.spin_multiplicity != 1:
        print("Mulliken atomic spin charges (|e|):", *spin_diff_mulliken_atomic_charges)

    # Call Mulliken charge calculation
    # Mulliken電荷計算を呼び出す
    calc_mulliken_atomic_charges(density_matrix_in_ao_basis, ao_overlap_integral)
