"""
Psi4 interface module for quantum chemistry calculations
Psi4インターフェースモジュール（量子化学計算用）

This module provides wrapper functions and classes to interact with Psi4
quantum chemistry software for basis set operations and integral calculations.
このモジュールはPsi4量子化学ソフトウェアと連携し、
基底関数操作と積分計算を行うラッパー関数とクラスを提供します。
"""

from typing import Tuple, Optional
import numpy as np
import psi4
from basis_set_exchange import lut


def get_psi4_version() -> str:
    """
    Get Psi4 version string
    Psi4のバージョン文字列を取得

    Returns:
        Version string / バージョン文字列
    """
    return psi4.__version__


def generate_psi4_geometry(
    nuclear_numbers: np.ndarray,
    geom_coordinates: np.ndarray
) -> list[str]:
    """
    Convert atomic numbers to element symbols for Psi4
    原子番号をPsi4用に元素記号に変換

    Args:
        nuclear_numbers: Array of atomic numbers / 原子番号の配列
        geom_coordinates: Atomic coordinates (Nx3 array) / 原子座標 (Nx3配列)

    Returns:
        List of element symbols / 元素記号のリスト
    """
    # Initialize list to store element symbols
    # 元素記号を格納するリストを初期化
    symbol_list = []
    # Convert each atomic number to element symbol
    # 各原子番号を元素記号に変換
    for atomic_number in nuclear_numbers:
        element_symbol = lut.element_sym_from_Z(atomic_number)
        symbol_list.append(element_symbol)
    return symbol_list


class Psi4Interface:
    """
    Interface class to Psi4 quantum chemistry software
    Psi4量子化学ソフトウェアへのインターフェースクラス

    This class provides methods to compute atomic orbital integrals,
    generate numerical integration grids, and access basis set information.
    このクラスは原子軌道積分の計算、数値積分グリッドの生成、
    基底関数セット情報へのアクセスを行うメソッドを提供します。
    """

    def __init__(
        self,
        mol_xyz: str,
        nuclear_numbers: np.ndarray,
        geom_coordinates: np.ndarray,
        basis_set_name: str,
        ksdft_functional_name: Optional[str] = None
    ):
        """
        Initialize Psi4 interface with molecular information
        分子情報を使ってPsi4インターフェースを初期化

        Args:
            mol_xyz: XYZ format molecular geometry string / XYZ形式の分子構造文字列
            nuclear_numbers: Array of atomic numbers / 原子番号の配列
            geom_coordinates: Atomic coordinates (Nx3) / 原子座標 (Nx3)
            basis_set_name: Name of basis set (e.g., 'sto-3g') / 基底関数セット名（例: 'sto-3g'）
            ksdft_functional_name: DFT functional name (None for HF) / DFT汎関数名（HFの場合None）
        """
        # Create Psi4 molecule object from XYZ string
        # XYZ文字列からPsi4分子オブジェクトを作成
        # To prevent Psi4 from reorienting and recentring the molecule, unfortunatelly, very redundant procedure is required.
        # Psi4が分子を再配置および再中心化するのを防ぐために、非常に冗長な手順が必要です。

        # Check if mol_xyz contains atom count and comment lines (standard XYZ format)
        # mol_xyzが原子数とコメント行を含むか確認（標準XYZ形式）
        lines = mol_xyz.strip().splitlines()
        if lines and lines[0].strip().isdigit():
            # Full XYZ format: remove first two lines (atom count and comment)
            # 完全なXYZ形式：最初の2行（原子数とコメント）を削除
            geom_part = "\n".join(lines[2:])
        else:
            # Already atom data only: use as is
            # すでに原子データのみ：そのまま使用
            geom_part = mol_xyz.strip()

        mol_xyz_for_psi4 = "nocom\nnoreorient\n" + geom_part + "\n"
        mol = psi4.geometry(mol_xyz_for_psi4)
        # Prevent Psi4 from reorienting or shifting the molecule any further.
        mol.fix_orientation(True)
        mol.fix_com(True)

        # Store molecular information
        # 分子情報を保存
        self.mol = mol
        self.nuclear_numbers = nuclear_numbers
        self.geom_coordinates = geom_coordinates
        self.basis_set_name = basis_set_name
        self.ksdft_functional_name = ksdft_functional_name
        # Set basis set option in Psi4
        # Psi4で基底関数セットオプションを設定
        # psi4.set_options({
        #     'basis': self.basis_set_name,
        #     'dft_radial_points': 120,
        #     'dft_spherical_points': 434,
        # })
        psi4.set_options({
            'basis': self.basis_set_name,
            'dft_radial_points': 230,
            'dft_spherical_points': 770,
        })
        # Build wavefunction object with specified basis
        # 指定した基底関数でwavefunctionオブジェクトを構築
        wfn = psi4.core.Wavefunction.build(
            self.mol,
            psi4.core.get_global_option('BASIS')
        )
        self.wfn = wfn
        # Get basis set object from wavefunction
        # wavefunctionから基底関数セットオブジェクトを取得
        self.psi4_basis_set = wfn.basisset()
        # Create molecular integrals helper
        # 分子積分ヘルパーを作成
        self.mints = psi4.core.MintsHelper(self.psi4_basis_set)
        # Build DFT functional object (using SVWN as default)
        # DFT汎関数オブジェクトを構築（デフォルトとしてSVWNを使用）
        # Note: True means restricted (closed-shell) system
        # 注：Trueは制限（閉殻）系を意味する
        self.psi4_dft_functional = psi4.driver.dft.build_superfunctional(
            'svwn', True
        )[0]

    def ao_kinetic_integral(self) -> np.ndarray:
        """
        Calculate atomic orbital kinetic energy integrals
        原子軌道基底の運動エネルギー積分を計算

        Returns:
            Kinetic energy integral matrix (NxN) / 運動エネルギー積分行列 (NxN)
        """
        # Compute kinetic energy integrals using Psi4
        # Psi4を使って運動エネルギー積分を計算
        kinetic_integrals = self.mints.ao_kinetic()
        # Convert to numpy array with float64 precision
        # float64精度のnumpy配列に変換
        return np.asarray(kinetic_integrals, dtype='float64')

    def ao_nuclear_attraction_integral(self) -> np.ndarray:
        """
        Calculate atomic orbital nuclear attraction integrals
        原子軌道基底の核引力積分を計算

        Returns:
            Nuclear attraction integral matrix (NxN) / 核引力積分行列 (NxN)
        """
        # Compute nuclear attraction (potential) integrals
        # 核引力（ポテンシャル）積分を計算
        potential_integrals = self.mints.ao_potential()
        # Convert to numpy array with float64 precision
        # float64精度のnumpy配列に変換
        return np.asarray(potential_integrals, dtype='float64')

    def ao_electron_repulsion_integral(self) -> np.ndarray:
        """
        Calculate atomic orbital electron repulsion integrals (ERIs)
        原子軌道基底の電子反発積分（ERI）を計算

        Returns:
            Four-index ERI tensor (NxNxNxN) / 4階のERI行列 (NxNxNxN)
        """
        # Compute two-electron repulsion integrals
        # 2電子反発積分を計算
        eri_integrals = self.mints.ao_eri()
        # Convert to numpy array with float64 precision
        # float64精度のnumpy配列に変換
        return np.asarray(eri_integrals, dtype='float64')

    def ao_overlap_integral(self) -> np.ndarray:
        """
        Calculate atomic orbital overlap integrals
        原子軌道基底の重なり積分を計算

        Returns:
            Overlap integral matrix (NxN) / 重なり積分行列 (NxN)
        """
        # Compute overlap integrals between atomic orbitals
        # 原子軌道間の重なり積分を計算
        overlap_integrals = self.mints.ao_overlap()
        # Convert to numpy array with float64 precision
        # float64精度のnumpy配列に変換
        return np.asarray(overlap_integrals, dtype='float64')

    def generate_numerical_integration_grids_and_weights(
        self,
        mm_coords: Optional[np.ndarray] = None,
        mm_charges: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate 3D Cartesian grids and weights for numerical integration
        数値積分用の3Dデカルトグリッドと重みを生成

        Uses Becke's partitioning scheme for DFT grid generation.
        Reference: https://github.com/psi4/psi4numpy/blob/master/Tutorials/04_Density_Functional_Theory/4a_Grids.ipynb

        DFTグリッド生成にBeckeの分割法を使用。
        参考文献: https://github.com/psi4/psi4numpy/blob/master/Tutorials/04_Density_Functional_Theory/4a_Grids.ipynb

        Args:
            mm_coords: Optional MM coordinates in Angstrom / オプションのMM座標（オングストローム）
            mm_charges: Optional MM charges / オプションのMM電荷

        Returns:
            grids: 3D grid points (Mx3 array) in Bohr / 3Dグリッド点（Mx3配列）（ボーア単位）
            weights: Integration weights for each grid / 各グリッドの積分重み
        """
        # Build potential object for DFT grid generation
        # DFTグリッド生成用のポテンシャルオブジェクトを構築
        # "RV" indicates restricted (closed-shell) potential
        # "RV"は制限（閉殻）ポテンシャルを示す
        v_potential = psi4.core.VBase.build(
            self.psi4_basis_set,
            self.psi4_dft_functional,
            "RV"
        )
        # Initialize the potential (generates grids internally)
        # ポテンシャルを初期化（内部でグリッドを生成）
        v_potential.initialize()
        # Extract x, y, z coordinates and weights from Psi4
        # Psi4からx、y、z座標と重みを抽出
        grids_x, grids_y, grids_z, weights = v_potential.get_np_xyzw()
        # Get total number of grid points
        # グリッド点の総数を取得
        num_grids = len(weights)
        # Combine x, y, z coordinates into single array
        # x、y、z座標を1つの配列にまとめる
        grids = np.zeros((num_grids, 3))
        grids[:, 0] = grids_x  # x coordinates / x座標
        grids[:, 1] = grids_y  # y coordinates / y座標
        grids[:, 2] = grids_z  # z coordinates / z座標
        return grids, weights

    def generate_ao_values_at_grids(self, real_space_grids: np.ndarray) -> np.ndarray:
        """
        Calculate atomic orbital values at specified grid points
        指定したグリッド点での原子軌道の値を計算

        Args:
            real_space_grids: Grid points in real space (Mx3 array) / 実空間のグリッド点 (Mx3配列)

        Returns:
            AO values at each grid point (MxN array) / 各グリッド点でのAO値 (MxN配列)
            M = number of grids, N = number of AOs
            M = グリッド数、N = AO数
        """
        # Get number of grid points
        # グリッド点の数を取得
        num_grids = len(real_space_grids)
        # Get number of atomic orbitals (basis functions)
        # 原子軌道（基底関数）の数を取得
        num_ao = self.psi4_basis_set.nbf()
        # Initialize array to store AO values
        # AO値を格納する配列を初期化
        ao_values_at_grids = np.zeros((num_grids, num_ao))
        # Loop over each grid point
        # 各グリッド点についてループ
        for idx_grid in range(num_grids):
            # Compute all AO values at this grid point
            # このグリッド点での全AO値を計算
            # *real_space_grids[idx_grid, :] unpacks x, y, z coordinates
            # *real_space_grids[idx_grid, :]はx、y、z座標を展開
            ao_values_at_grids[idx_grid] = self.psi4_basis_set.compute_phi(
                *real_space_grids[idx_grid, :]
            )
        return ao_values_at_grids

    def get_basis_atomic_affiliation(self) -> np.ndarray:
        """
        Get atomic center index for each atomic orbital
        各原子軌道の原子中心インデックスを取得

        Returns:
            Array mapping each AO to its atomic center / 各AOを原子中心にマッピングする配列
            (N-length array, where N = number of AOs)
            (長さNの配列、N = AO数)
        """
        # Note: In Psi4, this returns symmetry orbital (SO) affiliation
        # 注：Psi4では対称性軌道（SO）の所属を返す
        # Get number of atomic orbitals
        # 原子軌道の数を取得
        num_ao = self.psi4_basis_set.nbf()
        # Initialize array to store atomic affiliation
        # 原子所属を格納する配列を初期化
        ao_atomic_affiliation = np.zeros(num_ao, dtype=int)
        # Loop over each atomic orbital
        # 各原子軌道についてループ
        for i in range(num_ao):
            # Get the atomic center index for this AO
            # このAOの原子中心インデックスを取得
            ao_atomic_affiliation[i] = self.psi4_basis_set.function_to_center(i)
        return ao_atomic_affiliation

    def compute_external_potential_integrals(
        self,
        mm_coords: np.ndarray,
        mm_charges: np.ndarray,
        grids: Optional[np.ndarray] = None,
        grid_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute external potential integrals from MM point charges using numerical integration.
        数値積分を用いてMM点電荷からの外場ポテンシャル積分を計算

        V_pq = Σ_A q_A ∫ φ_p(r) (1/|r-R_A|) φ_q(r) dr

        Args:
            mm_coords: MM point charge coordinates in Angstrom / MM点電荷座標（オングストローム単位）
            mm_charges: MM point charges in elementary charge / MM点電荷（素電荷単位）
            grids: DFT grid points in Bohr (optional) / DFTグリッド点（ボーア単位）（オプション）
            grid_weights: DFT grid weights (optional) / DFTグリッド重み（オプション）

        Returns:
            External potential matrix in AO basis / AO基底での外場ポテンシャル行列
        """
        if grids is None or grid_weights is None:
            # Generate grids only for QM region (MM region grids are not needed)
            # QM領域のみのグリッドを生成（MM領域のグリッドは不要）
            grids, grid_weights = self.generate_numerical_integration_grids_and_weights()

        return self._compute_external_potential_integrals_numeric(
            mm_coords, mm_charges, grids, grid_weights)

    def _compute_external_potential_integrals_numeric(
        self,
        mm_coords: np.ndarray,
        mm_charges: np.ndarray,
        grids: np.ndarray,
        grid_weights: np.ndarray
    ) -> np.ndarray:
        """
        Compute MM external potential integrals using numerical integration on DFT grids.
        DFTグリッド上の数値積分を用いてMM外場ポテンシャル積分を計算

        V_pq = ∫ φ_p(r) * V_MM(r) * φ_q(r) dr
        where V_MM(r) = Σ_A q_A / |r - R_A|

        Args:
            mm_coords: MM coordinates in Angstrom / MM座標（オングストローム単位）
            mm_charges: MM charges in elementary charge / MM電荷（素電荷単位）
            grids: DFT grid points in Bohr / DFTグリッド点（ボーア単位）
            grid_weights: Integration weights / 積分重み

        Returns:
            External potential matrix in AO basis / AO基底での外場ポテンシャル行列
        """
        # Convert MM coordinates from Angstrom to Bohr
        # MM座標をオングストロームからボーアに変換
        ang_to_bohr = 1.0 / 0.52917721067
        mm_coords_bohr = mm_coords * ang_to_bohr

        num_ao = self.psi4_basis_set.nbf()
        num_grids = len(grid_weights)

        # Compute AO values at all grid points at once (more efficient)
        # 全グリッド点でのAO値を一度に計算（より効率的）
        ao_values = self.generate_ao_values_at_grids(grids)

        # Compute MM electrostatic potential at each grid point
        # 各グリッド点でのMM静電ポテンシャルを計算
        # V_MM(r) = -Σ_A q_A / |r - R_A| where all coordinates are in Bohr
        # V_MM(r) = -Σ_A q_A / |r - R_A| 全ての座標はボーア単位
        # Note: Negative sign because we need attractive potential for electrons
        # 注: 電子に対する引力ポテンシャルなので負符号
        potential_at_grids = np.zeros(num_grids)
        for n in range(num_grids):
            for i_mm in range(len(mm_charges)):
                # Both grids and mm_coords_bohr are in Bohr
                # gridsとmm_coords_bohrは両方ともボーア単位
                r_vec = grids[n, :] - mm_coords_bohr[i_mm, :]
                r = np.linalg.norm(r_vec)
                if r > 1e-10:  # Avoid division by zero / ゼロ除算を回避
                    potential_at_grids[n] += -mm_charges[i_mm] / r

        # Integrate: V_pq = Σ_n w_n * φ_p(r_n) * V_MM(r_n) * φ_q(r_n)
        # 積分: V_pq = Σ_n w_n * φ_p(r_n) * V_MM(r_n) * φ_q(r_n)
        v_ext = np.zeros((num_ao, num_ao))
        for p in range(num_ao):
            for q in range(num_ao):
                integrand = ao_values[:, p] * potential_at_grids * ao_values[:, q] * grid_weights
                v_ext[p, q] = np.sum(integrand)

        print("Computed external potential using numerical integration on DFT grids.")
        return v_ext
