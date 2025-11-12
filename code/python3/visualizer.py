"""
Electron density visualizer on arbitrary planes
任意の平面上の電子密度可視化モジュール

This module provides functions to visualize electron density
on user-defined planes in 3D space using Psi4's built-in functionality.
このモジュールはPsi4の組み込み機能を使って3D空間の
ユーザー定義平面上の電子密度を可視化する機能を提供します。
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class DensityVisualizer():
  """
  Visualize electron density on arbitrary planes
  任意の平面上の電子密度を可視化
  """

  def __init__(self, scf_calculator):
    """
    Initialize visualizer with SCF results
    SCF結果で可視化ツールを初期化

    Args:
        scf_calculator: SCF calculator object with completed calculation
                       計算完了済みのSCF計算オブジェクト
    """
    self.scf = scf_calculator
    # Check if SCF has been performed
    # SCFが実行済みか確認
    if not hasattr(scf_calculator, 'density_matrix_in_ao_basis'):
      raise ValueError("SCF calculation must be performed before visualization.")

    self.density_matrix = scf_calculator.density_matrix_in_ao_basis
    self.basis_set_object = scf_calculator.basis_set_object
    self.num_ao = scf_calculator.num_ao
    self.nuclear_numbers = scf_calculator.nuclear_numbers
    self.geom_coordinates = scf_calculator.geom_coordinates

    # Use Psi4 interface from SCF calculator
    # SCF計算オブジェクトのPsi4インターフェースを使用
    self.psi4_interface = scf_calculator.psi4_interface


  def calculate_density_at_points(self, points):
    """
    Calculate electron density at given 3D points
    指定された3D点での電子密度を計算

    Uses interface_psi4.generate_ao_values_at_grids for educational purposes
    教育目的でinterface_psi4.generate_ao_values_at_gridsを使用

    Args:
        points: Array of 3D coordinates in Bohr (N x 3)
               ボーア単位の3D座標配列 (N x 3)

    Returns:
        Electron density at each point / 各点での電子密度
    """
    num_points = len(points)
    densities = np.zeros(num_points)

    # Get density matrix
    # 密度行列を取得
    if self.scf.spin_multiplicity == 1:
      # Restricted case / 制限系の場合
      density = self.density_matrix
    else:
      # Unrestricted case: use total density
      # 非制限系の場合: 全密度を使用
      density = self.density_matrix[0] + self.density_matrix[1]

    # Calculate AO values at all grid points using interface_psi4
    # interface_psi4を使って全グリッド点でのAO値を計算
    ao_values = self.psi4_interface.generate_ao_values_at_grids(points)

    # Calculate density at each point: rho(r) = sum_mu,nu P_mu,nu * phi_mu(r) * phi_nu(r)
    # 各点で密度を計算: rho(r) = sum_mu,nu P_mu,nu * phi_mu(r) * phi_nu(r)
    # This is equivalent to: phi^T * P * phi
    # これは以下と等価: phi^T * P * phi
    for i_point in range(num_points):
      phi = ao_values[i_point, :]
      temp = np.dot(density, phi)
      densities[i_point] = np.dot(phi, temp)

    return densities


  @staticmethod
  def get_atom_properties(atomic_number):
    """
    Get atom color and radius based on atomic number
    原子番号に基づいて原子の色と半径を取得

    Args:
        atomic_number: Atomic number / 原子番号

    Returns:
        tuple: (color, radius) / (色, 半径)
    """
    # Common atom colors (CPK coloring scheme)
    # 一般的な原子色（CPKカラースキーム）
    atom_colors = {
      1: "white",      # H
      2: "cyan",       # He
      3: "violet",     # Li
      4: "darkgreen",  # Be
      5: "salmon",     # B
      6: "gray",       # C
      7: "blue",       # N
      8: "red",        # O
      9: "lightgreen", # F
      10: "cyan",      # Ne
      11: "violet",    # Na
      12: "darkgreen", # Mg
      13: "gray",      # Al
      14: "goldenrod", # Si
      15: "orange",    # P
      16: "yellow",    # S
      17: "green",     # Cl
      18: "cyan",      # Ar
      # Add more elements as needed / 必要に応じて元素を追加
    }

    # Van der Waals radii (in Angstrom, approximate)
    # ファンデルワールス半径（オングストローム、近似値）
    # For visualization, we use smaller values
    # 可視化用に小さい値を使用
    vdw_radii = {
      1: 0.31,   # H
      2: 0.28,   # He
      3: 0.76,   # Li
      4: 0.59,   # Be
      5: 0.54,   # B
      6: 0.52,   # C
      7: 0.48,   # N
      8: 0.46,   # O
      9: 0.42,   # F
      10: 0.38,  # Ne
      11: 0.97,  # Na
      12: 0.86,  # Mg
      13: 0.75,  # Al
      14: 0.70,  # Si
      15: 0.65,  # P
      16: 0.60,  # S
      17: 0.55,  # Cl
      18: 0.51,  # Ar
      # Add more elements as needed / 必要に応じて元素を追加
    }

    # Get color (default to gray for unknown elements)
    # 色を取得（未知の元素はグレー）
    color = atom_colors.get(atomic_number, "gray")

    # Get radius (default to 0.5 Å for unknown elements)
    # 半径を取得（未知の元素は0.5 Å）
    radius = vdw_radii.get(atomic_number, 0.5)

    # Use smaller radius for visualization clarity
    # 視認性のため小さい半径を使用
    display_radius = radius * 0.35

    return color, display_radius


  @staticmethod
  def conv_log_scale(values):
    """
    Convert values to log scale for better visualization
    より良い可視化のために値をlogスケールに変換

    Values below threshold are set to zero, others are converted to signed log10
    閾値以下の値はゼロに設定し、それ以外は符号付きlog10に変換

    Args:
        values: 2D array of density values / 密度値の2D配列

    Returns:
        Log-scaled values / logスケール変換された値
    """
    thres = 1.e-6

    # All zero values
    # 全てゼロの場合
    if np.amax(values) == 0.0 and np.amin(values) == 0.0:
      return values
    else:
      values = values / thres
      for i in range(len(values)):
        for j in range(len(values[i])):
          if np.abs(values[i, j]) < 1.0:
            values[i, j] = 0.0
          else:
            values[i, j] = np.sign(values[i, j]) * np.log10(np.abs(values[i, j]))

    return values


  def plot_density_on_plane(self, plane_origin, plane_normal, plane_extent=5.0,
                           num_points=50, output_file='density_plane.pdf'):
    """
    Plot electron density on an arbitrary plane
    任意の平面上の電子密度をプロット

    Args:
        plane_origin: Origin point of the plane in Angstrom (3D array)
                     平面の原点（オングストローム単位、3D配列）
        plane_normal: Normal vector of the plane (3D array, will be normalized)
                     平面の法線ベクトル（3D配列、正規化されます）
        plane_extent: Half-width of the plane in Angstrom / 平面の半幅（オングストローム）
        num_points: Number of grid points in each direction / 各方向のグリッド点数
        output_file: Output PDF file name / 出力PDFファイル名
    """
    # Convert Angstrom to Bohr
    # オングストロームをボーアに変換
    ang_to_bohr = 1.0 / 0.52917721067
    plane_origin_bohr = np.array(plane_origin) * ang_to_bohr
    plane_extent_bohr = plane_extent * ang_to_bohr

    # Normalize the normal vector
    # 法線ベクトルを正規化
    normal = np.array(plane_normal)
    normal = normal / np.linalg.norm(normal)

    # Determine axis labels based on plane orientation
    # 平面の向きに基づいて軸ラベルを決定
    # Check if this is a standard plane (XY, XZ, or YZ)
    # 標準平面（XY、XZ、YZ）かどうかを確認
    # For standard planes, we set v1 and v2 explicitly to match expected axes
    # 標準平面の場合、期待される軸に合わせてv1とv2を明示的に設定

    # Create two orthogonal vectors in the plane
    # 平面内の2つの直交ベクトルを作成
    if np.allclose(np.abs(normal), [0, 0, 1]):  # XY plane (normal along Z)
      # v1 along X, v2 along Y
      v1 = np.array([1.0, 0.0, 0.0])
      v2 = np.array([0.0, 1.0, 0.0])
      xlabel = '$\it{x}$ / Å'
      ylabel = '$\it{y}$ / Å'
    elif np.allclose(np.abs(normal), [0, 1, 0]):  # XZ plane (normal along Y)
      # v1 along X, v2 along Z
      v1 = np.array([1.0, 0.0, 0.0])
      v2 = np.array([0.0, 0.0, 1.0])
      xlabel = '$\it{x}$ / Å'
      ylabel = '$\it{z}$ / Å'
    elif np.allclose(np.abs(normal), [1, 0, 0]):  # YZ plane (normal along X)
      # v1 along Y, v2 along Z
      v1 = np.array([0.0, 1.0, 0.0])
      v2 = np.array([0.0, 0.0, 1.0])
      xlabel = '$\it{y}$ / Å'
      ylabel = '$\it{z}$ / Å'
    else:  # Arbitrary plane
      # Choose an arbitrary vector not parallel to normal
      # 法線に平行でない任意のベクトルを選択
      if abs(normal[0]) < 0.9:
        arbitrary = np.array([1.0, 0.0, 0.0])
      else:
        arbitrary = np.array([0.0, 1.0, 0.0])

      # First basis vector: cross product
      # 第1基底ベクトル: 外積
      v1 = np.cross(normal, arbitrary)
      v1 = v1 / np.linalg.norm(v1)

      # Second basis vector: cross product of normal and v1
      # 第2基底ベクトル: 法線とv1の外積
      v2 = np.cross(normal, v1)
      v2 = v2 / np.linalg.norm(v2)

      xlabel = '$\it{u}$ / Å'
      ylabel = '$\it{v}$ / Å'

    # Create grid in the plane
    # 平面内にグリッドを作成
    u = np.linspace(-plane_extent_bohr, plane_extent_bohr, num_points)
    v = np.linspace(-plane_extent_bohr, plane_extent_bohr, num_points)
    U, V = np.meshgrid(u, v)

    # Calculate 3D coordinates for each grid point
    # 各グリッド点の3D座標を計算
    # Note: meshgrid returns arrays where U[i,j] corresponds to u[j] and V[i,j] corresponds to v[i]
    # 注: meshgridはU[i,j]がu[j]に、V[i,j]がv[i]に対応する配列を返す
    points = np.zeros((num_points * num_points, 3))
    for i in range(num_points):
      for j in range(num_points):
        idx = i * num_points + j
        # U[i,j] = u[j], V[i,j] = v[i]
        points[idx, :] = plane_origin_bohr + U[i, j] * v1 + V[i, j] * v2

    print("Calculating electron density at %d points..." % len(points))
    print("%d個の点で電子密度を計算中..." % len(points))

    # Calculate densities at all points
    # 全点で密度を計算
    densities = self.calculate_density_at_points(points)

    # Reshape to 2D grid
    # 2Dグリッドに整形
    density_grid = densities.reshape((num_points, num_points))

    # Apply log10 scale to density for better visualization
    # 視覚化を改善するため密度にlog10スケールを適用
    log_density_grid = self.conv_log_scale(density_grid.copy())

    # Set up matplotlib style
    # matplotlibのスタイルを設定
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams["xtick.major.size"] = 8
    plt.rcParams["ytick.major.size"] = 8

    # Create plot
    # プロットを作成
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    # Convert grid back to Angstrom for plotting
    # プロット用にグリッドをオングストロームに戻す
    U_ang = U / ang_to_bohr
    V_ang = V / ang_to_bohr

    # Determine contour range
    # 等高線の範囲を決定
    min_value = np.amin(log_density_grid)
    max_value = np.amax(log_density_grid)

    if min_value == 0.0 and max_value == 0.0:
      contour_range = np.linspace(0.0, 1.0, 11)
      map_color = 'Blues'
    else:
      if min_value >= 0.0:
        contour_range = np.linspace(0.0, max_value, 11)
        map_color = 'Blues'
      else:
        abs_max = max(abs(min_value), abs(max_value))
        contour_range = np.linspace(-abs_max, abs_max, 21)
        map_color = 'RdBu_r'

    # Plot density as contour map in log10 scale
    # 密度をlog10スケールで等高線図としてプロット
    contour = ax.contourf(U_ang, V_ang, log_density_grid, contour_range, cmap=map_color, extend='both')

    # Add colorbar
    # カラーバーを追加
    cbar = plt.colorbar(contour, ax=ax, ticks=contour_range[::2], format='%1.2f')
    cbar.set_label('log₁₀(ρ/10⁻⁶) [e/Bohr³]', fontsize=12)

    # Draw atoms on the plane
    # 平面上に原子を描画
    for atomidx, atom_number in enumerate(self.nuclear_numbers):
      atom_pos = self.geom_coordinates[atomidx]

      # Project atom position onto the plane
      # 原子位置を平面に投影
      atom_pos_bohr = atom_pos * ang_to_bohr
      relative_pos = atom_pos_bohr - plane_origin_bohr
      u_coord = np.dot(relative_pos, v1)
      v_coord = np.dot(relative_pos, v2)

      # Check if atom is close to the plane (within 0.5 Bohr)
      # 原子が平面に近いか確認（0.5 Bohr以内）
      distance_to_plane = abs(np.dot(relative_pos, normal))
      if distance_to_plane < 0.5:
        # Get atom color and radius based on atomic number
        # 原子番号に基づいて原子の色と半径を取得
        atom_color, atom_radius = self.get_atom_properties(atom_number)

        atom = patches.Circle(
          xy=[u_coord / ang_to_bohr, v_coord / ang_to_bohr],
          radius=atom_radius,
          fc=atom_color,
          ec='black',
          linewidth=1.5,
          alpha=1.0
        )
        ax.add_patch(atom)    # Labels and title
    # ラベルとタイトル
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_aspect('equal')

    # Set axis limits
    # 軸の範囲を設定
    ax.set_xlim(-plane_extent, plane_extent)
    ax.set_ylim(-plane_extent, plane_extent)

    # Save figure as PDF
    # 図をPDFとして保存
    plt.tight_layout()
    plt.savefig(output_file, format='pdf', dpi=900)
    print("Density plot saved to %s" % output_file)
    print("密度プロットを %s に保存しました" % output_file)
    plt.close()

    return density_grid


  def plot_density_xy_plane(self, z_position=0.0, plane_extent=5.0,
                           num_points=50, output_file='density_xy.pdf'):
    """
    Plot electron density on XY plane at given Z coordinate
    指定されたZ座標のXY平面上の電子密度をプロット

    Args:
        z_position: Z coordinate of the plane in Angstrom / 平面のZ座標（オングストローム）
        plane_extent: Half-width of the plane in Angstrom / 平面の半幅（オングストローム）
        num_points: Number of grid points in each direction / 各方向のグリッド点数
        output_file: Output PDF file name / 出力PDFファイル名
    """
    return self.plot_density_on_plane(
      plane_origin=[0.0, 0.0, z_position],
      plane_normal=[0.0, 0.0, 1.0],
      plane_extent=plane_extent,
      num_points=num_points,
      output_file=output_file)


  def plot_density_xz_plane(self, y_position=0.0, plane_extent=5.0,
                           num_points=50, output_file='density_xz.pdf'):
    """
    Plot electron density on XZ plane at given Y coordinate
    指定されたY座標のXZ平面上の電子密度をプロット

    Args:
        y_position: Y coordinate of the plane in Angstrom / 平面のY座標（オングストローム）
        plane_extent: Half-width of the plane in Angstrom / 平面の半幅（オングストローム）
        num_points: Number of grid points in each direction / 各方向のグリッド点数
        output_file: Output PDF file name / 出力PDFファイル名
    """
    return self.plot_density_on_plane(
      plane_origin=[0.0, y_position, 0.0],
      plane_normal=[0.0, 1.0, 0.0],
      plane_extent=plane_extent,
      num_points=num_points,
      output_file=output_file)


  def plot_density_yz_plane(self, x_position=0.0, plane_extent=5.0,
                           num_points=50, output_file='density_yz.pdf'):
    """
    Plot electron density on YZ plane at given X coordinate
    指定されたX座標のYZ平面上の電子密度をプロット

    Args:
        x_position: X coordinate of the plane in Angstrom / 平面のX座標（オングストローム）
        plane_extent: Half-width of the plane in Angstrom / 平面の半幅（オングストローム）
        num_points: Number of grid points in each direction / 各方向のグリッド点数
        output_file: Output PDF file name / 出力PDFファイル名
    """
    return self.plot_density_on_plane(
      plane_origin=[x_position, 0.0, 0.0],
      plane_normal=[1.0, 0.0, 0.0],
      plane_extent=plane_extent,
      num_points=num_points,
      output_file=output_file)
