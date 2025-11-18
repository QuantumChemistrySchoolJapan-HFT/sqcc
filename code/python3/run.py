"""
run.py : Main driver for quantum chemistry calculations
run.py : 量子化学計算のメインプログラム

The following command runs this program:
python run.py
下記のコマンドで本プログラムを実行します：
python run.py
"""

# Import necessary modules
# 必要なモジュールをインポート
import time
import logging
import input as conf
import hf_ksdft
import cis_tdhf_tda_tddft
import mp2
import visualizer
import interface_psi4 as ipsi4

# Configure logging to display messages
# ログ出力の設定
logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_hf_ksdft():
    """
    Run Hartree-Fock or Kohn-Sham DFT calculation
    ハートリー・フォック法またはコーン・シャムDFT計算を実行

    Returns:
        myscf: SCF calculation result object / SCF計算結果オブジェクト
        flag_mp2: Boolean flag to run MP2 calculation / MP2計算を実行するかのフラグ
        flag_cis: Boolean flag to run CIS calculation / CIS計算を実行するかのフラグ
    """
    # Start timing the calculation
    # 計算時間の測定を開始
    start = time.perf_counter()

    # Read calculation parameters from configuration file
    # 設定ファイルから計算パラメータを読み込み
    mol_xyz, nuclear_numbers, geom_coordinates, basis_set_name, \
      ksdft_functional_name, molecular_charge, spin_multiplicity, \
      spin_orbital_treatment, flag_cis, flag_mp2, flag_qmmm, mm_charges_file = conf.get_calc_params()

    # Read MM charges if QM/MM is enabled
    # QM/MMが有効な場合、MM電荷を読み込む
    if flag_qmmm:
      mm_coords, mm_charges = conf.read_mm_charges(mm_charges_file)
    else:
      mm_coords = None
      mm_charges = None

    # Initialize the SCF Calculator with molecular information
    # 分子情報を使ってSCFドライバーを初期化
    myscf = hf_ksdft.Calculator(mol_xyz, nuclear_numbers,
                      geom_coordinates, basis_set_name,
                      ksdft_functional_name, molecular_charge,
                      spin_multiplicity, spin_orbital_treatment,
                      mm_coords, mm_charges)

    # Perform the SCF calculation
    # SCF計算を実行
    myscf.scf()

    # Save MO data to JSON file
    # MOデータをJSONファイルに保存
    myscf.save_mo_data_to_json('mo_data.json')

    # Calculate and display elapsed time
    # 経過時間を計算して表示
    elapsed = time.perf_counter() - start
    logging.info(f"run_hf_ksdft elapsed_time: {elapsed:.6f} [sec]\n")

    return myscf, flag_mp2, flag_cis

def run_cis(scf_object):
    """
    Run Configuration Interaction Singles (CIS) calculation
    配置間相互作用シングルス(CIS)計算を実行

    Args:
        scf_object: SCF calculation result object / SCF計算結果オブジェクト

    Returns:
        result: CIS calculation result / CIS計算結果
    """
    # Start timing the calculation
    # 計算時間の測定を開始
    start = time.perf_counter()

    # Initialize the CIS Calculator with SCF result
    # SCF結果を使ってCISドライバーを初期化
    mycis = cis_tdhf_tda_tddft.Calculator(scf_object)

    # Perform the CIS calculation
    # CIS計算を実行
    result = mycis.cis()

    # Calculate and display elapsed time
    # 経過時間を計算して表示
    elapsed = time.perf_counter() - start
    logging.info(f"run_cis elapsed_time: {elapsed:.6f} [sec]\n")

    return result

def run_mp2(scf_object):
    """
    Run second-order Møller-Plesset perturbation theory (MP2) calculation
    二次メラー・プレセット摂動論(MP2)計算を実行

    Args:
        scf_object: SCF calculation result object / SCF計算結果オブジェクト

    Returns:
        result: MP2 calculation result / MP2計算結果
    """
    # Start timing the calculation
    # 計算時間の測定を開始
    start = time.perf_counter()

    # Initialize the MP2 Calculator with SCF result
    # SCF結果を使ってMP2ドライバーを初期化
    mymp2 = mp2.Calculator(scf_object)

    # Perform the MP2 calculation
    # MP2計算を実行
    result = mymp2.mp2()

    # Calculate and display elapsed time
    # 経過時間を計算して表示
    elapsed = time.perf_counter() - start
    logging.info(f"run_mp2 elapsed_time: {elapsed:.6f} [sec]\n")

    return result

def run_analysis(scf_object, analysis_params):
    """
    Run analysis tasks based on configuration
    設定に基づいて解析タスクを実行

    Args:
        scf_object: SCF calculation result object (can be None for density difference only)
                   SCF計算結果オブジェクト（密度差分のみの場合はNone可）
        analysis_params: Dictionary of analysis parameters / 解析パラメータの辞書

    Returns:
        None
    """
    # Check if electron density visualization is requested
    # 電子密度の可視化が要求されているか確認
    if analysis_params.get('electron_density', False):
        if scf_object is None:
            print("Warning: electron_density is True but no SCF calculation was performed")
            print("警告: electron_densityがTrueですが、SCF計算が実行されていません")
        else:
            logging.info("Running electron density visualization...")
            logging.info("電子密度の可視化を実行中...")

            # Start timing the analysis
            # 解析時間の測定を開始
            start = time.perf_counter()

            # Initialize the density visualizer with SCF result
            # SCF結果を使って密度可視化オブジェクトを初期化
            vis = visualizer.DensityVisualizer(scf_object)

            # Plot density on XY, XZ, and YZ planes
            # XY、XZ、YZ平面で密度をプロット
            vis.plot_density_xy_plane(z_position=0.0, output_file='density_xy_plane.svg')
            vis.plot_density_xz_plane(y_position=0.0, output_file='density_xz_plane.svg')
            vis.plot_density_yz_plane(x_position=0.0, output_file='density_yz_plane.svg')

            # Plot 3D slices for better 3D visualization
            # より良い3D可視化のために3Dスライスをプロット
            vis.plot_density_3d_slices(output_file='density_3d_slices.svg')

            # Calculate and display elapsed time
            # 経過時間を計算して表示
            elapsed = time.perf_counter() - start
            logging.info(f"Electron density visualization elapsed_time: {elapsed:.6f} [sec]\n")
            logging.info(f"電子密度可視化の経過時間: {elapsed:.6f} [秒]\n")

    # Check if electron density difference analysis is requested
    # 電子密度差分解析が要求されているか確認
    if analysis_params.get('electron_density_difference', False):
        mo_file1 = analysis_params.get('mo_file1')
        mo_file2 = analysis_params.get('mo_file2')

        if mo_file1 and mo_file2:
            print("=" * 60)
            print("Electron Density Difference Analysis")
            print("電子密度差分解析")
            print("=" * 60)

            # Start timing the analysis
            # 解析時間の測定を開始
            start = time.perf_counter()

            # Load both calculations from JSON files
            # JSONファイルから両方の計算を読み込む
            print(f"Loading MO data from {mo_file1}...")
            print(f"{mo_file1} からMOデータを読み込み中...")
            vis1 = visualizer.DensityVisualizer.from_mo_json(mo_file1)

            print(f"Loading MO data from {mo_file2}...")
            print(f"{mo_file2} からMOデータを読み込み中...")
            vis2 = visualizer.DensityVisualizer.from_mo_json(mo_file2)

            # Plot density difference on XY, XZ, and YZ planes
            # XY、XZ、YZ平面で密度差分をプロット
            print("Plotting density difference on XY plane...")
            print("XY平面で密度差分をプロット中...")
            vis1.plot_density_difference(vis2, plane_origin=[0.0, 0.0, 0.0],
                                        plane_normal=[0.0, 0.0, 1.0],
                                        output_file='density_difference_xy.svg')

            print("Plotting density difference on XZ plane...")
            print("XZ平面で密度差分をプロット中...")
            vis1.plot_density_difference(vis2, plane_origin=[0.0, 0.0, 0.0],
                                        plane_normal=[0.0, 1.0, 0.0],
                                        output_file='density_difference_xz.svg')

            print("Plotting density difference on YZ plane...")
            print("YZ平面で密度差分をプロット中...")
            vis1.plot_density_difference(vis2, plane_origin=[0.0, 0.0, 0.0],
                                        plane_normal=[1.0, 0.0, 0.0],
                                        output_file='density_difference_yz.svg')

            # Calculate and display elapsed time
            # 経過時間を計算して表示
            elapsed = time.perf_counter() - start
            logging.info(f"Density difference analysis elapsed_time: {elapsed:.6f} [sec]\n")
            logging.info(f"密度差分解析の経過時間: {elapsed:.6f} [秒]\n")
        else:
            print("Warning: electron_density_difference is True but mo_file1 or mo_file2 is not specified")
            print("警告: electron_density_differenceがTrueですが、mo_file1またはmo_file2が指定されていません")

# If this run.py file is executed as the main program, run the following operations.
# この run.py ファイルがメインプログラムとして実行された場合、以下の操作を実行
if __name__ == "__main__":
    print("=" * 60)
    print("SQCC: Simple Quantum Chemistry Calculator")
    print("=" * 60)
    print()

    # Print Psi4 version information
    # Psi4バージョン情報を表示
    print("Psi4 version: %s" % ipsi4.get_psi4_version())
    print("Psi4バージョン: %s" % ipsi4.get_psi4_version())
    print()

    # Read analysis parameters first to check if only analysis is requested
    # まず解析パラメータを読み込んで、解析のみが要求されているか確認
    analysis_params = conf.get_analysis_params()

    # Check if only density difference analysis is requested (no SCF calculation needed)
    # 密度差分解析のみが要求されているか確認（SCF計算不要）
    only_density_diff = (
        analysis_params.get('electron_density_difference', False) and
        not analysis_params.get('electron_density', False)
    )

    if only_density_diff:
        # Only run density difference analysis without SCF
        # SCF計算なしで密度差分解析のみを実行
        print("Running density difference analysis only (no SCF calculation)")
        print("密度差分解析のみを実行（SCF計算なし）")
        print()
        run_analysis(None, analysis_params)
    else:
        # Run the initial SCF calculation (Hartree-Fock or Kohn-Sham DFT)
        # 初期のSCF計算を実行（ハートリー・フォック法またはコーン・シャムDFT）
        scf_result, flag_mp2, flag_cis = run_hf_ksdft()

        # If MP2 calculation is requested, run it using the SCF result
        # MP2計算が要求されている場合、SCF結果を使って実行
        if flag_mp2:
            mp2_result = run_mp2(scf_result)

        # If CIS calculation is requested, run it using the SCF result
        # CIS計算が要求されている場合、SCF結果を使って実行
        if flag_cis:
            cis_result = run_cis(scf_result)

        # Run analysis tasks if requested
        # 要求された解析タスクを実行
        if any(analysis_params.values()):
            run_analysis(scf_result, analysis_params)
