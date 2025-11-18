"""
Input file parser for quantum chemistry calculations
量子化学計算用の入力ファイルパーサー

This module handles reading molecular geometry from XYZ files
and parsing calculation parameters from configuration files.
このモジュールはXYZファイルから分子構造を読み込み、
設定ファイルから計算パラメータを解析します。
"""

import os
import configparser
from typing import Tuple, Optional
import numpy as np
from basis_set_exchange import lut


def read_xyz(xyz_file_name: str) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Read molecular structure from XYZ file
    XYZファイルから分子構造を読み込む

    Args:
        xyz_file_name: Path to XYZ file / XYZファイルのパス

    Returns:
        mol_xyz: XYZ format string / XYZ形式の文字列
        nuclear_numbers: Atomic numbers array / 原子番号の配列
        coordinates: Atomic coordinates array (Nx3) / 原子座標の配列 (Nx3)

    Raises:
        FileNotFoundError: If XYZ file does not exist / XYZファイルが存在しない場合
        ValueError: If XYZ file format is invalid / XYZファイルの形式が不正な場合
    """
    # Check if XYZ file exists
    # XYZファイルの存在を確認
    if not os.path.isfile(xyz_file_name):
        raise FileNotFoundError(f"XYZ file '{xyz_file_name}' not found.")

    # Open and read XYZ file
    # XYZファイルを開いて読み込む
    try:
        with open(xyz_file_name, 'r') as fh:
            xyz_lines = fh.readlines()
    except IOError as e:
        raise IOError(f"Error reading XYZ file: {e}")

    # Check if file has minimum required lines (header + comment + at least 1 atom)
    # ファイルが最低限必要な行数を持つか確認（ヘッダー+コメント+最低1原子）
    if len(xyz_lines) < 3:
        raise ValueError("XYZ file must have at least 3 lines (number of atoms, comment, atom data)")

    # Initialize lists to store atomic data
    # 原子データを格納するリストを初期化
    nuclear_numbers = []
    coordinates = []
    mol_xyz = ''

    # Parse XYZ file line by line
    # XYZファイルを1行ずつ解析
    for line_num, line in enumerate(xyz_lines, start=1):
        # First line: number of atoms
        # 1行目：原子数
        if line_num == 1:
            try:
                num_atoms = int(line.strip())
            except ValueError:
                raise ValueError(f"First line must be an integer (number of atoms), got: {line.strip()}")

        # Second line: comment (skip)
        # 2行目：コメント（スキップ）
        elif line_num == 2:
            continue

        # Third line onwards: atomic data (element, x, y, z)
        # 3行目以降：原子データ（元素、x、y、z）
        else:
            # Stop if empty line is encountered
            # 空行が見つかったら停止
            if len(line.strip()) == 0:
                break

            # Add line to XYZ string
            # XYZ文字列に行を追加
            mol_xyz += line

            # Split line into parts
            # 行を要素に分割
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Line {line_num} must have at least 4 elements (element x y z), got: {line.strip()}")

            # Parse element (either atomic number or symbol)
            # 元素を解析（原子番号または元素記号）
            try:
                # Try to parse as integer (atomic number)
                # 整数（原子番号）として解析を試みる
                nuclear_numbers.append(int(parts[0]))
            except ValueError:
                # If not integer, parse as element symbol
                # 整数でなければ元素記号として解析
                try:
                    nuclear_numbers.append(lut.element_Z_from_sym(parts[0]))
                except KeyError:
                    raise ValueError(f"Unknown element symbol: {parts[0]}")

            # Parse coordinates (x, y, z)
            # 座標を解析（x、y、z）
            try:
                coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError as e:
                raise ValueError(f"Invalid coordinate format on line {line_num}: {e}")

    # Verify that the number of atoms matches
    # 原子数が一致するか確認
    if len(nuclear_numbers) != num_atoms:
        raise ValueError(f"Number of atoms mismatch: expected {num_atoms}, got {len(nuclear_numbers)}")

    # Convert lists to numpy arrays
    # リストをnumpy配列に変換
    return mol_xyz, np.array(nuclear_numbers, dtype=int), np.array(coordinates, dtype=float)


def get_calc_params(
    flag_multiple_mols: bool = False,
    another_conf_file_name: str = 'sqc2.conf'
) -> Tuple[str, np.ndarray, np.ndarray, str, Optional[str], int, int, str, bool, bool, bool, Optional[str]]:
    """
    Parse configuration file and extract calculation parameters
    設定ファイルを解析して計算パラメータを取得

    Args:
        flag_multiple_mols: Use alternative config file / 代替設定ファイルを使用
        another_conf_file_name: Alternative config file name / 代替設定ファイル名

    Returns:
        mol_xyz: XYZ format string / XYZ形式の文字列
        nuclear_numbers: Atomic numbers / 原子番号
        geom_coordinates: Atomic coordinates / 原子座標
        basis_set_name: Basis set name / 基底関数セット名
        ksdft_functional_name: DFT functional name (None for HF) / DFT汎関数名（HFの場合None）
        molecular_charge: Molecular charge / 分子の電荷
        spin_multiplicity: Spin multiplicity (2S+1) / スピン多重度（2S+1）
        spin_orbital_treatment: 'restricted' or 'unrestricted' / 'restricted'または'unrestricted'
        flag_cis: Run CIS calculation / CIS計算を実行
        flag_mp2: Run MP2 calculation / MP2計算を実行
        flag_qmmm: Enable QM/MM calculation / QM/MM計算を有効化
        mm_charges_file: Path to MM charges file (None if not used) / MM電荷ファイルのパス（未使用の場合None）

    Raises:
        FileNotFoundError: If config file does not exist / 設定ファイルが存在しない場合
        KeyError: If required parameter is missing / 必須パラメータが欠落している場合
    """
    # Determine which config file to use
    # どの設定ファイルを使うか決定
    if flag_multiple_mols:
        conf_file_name = another_conf_file_name
    else:
        conf_file_name = 'sqc.conf'

    # Check if config file exists
    # 設定ファイルの存在を確認
    if not os.path.isfile(conf_file_name):
        raise FileNotFoundError(f"Configuration file '{conf_file_name}' does not exist.")

    # Initialize config parser
    # 設定パーサーを初期化
    sqc_conf = configparser.ConfigParser()

    # Read config file
    # 設定ファイルを読み込む
    try:
        sqc_conf.read(conf_file_name)
    except configparser.Error as e:
        raise ValueError(f"Error parsing config file: {e}")

    # Check if 'calc' section exists
    # 'calc'セクションが存在するか確認
    if 'calc' not in sqc_conf:
        raise KeyError("Config file must have a [calc] section")

    # Extract required parameters from config file
    # 設定ファイルから必須パラメータを取得
    try:
        xyz_file_name = sqc_conf['calc']['geom_xyz']
        basis_set_name = sqc_conf['calc']['gauss_basis_set']
        molecular_charge = int(sqc_conf['calc']['molecular_charge'])
        spin_multiplicity = int(sqc_conf['calc']['spin_multiplicity'])
    except KeyError as e:
        raise KeyError(f"Missing required parameter in config file: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid integer value in config file: {e}")

    # Extract optional DFT functional (None for Hartree-Fock)
    # オプションのDFT汎関数を取得（ハートリー・フォックの場合None）
    ksdft_functional_name = sqc_conf['calc'].get('ksdft_functional', None)

    # Extract spin orbital treatment with automatic default based on spin multiplicity
    # スピン軌道の扱いを取得（スピン多重度に基づいて自動デフォルト設定）
    if sqc_conf['calc'].get('spin_orbital_treatment'):
        spin_orbital_treatment = sqc_conf['calc'].get('spin_orbital_treatment').lower()
        if spin_orbital_treatment not in ['restricted', 'unrestricted']:
            raise ValueError(f"spin_orbital_treatment must be 'restricted' or 'unrestricted', got '{spin_orbital_treatment}'")
    else:
        # Automatic default: restricted for closed-shell (singlet), unrestricted for open-shell
        # 自動デフォルト: 閉殻系（一重項）ではrestricted、開殻系ではunrestricted
        if spin_multiplicity == 1:
            spin_orbital_treatment = 'restricted'
        else:
            spin_orbital_treatment = 'unrestricted'
        print(f"spin_orbital_treatment not specified, using automatic default: '{spin_orbital_treatment}'")
        print(f"spin_orbital_treatmentが指定されていないため、自動デフォルト '{spin_orbital_treatment}' を使用します")

    # Read molecular geometry from XYZ file
    # XYZファイルから分子構造を読み込む
    mol_xyz, nuclear_numbers, geom_coordinates = read_xyz(xyz_file_name)

    # Check if CIS (excited state) calculation is requested
    # CIS（励起状態）計算が要求されているか確認
    excited_state = sqc_conf['calc'].get('excited_state', '').lower()
    flag_cis = (excited_state == 'cis')

    # Check if MP2 (post-Hartree-Fock) calculation is requested
    # MP2（ポストハートリー・フォック）計算が要求されているか確認
    post_hf = sqc_conf['calc'].get('post_hartree-fock', '').lower()
    flag_mp2 = (post_hf == 'mp2')

    # Check if QM/MM calculation is requested
    # QM/MM計算が要求されているか確認
    qmmm_str = sqc_conf['calc'].get('qmmm', 'false').lower()
    flag_qmmm = (qmmm_str == 'true')

    # Get MM charges file path if QM/MM is enabled
    # QM/MMが有効な場合、MM電荷ファイルのパスを取得
    if flag_qmmm:
        mm_charges_file = sqc_conf['calc'].get('mm_charges', None)
        if mm_charges_file is None:
            raise ValueError("QM/MM is enabled but 'mm_charges' parameter is missing in config file")
    else:
        mm_charges_file = None

    # Return all parameters
    # 全パラメータを返す
    return (
        mol_xyz,
        nuclear_numbers,
        geom_coordinates,
        basis_set_name,
        ksdft_functional_name,
        molecular_charge,
        spin_multiplicity,
        spin_orbital_treatment,
        flag_cis,
        flag_mp2,
        flag_qmmm,
        mm_charges_file
    )


def get_analysis_params(
    flag_multiple_mols: bool = False,
    another_conf_file_name: str = 'sqc2.conf'
) -> dict:
    """
    Parse configuration file and extract analysis parameters
    設定ファイルを解析して解析パラメータを取得

    Args:
        flag_multiple_mols: Use alternative config file / 代替設定ファイルを使用
        another_conf_file_name: Alternative config file name / 代替設定ファイル名

    Returns:
        Dictionary of analysis parameters / 解析パラメータの辞書
        {
            'electron_density': bool,  # Whether to visualize electron density / 電子密度を可視化するか
        }

    Raises:
        FileNotFoundError: If config file does not exist / 設定ファイルが存在しない場合
    """
    # Determine which config file to use
    # どの設定ファイルを使うか決定
    if flag_multiple_mols:
        conf_file_name = another_conf_file_name
    else:
        conf_file_name = 'sqc.conf'

    # Check if config file exists
    # 設定ファイルの存在を確認
    if not os.path.isfile(conf_file_name):
        raise FileNotFoundError(f"Configuration file '{conf_file_name}' does not exist.")

    # Initialize config parser
    # 設定パーサーを初期化
    sqc_conf = configparser.ConfigParser()

    # Read config file
    # 設定ファイルを読み込む
    try:
        sqc_conf.read(conf_file_name)
    except configparser.Error as e:
        raise ValueError(f"Error parsing config file: {e}")

    # Initialize analysis parameters dictionary with default values
    # 解析パラメータの辞書をデフォルト値で初期化
    analysis_params = {
        'electron_density': False,
        'electron_density_difference': False,
        'mo_file1': None,
        'mo_file2': None,
    }

    # Check if 'analysis' section exists
    # 'analysis'セクションが存在するか確認
    if 'analysis' in sqc_conf:
        # Get electron density visualization flag
        # 電子密度可視化フラグを取得
        electron_density_str = sqc_conf['analysis'].get('electron_density', 'false').lower()
        analysis_params['electron_density'] = (electron_density_str == 'true')

        # Get electron density difference visualization flag
        # 電子密度差分可視化フラグを取得
        density_diff_str = sqc_conf['analysis'].get('electron_density_difference', 'false').lower()
        analysis_params['electron_density_difference'] = (density_diff_str == 'true')

        # Get MO file paths for density difference calculation
        # 密度差分計算用のMOファイルパスを取得
        if analysis_params['electron_density_difference']:
            analysis_params['mo_file1'] = sqc_conf['analysis'].get('mo_file1', None)
            analysis_params['mo_file2'] = sqc_conf['analysis'].get('mo_file2', None)

    return analysis_params


def read_mm_charges(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read MM point charges from file
    ファイルからMM点電荷を読み込む

    File format / ファイル形式:
        num_charges
        (blank line)
        charge1 x1 y1 z1
        charge2 x2 y2 z2
        ...

    Args:
        file_path: Path to MM charges file / MM電荷ファイルのパス

    Returns:
        mm_coords: MM coordinates in Angstrom (num_charges, 3) / MM座標（オングストローム）
        mm_charges: MM charges in elementary charge (num_charges,) / MM電荷（素電荷）

    Raises:
        FileNotFoundError: If MM charges file does not exist / MM電荷ファイルが存在しない場合
        ValueError: If file format is invalid / ファイル形式が不正な場合
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"MM charges file '{file_path}' not found.")

    with open(file_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 3:
        raise ValueError("MM charges file must have at least 3 lines")

    try:
        num_charges = int(lines[0].strip())
    except ValueError:
        raise ValueError(f"First line must be an integer (number of charges), got: {lines[0].strip()}")

    mm_coords = np.zeros((num_charges, 3))
    mm_charges = np.zeros(num_charges)

    for i in range(num_charges):
        line_idx = i + 2  # Skip first line (count) and second line (blank)
        if line_idx >= len(lines):
            raise ValueError(f"Expected {num_charges} charges, but file has fewer lines")

        parts = lines[line_idx].split()
        if len(parts) < 4:
            raise ValueError(f"Line {line_idx+1} must have 4 values (charge x y z), got: {lines[line_idx].strip()}")

        try:
            mm_charges[i] = float(parts[0])
            mm_coords[i, 0] = float(parts[1])
            mm_coords[i, 1] = float(parts[2])
            mm_coords[i, 2] = float(parts[3])
        except ValueError as e:
            raise ValueError(f"Invalid number format on line {line_idx+1}: {e}")

    return mm_coords, mm_charges
