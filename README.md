# SQCC
SQCC (Simple Quantum Chemistry Code)

Japanese follows English.  
英語の後に日本語の説明があります。

## Feature 特徴
SQCC is a super simple quantum chemistry code written in Python3 and Fortran90.
It is developed for educational purposes. The main purpose of this code is to
provide a simple, but general implementation of quantum chemistry methods that
can be easily understood and modified by users.
SQCC emphasizes the direct implementation of textbook equations rather than
high computational performance or efficiency.
However, it supports almost all basis functions and is in principle applicable
to all molecules and materials.
Thanks to Psi4, SQCC can handle various basis sets and molecular integrals.
In modern quantum chemistry researchn, it is common to leverage
established software packages like Psi4 at the code level as a foundation for implementing
new computational methods and ideas.
SQCC may be used as a starting point for learning such software development.

The following quantum chemistry methods are implemented in this code:
- Hartree-Fock theory (restricted and unrestricted)
- Configuration interaction singles (CIS) theory and excited-state calculations
- Second-order Møller-Plesset perturbation theory (MP2)
- Kohn-Sham density functional theory (KS-DFT) with local density approximation (LDA) (restricted and unrestricted)
- Electrostatic embedding QM/MM calculations with RHF, UHF, RKS, and UKS

Note that only the exchange energy and potential of the local density approximation (LDA) are implemented in the current version.

SQCCは、Python3とFortran90で書かれた非常に単純な量子化学計算コードです。
教育目的で開発されており、ユーザーが簡単に理解し、修正できるような単純で一般的な量子化学手法の実装を提供することを主な目的としています。
SQCCは、高い計算性能や効率というよりも、教科書に載っているような式をそのまま実装することに重点を置いています。
しかしながら、ほとんど全ての基底関数をサポートし、原理的には全ての分子と物質に適用可能です。
Psi4のおかげで、SQCCは様々な基底関数と分子積分を扱うことができます。
現代の量子化学の研究においては、Psi4のような確立されたソフトウェアパッケージをコードレベルで活用し、新しい計算手法やアイデアを実装する基盤とすることがよく行われています。
SQCCは、そのようなソフトウェア開発を学ぶための出発点として利用できます。

以下の量子化学手法がこのコードに実装されています：
- ハートリー・フォック理論（制限および非制限）
- 単一励起配置相互作用（CIS）理論と励起状態計算
- MP2理論
- 局所密度近似（LDA）を用いたコーン・シャム密度汎関数理論（KS-DFT）（制限および非制限）
- 静電埋め込みQM/MM計算（RHF、UHF、RKS、UKS）

注意：現在のバージョンでは、局所密度近似（LDA）の交換エネルギーとポテンシャルのみが実装されています。

## Installation インストール
To install sqcc, assuming you have Anaconda (available from https://www.anaconda.com/download/success) installed, run the following commands:  
sqccをインストールするには、Anaconda（https://www.anaconda.com/download/success から入手可能）がインストールされていることを前提に、以下のコマンドを実行します：  
```bash
git clone https://github.com/QuantumChemistrySchoolJapan-HFT/sqcc.git
cd sqcc
conda create -n sqcc_env psi4 numpy scipy matplotlib pandas -c conda-forge/label/libint_dev -c conda-forge
```

## For biginners 初心者向け
We assume that you can use a terminal in your operating system.
If your operating system is Windows, we recommend using WSL (Windows Subsystem for Linux).
If your operating system is MacOS or Linux, you can use the terminal directly.

我々は、あなたがオペレーティングシステムのターミナルを使用できることを前提としています。
オペレーティングシステムがWindowsの場合、WSL（Windows Subsystem for Linux）の使用をお勧めします。
オペレーティングシステムがMacOSまたはLinuxの場合、ターミナルを直接使用できます。

## Quick Start クイックスタート
To run a simple Hartree-Fock calculation using sqcc, use the following command:  
単純なハートリー・フォック計算をsqccで実行するには、以下のコマンドを使用します：
```bash
cd sqcc/tests/hf/n2_singlet/
# Anaconda environment activation example
# Anaconda環境のアクティベート例
conda activate sqcc_env
python ../../../code/python3/run.py
```

## Inputs インプット
sqc.conf: Configuration file of SimpleQC  
*.xyz: XYZ file of a molecular geometry

sqc.conf: SimpleQCの設定ファイル  
*.xyz: 分子のジオメトリを記述したXYZファイル

sqc.conf example:  
sqc.confの例：
```
[calc]
geom_xyz = ../../n2.xyz
# gauss_basis_set = def2-tzvp
gauss_basis_set = def2-tzvp
ksdft_functional = lda
molecular_charge = 0
# 2S+1
spin_multiplicity = 1
```
"#" indicates a comment line.  
"#"はコメント行を示します。

For more details, see the example files in the `tests` directory.  
詳細については、`tests`ディレクトリ内のサンプルファイルを参照してください。

## Dependencies 依存関係
Psi4: for AO integral and for generating numerical grids and weights
(T.S. only tested in Psi4 1.9.1)  
Basis_Set_Exchange: for getting basis sets

Psi4: AO積分および数値グリッドと重みの生成のため
(Psi4 1.9.1でのみ動作確認済み)  
Basis_Set_Exchange: 基底関数セットの取得のため

## License ライセンス
This project is licensed under the MIT License. See the LICENSE file for details.
The MIT license is a permissive free software license originating at the Massachusetts Institute of Technology (MIT).  
このプロジェクトはMITライセンスの下でライセンスされています。詳細はLICENSEファイルを参照してください。
MITライセンスは、マサチューセッツ工科大学（MIT）に由来する寛容なフリーソフトウェアライセンスです。
