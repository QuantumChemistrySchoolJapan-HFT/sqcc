# SQCC

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Psi4](https://img.shields.io/badge/Psi4-1.9.1-green.svg)](http://www.psicode.org/)

SQCC (Simple Quantum Chemistry Code)

<div align="center">
  <img src="./logo.png" alt="SQCC Logo" width="auto" height="auto">
</div>

---

This README contains both English and Japanese descriptions.

このREADMEには英語と日本語の説明が含まれています。


## Table of Contents / 目次

- [English](#english)
  - [Features](#features)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Inputs](#inputs)
  - [Dependencies](#dependencies)
  - [License](#license)
  - [Note](#note)
  - [References](#references)
  - [Contributors](#contributors)
  - [Acknowledgments](#acknowledgments)
- [日本語](#日本語)
  - [特徴](#特徴)
  - [インストール](#インストール)
  - [クイックスタート](#クイックスタート)
  - [インプット](#インプット)
  - [依存関係](#依存関係)
  - [ライセンス](#ライセンス)
  - [注意](#注意)
  - [参考文献](#参考文献)
  - [貢献者](#貢献者)
  - [謝辞](#謝辞)
---


## English

### Feature
SQCC is a super simple quantum chemistry code written in Python.
It is developed for educational purposes. The main purpose of this code is to
provide a simple, but general implementation of quantum chemistry methods that
can be easily understood and modified by users.
SQCC emphasizes the direct implementation of textbook equations rather than
high computational performance or efficiency.
However, it supports almost all basis functions and is in principle applicable
to all molecules and materials.
Thanks to Psi4, SQCC can handle various basis sets and molecular integrals.
In modern quantum chemistry research, it is common to leverage
established software packages like Psi4 at the code level as a foundation for implementing
new computational methods and ideas.
SQCC may be used as a starting point for learning such software development.

The following quantum chemistry methods are implemented in this code:
- Hartree-Fock theory (restricted and unrestricted)
- Configuration interaction singles (CIS) theory and excited-state calculations
- Second-order Møller-Plesset perturbation theory (MP2)
- Kohn-Sham density functional theory (KS-DFT) with local density approximation (LDA) (restricted and unrestricted)
- Electrostatic embedding QM/MM calculations with RHF, UHF, RKS, UKS, CIS, and MP2

Note that only the exchange energy and potential of the local density approximation (LDA) are implemented in the current version.

### Installation
To install sqcc, assuming you have Anaconda (available from https://www.anaconda.com/download/success) and Git (available from https://git-scm.com/install/windows) installed, run the following commands:
```bash
git clone https://github.com/QuantumChemistrySchoolJapan-HFT/sqcc.git
cd sqcc
conda create -n sqcc_env psi4 numpy scipy matplotlib pandas -c conda-forge/label/libint_dev -c conda-forge
```

### For Beginners
We assume that you can use a terminal in your operating system.
If your operating system is Windows, we recommend using WSL (Windows Subsystem for Linux).
If your operating system is MacOS or Linux, you can use the terminal directly.

### Quick Start
To run a simple Hartree-Fock calculation using sqcc, use the following command:
```bash
cd sqcc/tests/hf/n2_singlet/
# Anaconda environment activation example
conda activate sqcc_env
python ../../../code/python3/run.py
```

### Inputs
- `sqc.conf`: Configuration file of SimpleQC
- `*.xyz`: XYZ file of a molecular geometry

Example of `sqc.conf`:
```
[calc]
geom_xyz = ../../n2.xyz
# gauss_basis_set = def2-tzvp
gauss_basis_set = sto-3g
# ksdft_functional = None indicates the HF calculation
ksdft_functional = lda
molecular_charge = 0
# 2S+1
spin_multiplicity = 1
# spin_orbital_treatment can be optionally specified:
# spin_orbital_treatment = restricted
# or
# spin_orbital_treatment = unrestricted
# If not specified, 'restricted' is used for closed-shell (spin_multiplicity=1),
# and 'unrestricted' is used for open-shell systems.

[analysis]
# True or False
electron_density = True
# electron_density_difference = True
# mo_file1 = ../../hf/benzene_singlet/mo_data.json
# mo_file2 = ../../hf_mm/benzene_singlet/mo_data.json
```
"#" indicates a comment line.

For more details, see the example files in the `tests` directory.

For the xyz file format, see the following link:
https://en.wikipedia.org/wiki/XYZ_file_format

For QM/MM calculations, the xyz format is a little bit extended to include MM point charges with fractional values.
The corresponding `sqc.conf` reads
```
[calc]
geom_xyz = n2.xyz
gauss_basis_set = def2-tzvp
molecular_charge = 0
spin_multiplicity = 1
qmmm = true
mm_charges = mm_charges.xyz

[analysis]
electron_density = true
```
and `mm_charges.xyz` contains
```
2

0.2 0.0 1.0 4.0
0.2 0.0 -1.0 4.0
```
where the first column is the charge value (in elementary charge units), and the next three columns are the x, y, z coordinates of the MM point charges.

### Dependencies
- **Psi4**: for AO integral and for generating numerical grids and weights (Tested with Psi4 1.9.1)
- **Basis_Set_Exchange**: for molecular properties and (getting basis set information in future)

### License
This project is licensed under the MIT License. See the LICENSE file for details.
The MIT license is a permissive free software license originating at the Massachusetts Institute of Technology (MIT).

### Note
Most of the code comments are generated based on those of my old code by using AI (Claude Sonnet 4.5) and I do not guarantee their correctness.
Moreover, most of plotting tools are generated with Claude Sonnet 4.5 and may contain some mistakes.

### References
Most of the equations shown in the comments are from "Modern Quantum Chemistry".

- Attila Szabo, Neil S. Ostlund, "Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory"
- 中井浩巳、「手で解く量子化学 I、 II」
- Frank Jensen, "Introduction to Computational Chemistry"
- R G. Parr, W. Yang, "Density-Functional Theory of Atoms and Molecules".
- D. Cook, "Handbook of Computational Chemistry".

### Contributors
- Takafumi Shiraogawa (Human)
- Claude Sonnet 4.5 (AI)
- ChatGPT-5.1-Codex (AI)

### Acknowledgments
T.S. acknowledges GitHub and Microsoft for providing excellent AI for free.

---

## 日本語

### 特徴
SQCCは、Python3で書かれた非常に単純な量子化学計算コードです。
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
- 静電埋め込みQM/MM計算（RHF、UHF、RKS、UKS、CIS、MP2）

注意：現在のバージョンでは、局所密度近似（LDA）の交換エネルギーとポテンシャルのみが実装されています。

### インストール
sqccをインストールするには、Anaconda（https://www.anaconda.com/download/success から入手可能）とGit（https://git-scm.com/install/windowsから入手可能）がインストールされていることを前提に、以下のコマンドを実行します：
```bash
git clone https://github.com/QuantumChemistrySchoolJapan-HFT/sqcc.git
cd sqcc
conda create -n sqcc_env psi4 numpy scipy matplotlib pandas -c conda-forge/label/libint_dev -c conda-forge
```

### 初心者向け
我々は、あなたがオペレーティングシステムのターミナルを使用できることを前提としています。
オペレーティングシステムがWindowsの場合、WSL（Windows Subsystem for Linux）の使用をお勧めします。
オペレーティングシステムがMacOSまたはLinuxの場合、ターミナルを直接使用できます。

### クイックスタート
単純なハートリー・フォック計算をsqccで実行するには、以下のコマンドを使用します：
```bash
cd sqcc/tests/hf/n2_singlet/
# Anaconda環境のアクティベート例
conda activate sqcc_env
python ../../../code/python3/run.py
```

### インプット
- `sqc.conf`: SimpleQCの設定ファイル
- `*.xyz`: 分子のジオメトリを記述したXYZファイル

`sqc.conf`の例：
```
[calc]
geom_xyz = ../../n2.xyz
# gauss_basis_set = def2-tzvp
gauss_basis_set = sto-3g
# ksdft_functional = None はHF計算を示します
ksdft_functional = lda
molecular_charge = 0
# 2S+1
spin_multiplicity = 1
# spin_orbital_treatment はオプションで指定可能：
# spin_orbital_treatment = restricted
# または
# spin_orbital_treatment = unrestricted
# 指定しない場合、閉殻系（spin_multiplicity=1）では 'restricted'、
# 開殻系では 'unrestricted' が自動的に使用されます。

[analysis]
# True or False
electron_density = True
# electron_density_difference = True
# mo_file1 = ../../hf/benzene_singlet/mo_data.json
# mo_file2 = ../../hf_mm/benzene_singlet/mo_data.json
```
"#"はコメント行を示します。

詳細については、`tests`ディレクトリ内のサンプルファイルを参照してください。

xyzファイルフォーマットについては、以下のリンクを参照してください：
https://en.wikipedia.org/wiki/XYZ_file_format

QM/MM計算の場合、xyzフォーマットはMM点電荷を実数値で含むように少し拡張されています。
`sqc.conf`は以下のように読み込みます：
```
[calc]
geom_xyz = n2.xyz
gauss_basis_set = def2-tzvp
molecular_charge = 0
spin_multiplicity = 1
qmmm = true
mm_charges = mm_charges.xyz

[analysis]
electron_density = true
```
そして、`mm_charges.xyz`は以下の内容を含みます：
```
2

0.2 0.0 1.0 4.0
0.2 0.0 -1.0 4.0
```
ここで、最初の列は電荷値（実数値、単位は素電荷）、次の3列はMM点電荷のx、y、z座標を示します。


### 依存関係
- **Psi4**: AO積分および数値グリッドと数値積分の重みの生成のため（Psi4 1.9.1で動作確認済み）
- **Basis_Set_Exchange**: 基底関数セットの取得のため

### ライセンス
このプロジェクトはMITライセンスの下でライセンスされています。
詳細はLICENSEファイルを参照してください。
MITライセンスは、マサチューセッツ工科大学（MIT）に由来する寛容なフリーソフトウェアライセンスです。

### 注意
コードのコメントの大部分は、私の古いコードのそれらを元にAI（Claude Sonnet 4.5）を使用して生成・修正したものであり、その正確性を保証するものではありません。
さらに、プロットツールの大部分はClaude Sonnet 4.5で生成されており、いくつかの誤りが含まれている可能性があります。

### 参考文献
コメントに示したほとんどの式は、"Modern Quantum Chemistry"に記載されています。

- Attila Szabo, Neil S. Ostlund, "Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory"（和訳：「新しい量子化学」）
- 中井浩巳、「手で解く量子化学 I、 II」
- Frank Jensen, "Introduction to Computational Chemistry"（和訳：「計算化学」）
- R G. Parr, W. Yang, "Density-Functional Theory of Atoms and Molecules"".
- D. Cook, "Handbook of Computational Chemistry".

### 貢献者
- 白男川貴史（人間）
- Claude Sonnet 4.5（人工知能）
- ChatGPT-5.1-Codex（人工知能）

### 謝辞
素晴らしいAIを無料で提供してくださったGithub社とMicrosoft社に感謝します。
