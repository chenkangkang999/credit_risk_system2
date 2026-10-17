#!/usr/bin/env python
"""
数据探索脚本
============
对数据集做基础探索性分析（EDA），输出：
    1. 数据基本信息（行列数、字段类型）
    2. 缺失值统计
    3. 目标变量分布（违约率）
    4. 数值特征统计摘要
    5. 类别特征分布
    6. 特征与目标变量的相关性

使用:
    python scripts/explore_data.py                        # 自动找数据
    python scripts/explore_data.py --data data/sample/sample.csv  # 指定数据
    python scripts/explore_data.py --output reports/eda.txt        # 输出到文件
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 让脚本能 import src
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_data(data_path: str | None = None) -> tuple[pd.DataFrame, str]:
    """加载数据，优先真实数据，回退到合成数据"""
    if data_path:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在: {path}")
        print(f"📂 加载指定数据: {path}")
        return pd.read_csv(path), str(path)

    # 自动找数据
    root = Path(__file__).parent.parent
    real_path = root / "data" / "raw" / "train.csv"
    sample_path = root / "data" / "sample" / "sample.csv"

    if real_path.exists():
        print(f"📂 加载真实数据: {real_path}")
        return pd.read_csv(real_path), str(real_path)
    elif sample_path.exists():
        print(f"📂 加载合成数据: {sample_path}")
        return pd.read_csv(sample_path), str(sample_path)
    else:
        raise FileNotFoundError(
            "未找到数据文件！\n"
            "  请先运行: python scripts/generate_sample_data.py\n"
            "  或下载真实数据放到 data/raw/train.csv"
        )


def explore_basic_info(df: pd.DataFrame) -> list[str]:
    """输出基本信息"""
    lines = []
    lines.append("=" * 60)
    lines.append("1. 数据基本信息")
    lines.append("=" * 60)
    lines.append(f"  行数:         {len(df):,}")
    lines.append(f"  列数:         {df.shape[1]}")
    lines.append(f"  内存占用:     {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    lines.append("")
    lines.append("  字段类型分布:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        lines.append(f"    {dtype}: {count} 列")
    lines.append("")
    return lines


def explore_missing_values(df: pd.DataFrame) -> list[str]:
    """输出缺失值统计"""
    lines = []
    lines.append("=" * 60)
    lines.append("2. 缺失值统计")
    lines.append("=" * 60)

    missing = df.isnull().sum()
    missing_rate = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "缺失数": missing,
        "缺失率(%)": missing_rate,
    })
    missing_df = missing_df[missing_df["缺失数"] > 0].sort_values("缺失率(%)", ascending=False)

    if len(missing_df) == 0:
        lines.append("  ✅ 无缺失值")
    else:
        lines.append(f"  共 {len(missing_df)} 列有缺失值:")
        for col, row in missing_df.iterrows():
            lines.append(f"    {col:25s}  {int(row['缺失数']):>6,}  ({row['缺失率(%)']:.2f}%)")

    lines.append("")
    return lines


def explore_target(df: pd.DataFrame, target_col: str = "isDefault") -> list[str]:
    """输出目标变量分布"""
    lines = []
    lines.append("=" * 60)
    lines.append("3. 目标变量分布（违约率）")
    lines.append("=" * 60)

    if target_col not in df.columns:
        lines.append(f"  ⚠️ 未找到目标列: {target_col}")
        lines.append("")
        return lines

    counts = df[target_col].value_counts().sort_index()
    total = len(df)
    lines.append(f"  正常样本 (0):  {counts.get(0, 0):>6,}  ({counts.get(0, 0) / total:.2%})")
    lines.append(f"  违约样本 (1):  {counts.get(1, 0):>6,}  ({counts.get(1, 0) / total:.2%})")
    lines.append(f"  违约率:        {df[target_col].mean():.2%}")
    lines.append("")

    if df[target_col].mean() < 0.05:
        lines.append("  ⚠️ 违约率过低，类别极度不平衡，需注意类别加权")
    elif df[target_col].mean() > 0.5:
        lines.append("  ⚠️ 违约率过高，数据可能有问题")
    else:
        lines.append("  ✅ 类别相对平衡")

    lines.append("")
    return lines


def explore_numerical_features(df: pd.DataFrame, target_col: str = "isDefault") -> list[str]:
    """输出数值特征统计摘要"""
    lines = []
    lines.append("=" * 60)
    lines.append("4. 数值特征统计摘要")
    lines.append("=" * 60)

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 排除 id 和 target
    exclude = ["id", target_col]
    numerical_cols = [c for c in numerical_cols if c not in exclude]

    if len(numerical_cols) == 0:
        lines.append("  无数值特征")
        lines.append("")
        return lines

    desc = df[numerical_cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    lines.append(f"  共 {len(numerical_cols)} 个数值特征:")
    lines.append("")

    # 打印前20个，太长就截断
    show_cols = numerical_cols[:20]
    for col in show_cols:
        row = desc.loc[col]
        lines.append(
            f"  {col:25s}  "
            f"均值={row['mean']:>10.2f}  "
            f"标准差={row['std']:>10.2f}  "
            f"范围=[{row['min']:.2f}, {row['max']:.2f}]"
        )

    if len(numerical_cols) > 20:
        lines.append(f"  ... 还有 {len(numerical_cols) - 20} 个特征省略")

    lines.append("")
    return lines


def explore_categorical_features(df: pd.DataFrame) -> list[str]:
    """输出类别特征分布"""
    lines = []
    lines.append("=" * 60)
    lines.append("5. 类别特征分布")
    lines.append("=" * 60)

    # 找类别特征（object 类型，或取值少的数值列）
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    # 加上取值数 < 20 的数值列也算类别
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].nunique() < 20 and col not in ["id", "isDefault"]:
            categorical_cols.append(col)

    if len(categorical_cols) == 0:
        lines.append("  无类别特征")
        lines.append("")
        return lines

    lines.append(f"  共 {len(categorical_cols)} 个类别特征:")
    lines.append("")

    for col in categorical_cols[:10]:  # 只看前10个
        lines.append(f"  📌 {col}:")
        value_counts = df[col].value_counts().head(8)
        total = len(df)
        for val, count in value_counts.items():
            lines.append(f"      {str(val):20s}  {count:>6,}  ({count / total:.1%})")
        if df[col].nunique() > 8:
            lines.append(f"      ... 共 {df[col].nunique()} 个类别")
        lines.append("")

    if len(categorical_cols) > 10:
        lines.append(f"  ... 还有 {len(categorical_cols) - 10} 个类别特征省略")
        lines.append("")

    return lines


def explore_correlation(df: pd.DataFrame, target_col: str = "isDefault") -> list[str]:
    """输出特征与目标的相关性"""
    lines = []
    lines.append("=" * 60)
    lines.append("6. 特征与目标变量的相关性")
    lines.append("=" * 60)

    if target_col not in df.columns:
        lines.append(f"  ⚠️ 未找到目标列: {target_col}")
        lines.append("")
        return lines

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [c for c in numerical_cols if c != "id"]

    correlations = df[numerical_cols].corr()[target_col].sort_values(key=abs, ascending=False)
    correlations = correlations.drop(target_col)

    lines.append("  与违约相关性最高的 Top 15:")
    lines.append("")
    for col, corr in correlations.head(15).items():
        direction = "↑ 正相关" if corr > 0 else "↓ 负相关"
        lines.append(f"  {col:25s}  {corr:+.4f}  {direction}")

    lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser(description="数据探索性分析")
    parser.add_argument("--data", type=str, default=None, help="数据文件路径（默认自动找）")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径（默认只打印到屏幕）")
    args = parser.parse_args()

    # 加载数据
    df, data_source = load_data(args.data)

    # 运行探索
    all_lines = []
    all_lines.append(f"🔍 数据探索报告")
    all_lines.append(f"📁 数据来源: {data_source}")
    all_lines.append("")

    all_lines.extend(explore_basic_info(df))
    all_lines.extend(explore_missing_values(df))
    all_lines.extend(explore_target(df))
    all_lines.extend(explore_numerical_features(df))
    all_lines.extend(explore_categorical_features(df))
    all_lines.extend(explore_correlation(df))

    all_lines.append("=" * 60)
    all_lines.append("✅ 探索完成")
    all_lines.append("=" * 60)

    # 打印到屏幕
    report = "\n".join(all_lines)
    print("\n" + report)

    # 保存到文件
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"\n💾 报告已保存至: {output_path}")


if __name__ == "__main__":
    main()
