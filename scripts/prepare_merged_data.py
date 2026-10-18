#!/usr/bin/env python
"""
合并数据集准备脚本
==================
把天池 train.csv 和 Lending Club（ali.csv）合并成统一的天池格式训练集。

流程:
    1. 天池数据已是标准格式，直接读取
    2. ali.csv 经 LendingClubAdapter 转成天池格式
    3. 以天池列为准对齐：ali 缺失的 n0~n14 等列填 NaN（交给预处理器填充）
    4. 重新生成全局唯一 id，避免两个数据源 id 冲突
    5. 合并并打乱，输出到 data/raw/train_merged.csv

使用:
    python scripts/prepare_merged_data.py

    # 可选参数
    python scripts/prepare_merged_data.py --tianchi data/raw/train.csv --ali "src/数据集/ali.csv" --output data/raw/train_merged.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data.adapters import LendingClubAdapter
from src.utils.logger import get_logger

logger = get_logger("scripts.prepare_merged")

# 合并后的训练集每块读取行数（ali.csv 较大，分块转换避免内存峰值）
CHUNK_SIZE = 100_000


def load_tianchi(path: Path) -> pd.DataFrame:
    """读取天池标准格式数据"""
    logger.info(f"读取天池数据: {path}")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"天池数据: {df.shape[0]:,} 行, {df.shape[1]} 列")
    return df


def load_lending_club(path: Path) -> pd.DataFrame:
    """分块读取并转换 Lending Club 数据"""
    logger.info(f"读取并转换 Lending Club 数据: {path}")
    adapter = LendingClubAdapter()
    chunks = []
    for i, chunk in enumerate(pd.read_csv(path, low_memory=False, chunksize=CHUNK_SIZE)):
        converted = adapter.transform(chunk)
        chunks.append(converted)
        logger.info(f"  已处理第 {i + 1} 块，累计 {sum(len(c) for c in chunks):,} 行")
    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"Lending Club 数据转换完成: {df.shape[0]:,} 行, {df.shape[1]} 列")
    return df


def merge_datasets(tianchi: pd.DataFrame, lending: pd.DataFrame) -> pd.DataFrame:
    """
    以天池列为准对齐并合并两个数据集。

    - 天池有、LC 没有的列（n0~n14）：LC 部分为 NaN，由预处理器填充
    - LC 有、天池没有的列：丢弃（adapter 已基本清干净）
    """
    target_cols = tianchi.columns.tolist()

    # 给 LC 补齐缺失列（NaN）
    for col in target_cols:
        if col not in lending.columns:
            lending[col] = np.nan
            logger.info(f"LC 缺少列 [{col}]，已补 NaN（后续由预处理器填充）")

    # 只保留天池标准列，丢弃 LC 多余列
    lending = lending[target_cols].copy()
    tianchi = tianchi[target_cols].copy()

    # 标记数据来源（便于排查，不参与建模，最后会删除）
    tianchi["__source__"] = "tianchi"
    lending["__source__"] = "lending_club"

    merged = pd.concat([tianchi, lending], ignore_index=True)

    # 重新生成全局唯一 id（避免两个数据源 id 冲突）
    merged["id"] = np.arange(1, len(merged) + 1)

    # 统一标签列为整数
    merged["isDefault"] = pd.to_numeric(merged["isDefault"], errors="coerce").fillna(0).astype(int)

    # 打乱顺序，防止两个数据源前后分布集中
    merged = merged.sample(frac=1.0, random_state=42).reset_index(drop=True)

    source_count = merged["__source__"].value_counts()
    logger.info(f"合并完成: {len(merged):,} 行，来源分布: {dict(source_count)}")
    merged = merged.drop(columns=["__source__"])
    return merged


def main():
    parser = argparse.ArgumentParser(description="合并天池与 Lending Club 数据集")
    parser.add_argument("--tianchi", default=str(PROJECT_ROOT / "data" / "raw" / "train.csv"),
                        help="天池数据路径")
    parser.add_argument("--ali", default=str(PROJECT_ROOT / "src" / "数据集" / "ali.csv"),
                        help="Lending Club (ali.csv) 数据路径")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "raw" / "train_merged.csv"),
                        help="合并后输出路径")
    args = parser.parse_args()

    tianchi_path = Path(args.tianchi)
    ali_path = Path(args.ali)
    output_path = Path(args.output)

    if not tianchi_path.exists():
        raise FileNotFoundError(f"天池数据不存在: {tianchi_path}")
    if not ali_path.exists():
        raise FileNotFoundError(f"ali.csv 不存在: {ali_path}")

    # 1. 读取两个数据集
    tianchi = load_tianchi(tianchi_path)
    lending = load_lending_club(ali_path)

    # 2. 对齐并合并
    merged = merge_datasets(tianchi, lending)

    # 3. 输出
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"合并数据已保存: {output_path} ({output_path.stat().st_size / 1024**2:.1f} MB)")

    # 4. 标签分布检查
    pos_rate = merged["isDefault"].mean()
    print("\n" + "=" * 60)
    print("合并数据准备完成")
    print("=" * 60)
    print(f"总样本数:   {len(merged):,}")
    print(f"特征列数:   {merged.shape[1]}")
    print(f"违约率:     {pos_rate:.2%}")
    print(f"输出文件:   {output_path}")
    print("\n训练时把 configs/data_config.yaml 的 train_file 改为 train_merged.csv")
    print("或直接运行: python scripts/train_pipeline.py --data-file train_merged.csv")


if __name__ == "__main__":
    main()
