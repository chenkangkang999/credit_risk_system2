# 数据集说明

> 数据文件体积较大（单个 CSV 超过 100MB），不纳入 Git 仓库，请按本说明在本地放置。

## 目录结构

```
data/
├── raw/                     # 原始数据（不入库，需自行准备）
│   ├── train.csv            # 天池真实训练数据（约 80 万行，174MB）
│   └── train_merged.csv     # 天池 + Lending Club 合并数据（约 105 万行，230MB，脚本生成）
├── sample/
│   └── sample.csv           # 合成示例数据（约 1 万行，用于快速调试）
└── models/                  # 训练产物（自动生成，不入库）
```

## 数据来源与字段

| 文件 | 来源 | 格式 | 目标列 |
|------|------|------|--------|
| `train.csv` | 天池《零基础入门金融风控-贷款违约预测》 | 驼峰字段，本项目标准 schema | `isDefault` |
| `ali.csv` | Lending Club 公开数据 | 下划线字段，需经适配器转换 | `loan_status` |

## 准备步骤

### 1. 天池数据

将天池官方 `train.csv` 放入 `data/raw/train.csv`，无需任何转换，字段与
`configs/data_config.yaml` 定义完全一致。

### 2. Lending Club（ali.csv）数据

`ali.csv` 字段格式与天池不同，需先经 `LendingClubAdapter`
（`src/data/adapters/lending_club_adapter.py`）转换。适配器会：

- 字段重命名：`loan_amnt` → `loanAmnt` 等
- 清洗数值：去除 `%`、`months` 后缀
- 解析日期、类别编码、邮编/州映射
- 目标变量：`Fully Paid`→0、`Charged Off/Default`→1，剔除未到期 Current
- 剔除贷款发放后产生的泄漏字段
- `pubRecBankruptcies` 填 0、FICO 用 grade 反推、n0~n14 缺失列交由预处理器填充

### 3. 生成天池 + Lending Club 合并数据

```bash
python scripts/prepare_merged_data.py
```

默认读取 `data/raw/train.csv` 与 `src/数据集/ali.csv`，输出
`data/raw/train_merged.csv`（可用 `--tianchi/--ali/--output` 自定义路径）。

## 训练方式

```bash
# 合成数据（快速调试）
python scripts/train_pipeline.py --use-sample

# 天池真实数据（约 80 万行）
python scripts/train_pipeline.py

# 天池 + Lending Club 合并数据（约 105 万行）
python scripts/train_pipeline.py --data-file train_merged.csv

# 跳过神经网络（更快）
python scripts/train_pipeline.py --data-file train_merged.csv --skip-mlp
```

训练产物自动保存到 `data/models/latest/`，Streamlit 页面在侧边栏
「当前模型来源」选择「本地最新训练」即可加载。
