# 智能信贷风控分析系统 · 个人负责部分

项目位置：D:\credit_risk_system3

这是在项目组现有仓库上完成的个人负责部分。已完成本地验收和 DeepSeek 真实连接验证，用户已同意检查通过后提交到原仓库 main。公网展示尚未发布。

## 现在怎么查看

本机环境和演示模型已经准备好。预览地址：http://127.0.0.1:8503

若服务已关闭，在 PowerShell 执行：

    cd D:\credit_risk_system3
    powershell -ExecutionPolicy Bypass -File .\启动项目.ps1

只修改本次进程的脚本执行策略，不改变系统设置。也可以直接执行：

    .\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.address=127.0.0.1 --server.port=8503

建议先看：首页 → 单笔预测 → 模型解释 → 风险情景对比 → A/B 模型测试 → 模型监控。

## 本次负责内容

完成第一份开发指导书的 4.1、4.3.4、4.6.4、4.7、4.8、5.1—5.6，以及 7.2、7.5、7.7、7.8。

原有七个页面均已完善，另有监控、A/B 和情景对比三个页面。同学的模型、数据预处理、调参、特征选择和 SHAP 核心代码保持不变，适配和页面衔接在新增文件中完成。

详细记录：

- docs/修改与衔接说明.md：逐文件修改位置、原因、同学代码保留清单。
- docs/章节完成对照表.md：章节与页面、验收路径。
- docs/答辩讲解.md：易于讲解的处理流程、扩展原理和演示顺序。
- docs/免费部署说明.md：免费公开展示的部署步骤。
- docs/change_manifest.json：机器可读修改清单。
- docs/existing_files.diff：原有文件的完整差异。

## API 密钥

本机的 DeepSeek 密钥只保存在 .env，已经通过真实调用验证。仓库仅提供空值 .env.example；其他电脑运行时需自行复制为 .env 并填写，不要填进 Python、YAML、README、截图或提交备注。

.gitignore 已排除 .env、.env.*、.local、Streamlit secrets、日志、原始数据及个人训练产物；.env.example 是可以提交的空值示例。页面仅显示是否配置，不显示密钥内容。

配置后重新启动应用即可使用真实报告、特征建议、异常分析和对话。未配置时提供明确标记的本地示例内容。没有使用本地电脑上其他项目继承来的密钥。

## 数据和训练

内置模型使用原项目合成数据生成器产生 5,000 条记录，随机种子为 42，按 3,500 / 500 / 1,000 划分训练、验证、测试集。预处理和特征筛选只拟合训练集；阈值只在验证集选择；测试集不用于调参。

准备演示：

    .\.venv\Scripts\python.exe scripts/prepare_demo.py

网页可选择 XGBoost、LightGBM、MLP，并使用 AUC 或 KS 进行调参。融合沿用原项目 XGBoost + MLP 的权重配置，LightGBM 单独训练及参与 A/B 对比。

命令行原入口仍可使用：

    .\.venv\Scripts\python.exe scripts/train_pipeline.py --use-sample --tune

原命令使用基础配置，可能比网页快速配置耗时。没有 --skip-tuning 参数，默认跳过调参，--tune 表示启用调参。

真实天池 train.csv 放入 data/raw，仍使用同学的数据和模型模块。大数据的 KNN 计算可能较慢，可先取小样本联调。真实数据不要复制到 demo_assets。

## 验证

本机使用 Python 3.12.14，独立虚拟环境在 .venv。复现安装：

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements-review.txt
    .\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

原 requirements.txt 保留，审阅版本的已验证依赖单独放在 requirements-review.txt。公开部署使用 app/requirements.txt。

测试命令：

    powershell -ExecutionPolicy Bypass -File .\运行测试.ps1

原 18 项测试保持原样；新增 26 项测试验证数据划分、推理一致性、A/B、JSON、规则检查和 API 脱敏等。本机 LightGBM 原生保存接口不支持部分中文临时路径，所以测试临时目录使用项目内的 .local/pytest，没有改动原测试。

页面检查使用 Streamlit AppTest，覆盖本地与公开展示模式各 11 页。交互检查使用 scripts/check_interactions.py，会执行一次本地快速训练。

## 展示边界

合成数据 AUC 约 0.53—0.58，只用于验证流程，不代表真实信贷效果。规则检查使用课程自定义阈值；SHAP 树模型原始输出不是概率。MLP 或融合结果旁的树模型 SHAP 会明确标注子模型名称。

以后发布公开站点时，默认不调用真实 API，只使用演示模型和示例报告。免费部署文件和说明已经准备，公开展示的部署入口及模式设置见 docs/免费部署说明.md。
