import ast
import json
import re
import subprocess
from pathlib import Path

from check_changes import CHECKOUT_BASE, ORIGINAL_TEAM_BASE, ROOT, USER_BASE, build_report, git


def symbols(source):
    found = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return found
    def visit(nodes, prefix=""):
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                visit(node.body, prefix+node.name+".")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found[prefix+node.name] = (ast.dump(node, include_attributes=False), node.lineno)
    visit(tree.body)
    return found


def main():
    report = build_report()
    reasons = {
        "README.md": "更新两人分工、真实数据接入状态、三模型融合与特征选择说明。",
        ".gitignore": "增加环境文件、私有设置、日志和产物排除，单独允许合成演示模型。",
        "app/assets/style.css": "统一主题、导航、指标卡片和窄屏布局。",
        "app/components/sidebar.py": "品牌名称与英文小字分组，删除标语；保留导航与模型来源逻辑。",
        "app/components/runtime.py": "真实文件缓存绑定修改时间并返回副本，加载前校验，失败明确提示。",
        "app/streamlit_app.py": "首页状态、模型指标、流程介绍与功能入口。",
        "configs/llm_config.yaml": "注册政策检查模板与 agents.compliance，实际使用启停、温度和 token 上限；API 仍通过环境变量读取。",
        "configs/prompts/feature_suggestion.j2": "改成固定操作的结构化建议，不要求返回可执行代码。",
        "configs/prompts/risk_report.j2": "使用实际模型名称与风险区间，移除固定融合模型文案。",
        "src/llm/agents/case_agent.py": "增加可选分类阈值，让案例筛选和分析与评估阈值一致。",
        "src/llm/agents/feature_agent.py": "只发送字段统计，解析结构化建议，预览固定特征操作。",
        "src/llm/agents/report_agent.py": "支持实际模型名称、阈值和可选 SHAP。",
        "src/llm/client.py": "密钥按 .env 读取，真实请求前脱敏；同步/流式记录真实 Token，按响应模型估算费用，计费故障不丢弃已生成回答。",
        "src/llm/prompts.py": "模板目录按项目定位，中文 JSON 渲染与模板缓存刷新。",
        "src/pipeline/orchestrator.py": "仅在融合条件和模型列表两处加入 LightGBM，衔接同学三模型权重配置。",
        "scripts/check_changes.py": "保护基线更新到 d329cbf，记录上次个人上传后的同学改动。",
        "scripts/check_staged.py": "只允许与同学基线一致的数据目录说明文件，继续阻止真实数据及私有文件上传。",
        "scripts/write_review.py": "更新本轮改动说明与三模型衔接记录，补充监控页面的说明映射。",
        "docs/拟提交说明.md": "记录用户已批准的提交范围、验证结果及隐私排除。",
        "docs/验收结果.md": "更新真实数据加载验证和针对性测试，说明未开展全量训练。",
        "tests/test_compliance.py": "同步政策合规检查标签名称的既有断言。",
        "src/utils/config_loader.py": "支持普通字典、属性访问及安全环境变量替换，明确读取项目 .env。",
        "src/utils/logger.py": "集中日志处理，避免重复 handler，日志中的密钥自动隐藏。",
        "src/utils/metrics.py": "统一概率格式，处理单类/非法输入，提供指标、阈值和曲线数据。",
    }
    page_reasons = {
        "1_": "切换数据来源时清理旧选择，加载与校验成功后才更新状态，失败保留原来源。",
        "2_": "精简 KNN 填充说明，去掉人员归属描述。",
        "3_": "模型选择、稳健训练方案与前后对照、AUC/KS/多目标调参、曲线、验证指标与特征重要性。",
        "4_": "合规标签改为政策合规检查，移除页面上的章节编号。",
        "5_": "全局 SHAP、分布、依赖及单笔解释。",
        "6_": "将聊天输入框放入普通布局，避免页面自动下滚裁掉顶部标识，对话和计费逻辑不变。",
        "7_": "只读查看同学基础配置，本地保存扩展配置，只显示 API 配置状态。",
        "8_": "精简模型监控页面说明，去掉人员归属描述。"
    }
    for path in report["modified_existing_files"]:
        if path.startswith("app/pages/"):
            reasons[path] = next(value for key, value in page_reasons.items() if Path(path).name.startswith(key))
    lines = ["# 修改与衔接说明", "",
             "项目目录：D:\\credit_risk_system3。用户已同意本次检查通过后上传原仓库 main；公网展示尚未发布。",
             "", "## 基线区别", "",
             f"- 用户提供的提交：{USER_BASE}",
             f"- 原有同学核心代码基线：{ORIGINAL_TEAM_BASE}",
             f"- 实际下载并保留的同学最新提交：{CHECKOUT_BASE}",
             "- 最新基线包含已上传的个人版本及同学的适配器、真实数据入口和三模型权重。27 个受保护核心文件保持原样；共用应用文件仅进行下表列明的小范围修复。",
             "", "## 改动过的原有文件", "",
             "以下原文件主要是你负责章节的基础实现。这里如实列出修改，不能理解为所有原文件一行未动。",
             "", "| 原有文件 | 修改位置（当前行号） | 原因 |",
             "|---|---|---|"]
    for path in report["modified_existing_files"]:
        changes = []
        if path.endswith(".py"):
            old = symbols(git("show", f"{CHECKOUT_BASE}:{path}").decode("utf-8"))
            new = symbols((ROOT/path).read_text(encoding="utf-8"))
            for name, (body, line) in new.items():
                if name not in old or old[name][0] != body:
                    changes.append(f"{name}（{line}）")
            removed = sorted(set(old)-set(new))
            if removed:
                changes.append("替换/移除旧辅助函数："+"、".join(removed))
        position = "；".join(changes) or "页面脚本 / 配置 / 样式"
        lines.append(f"| {path} | {position} | {reasons.get(path, '完善本次负责的基础实现。')} |")
    lines += ["", "## 同学核心文件保持原样", "",
              "下面文件与实际下载基线逐项比较一致（Git 的 CRLF/LF 行尾差异规范化后比较）：", ""]
    lines += ["- "+p for p in report["protected_unchanged"]]
    lines += ["", "## 新增文件", ""]
    lines += ["- "+p for p in report["new_files"] if not p.endswith(".png")]
    lines += ["", "## 关键衔接方式", "",
              "- KNN：ProjectPreprocessor 继承原 Preprocessor，继续调用同学的填充逻辑；仅在新增适配文件里处理全空数值列及标准化状态。",
              "- LightGBM：补充 AttrDict.to_dict，不修改 LGBModel.fit；完整产物使用可信 joblib 保存已训练对象，避免原 Booster 加载后预测接口不同。",
              "- 多目标调参：CompatibleOptimizer 调用原优化器，仅在取单一 best_params 不适用时，从 Pareto 结果按指标优先、复杂度次优选参数。",
              "- 特征选择：直接调用原 FeatureSelector，只改变调用时机，确保只拟合训练集。",
              "- SHAP：继续调用原 SHAPAnalyzer，页面负责画力图并区分原始输出与模型概率。",
              "- 漂移监控：页面直接调用同学 DriftDetector，不重写检测算法。",
              "- 融合：沿用同学模型类与 0.4/0.4/0.2 权重配置，三模型齐全且开启融合时加入 XGBoost、LightGBM 和 MLP；缺模型时跳过融合。",
              "- 真实数据：保留同学全量文件入口，补充来源切换、缓存刷新、返回副本和加载失败保护；原始 CSV 不修改、不上传。",
              "- 本地设置保存到 .local/settings.yaml，不覆盖同学的 data、feature、model 配置。",
              "", "## 接口与运行习惯", "",
              "- 原 TrainingPipeline.run 参数保留，新增 model_names、data、quick；原命令行 --use-sample、--skip-mlp、--tune 保留。",
              "- InferenceService.predict_proba 接收原始 DataFrame，返回一维违约概率。",
              "- 完整产物以 data/models/latest/bundle.joblib 为准；保留可用的原生模型文件。跳过某模型训练后请按 meta 中的 model_names 使用当前模型。",
              "- 新数据输入不重新 fit；缺少辅助字段采用训练集默认值。",
              "", "## 如何审核", "",
              "- 原有文件完整差异在 docs/existing_files.diff。",
              "- 结构化清单在 docs/change_manifest.json。",
              "- 重新执行 python scripts/check_changes.py 会检查受保护文件及忽略规则。",
              "- 用户已同意本次上传；提交前检查远端 main，并用 scripts/check_staged.py 核对实际提交内容。",
              "- 本地 DeepSeek 此前已通过真实连接验证；本轮未发起真实 API 请求，密钥仍只保留在被忽略的 .env。", ""]
    (ROOT/"docs/修改与衔接说明.md").write_text("\n".join(lines), encoding="utf-8")
    candidates = set(git("ls-files").decode("utf-8").splitlines()+report["new_files"])
    violations = []
    for path in sorted(candidates):
        file = ROOT/path
        if file.suffix.lower() not in {".py", ".yaml", ".yml", ".toml", ".md", ".txt", ".json", ".diff", ".ps1"} or not file.is_file():
            continue
        content = file.read_text(encoding="utf-8-sig", errors="replace")
        if re.search(r"sk-[A-Za-z0-9_-]{20,}", content):
            violations.append(path)
    tracked_env = git("ls-files", ".env").decode().strip()
    from dotenv import dotenv_values
    local_key_present = bool(dotenv_values(ROOT/".env").get("DEEPSEEK_API_KEY"))
    result = {"candidate_source_files_checked": len(candidates), "secret_pattern_matches": violations,
              "env_tracked": bool(tracked_env), "ignored_paths": report["ignored_secret_paths"],
              "real_api_configured": local_key_present,
              "local_head": git("rev-parse", "HEAD").decode().strip(),
              "staged_files": git("diff", "--cached", "--name-only").decode().splitlines()}
    (ROOT/"docs/security_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if violations or tracked_env:
        raise ValueError("发现疑似密钥或被跟踪的 .env，仅报告文件名，请本地检查")
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
