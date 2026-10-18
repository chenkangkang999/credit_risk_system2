import json
import math
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def usage_dict(usage):
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return usage if isinstance(usage, dict) else vars(usage)


def token_count(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def safe_model(model):
    model = str(model)
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9._:-]{0,99}", model) and not model.startswith(("sk-", "ghp_", "github_pat_")):
        return model
    return "unknown"


class BillingTracker:
    def __init__(self, config_path=None, ledger_path=None):
        self.config_path = Path(config_path or ROOT / "configs/billing_config.yaml")
        self.ledger_path = Path(ledger_path or ROOT / ".local/llm_billing.sqlite3")
        self.last_error = None

    def config(self):
        try:
            with self.config_path.open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
        except yaml.YAMLError:
            raise ValueError("计费 YAML 格式错误") from None
        if not isinstance(config, dict) or not isinstance(config.get("models"), dict):
            raise ValueError("计费配置及 models 应为映射")
        if config.get("period", "peak") not in ("peak", "off_peak"):
            raise ValueError("计价时段应为 peak 或 off_peak")
        unit = config.get("unit_tokens", 1000000)
        if isinstance(unit, bool) or not isinstance(unit, (int, float)) or not math.isfinite(unit) or unit <= 0:
            raise ValueError("计费单位应为正数")
        for model, details in config["models"].items():
            if not isinstance(model, str) or not isinstance(details, dict):
                raise ValueError("模型单价应为映射")
            aliases = details.get("aliases", [])
            if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
                raise ValueError("模型别名应为字符串列表")
            for period in ("peak", "off_peak"):
                rates = details.get(period)
                if not isinstance(rates, dict):
                    raise ValueError("峰谷单价应为映射")
                for key in ("input_cache_hit", "input_cache_miss", "output"):
                    rate = rates.get(key)
                    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate < 0:
                        raise ValueError("计费单价应为非负数")
        return config

    def price_for(self, model, config):
        for name, details in config.get("models", {}).items():
            if model == name or model in details.get("aliases", []):
                return details.get(config.get("period", "peak"))
        return None

    def make_record(self, model, usage, stream=False, status="ok"):
        usage = usage_dict(usage)
        model = safe_model(model)
        prompt = token_count(usage.get("prompt_tokens"))
        output = token_count(usage.get("completion_tokens"))
        hit = token_count(usage.get("prompt_cache_hit_tokens"))
        miss = token_count(usage.get("prompt_cache_miss_tokens"))
        details = usage.get("prompt_tokens_details") or {}
        if not isinstance(details, dict):
            details = usage_dict(details)
        invalid_cache = any(usage.get(key) is not None and token_count(usage[key]) is None
                            for key in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens"))
        if hit is None:
            hit = token_count(details.get("cached_tokens"))
            invalid_cache = invalid_cache or (details.get("cached_tokens") is not None and hit is None)
        if prompt is not None and not invalid_cache:
            if hit is not None and miss is None and hit <= prompt:
                miss = prompt - hit
            if miss is not None and hit is None and miss <= prompt:
                hit = prompt - miss
        record = {"id": uuid.uuid4().hex, "timestamp": datetime.now(timezone.utc).isoformat(),
                  "model": model, "stream": bool(stream), "status": status,
                  "prompt_tokens": prompt, "completion_tokens": output,
                  "cache_hit_tokens": hit, "cache_miss_tokens": miss,
                  "cost_cny": None, "cost_min_cny": None, "cost_max_cny": None,
                  "billing_status": "usage_missing", "price_verified_on": None,
                  "price_period": None, "rates": None}
        if prompt is None or output is None:
            return record
        if invalid_cache or (hit is not None and hit > prompt) or (miss is not None and miss > prompt) or (hit is not None and miss is not None and hit + miss != prompt):
            record["billing_status"] = "usage_invalid"
            return record
        try:
            config = self.config()
        except (OSError, ValueError):
            record["billing_status"] = "price_config_error"
            self.last_error = "计费配置暂时不可用，本次费用待核实；实际扣费以 DeepSeek 账单为准"
            return record
        record["price_verified_on"] = config.get("verified_on")
        record["price_period"] = config.get("period", "peak")
        price = self.price_for(model, config)
        if price is None:
            record["billing_status"] = "price_unknown"
            return record
        rates = {name: float(price[name]) for name in ("input_cache_hit", "input_cache_miss", "output")}
        if any(not math.isfinite(rate) or rate < 0 for rate in rates.values()):
            raise ValueError("计费单价应为非负数")
        unit = float(config.get("unit_tokens", 1000000))
        if not math.isfinite(unit) or unit <= 0:
            raise ValueError("计费单位应为正数")
        record["rates"] = rates
        if hit is None or miss is None:
            record["billing_status"] = "cache_split_missing"
            record["cost_min_cny"] = round((prompt * min(rates["input_cache_hit"], rates["input_cache_miss"]) + output * rates["output"]) / unit, 10)
            record["cost_max_cny"] = round((prompt * max(rates["input_cache_hit"], rates["input_cache_miss"]) + output * rates["output"]) / unit, 10)
        else:
            cost = round((hit * rates["input_cache_hit"] + miss * rates["input_cache_miss"] + output * rates["output"]) / unit, 10)
            record.update(cost_cny=cost, cost_min_cny=cost, cost_max_cny=cost, billing_status="estimated")
        return record

    def record(self, model, usage, stream=False, status="ok"):
        self.last_error = None
        record = self.make_record(model, usage, stream, status)
        try:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.ledger_path, timeout=10) as database:
                database.execute("CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, record TEXT NOT NULL)")
                database.execute("INSERT INTO calls VALUES (?, ?)", (record["id"], json.dumps(record, ensure_ascii=False)))
        except (OSError, sqlite3.Error):
            self.last_error = "本次用量未能写入本地账本，累计统计可能不完整"
        return record

    def records(self, limit=None):
        if not self.ledger_path.exists():
            return []
        try:
            with sqlite3.connect(self.ledger_path, timeout=10) as database:
                query = "SELECT record FROM calls ORDER BY rowid DESC"
                rows = database.execute(query + (" LIMIT ?" if limit is not None else ""),
                                        (limit,) if limit is not None else ()).fetchall()
            return [json.loads(row[0]) for row in rows]
        except (OSError, sqlite3.Error, ValueError):
            self.last_error = "暂时无法读取本地账本，累计统计可能不完整"
            return []

    def summary(self):
        records = self.records()
        priced = [record for record in records if record["cost_cny"] is not None]
        return {"calls": len(records), "prompt_tokens": sum(record["prompt_tokens"] or 0 for record in records),
                "completion_tokens": sum(record["completion_tokens"] or 0 for record in records),
                "cost_cny": round(sum(record["cost_cny"] for record in priced), 10),
                "pending_calls": len(records) - len(priced),
                "missing_usage_calls": sum(record["billing_status"] in ("usage_missing", "usage_invalid") for record in records)}
