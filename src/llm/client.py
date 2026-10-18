import json
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path

from openai import (APIConnectionError, APITimeoutError, AuthenticationError,
                    InternalServerError, OpenAI, RateLimitError)

from src.llm.billing import BillingTracker, safe_model, token_count, usage_dict
from src.utils.logger import get_logger

logger = get_logger("llm")
_TRANSIENT = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)


class DeepSeekClient:
    def __init__(self, config):
        self.config = config
        self.deepseek_cfg = config.deepseek
        self.model = self.deepseek_cfg.model
        self.client = None
        self._request_times = []
        self._lock = threading.Lock()
        self.audit_log_path = Path(__file__).resolve().parents[2] / "logs/llm_audit.log"
        self.billing = BillingTracker()
        self.last_billing = None

    @property
    def billing_log_path(self):
        return self.billing.ledger_path

    @billing_log_path.setter
    def billing_log_path(self, path):
        self.billing.ledger_path = Path(path)

    def _record_usage(self, model, usage, stream=False, status="ok"):
        try:
            self.last_billing = self.billing.record(model, usage, stream, status)
        except (OSError, ValueError, KeyError, TypeError):
            self.billing.last_error = "本次费用未能估算，请检查计费配置；实际扣费以服务商账单为准"
            self.last_billing = None

    @property
    def configured(self):
        key = os.getenv(self.deepseek_cfg.api_key_env, "")
        return bool(key and not key.startswith("sk-your")) and os.getenv("APP_ENV") != "public_demo" and os.getenv("CREDIT_RISK_PUBLIC_DEMO") != "1"

    def _require_client(self):
        if not self.configured:
            raise ValueError("请在本地 .env 中配置 DeepSeek API Key；公开演示模式不调用 API")
        if self.client is None:
            self.client = OpenAI(api_key=os.environ[self.deepseek_cfg.api_key_env],
                                 base_url=self.deepseek_cfg.base_url,
                                 timeout=self.deepseek_cfg.default_params.timeout, max_retries=0)

    def _sanitize_input(self, text):
        text = str(text)
        key = os.getenv(self.deepseek_cfg.api_key_env, "")
        if key:
            text = text.replace(key, "[已隐藏密钥]")
        text = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[已隐藏密钥]", text)
        text = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[已隐藏手机号]", text)
        return re.sub(r"(?<!\d)\d{17}[\dXx](?!\d)", "[已隐藏证件号]", text)

    def _build_messages(self, prompt, system_prompt=None, history=None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": self._sanitize_input(system_prompt)})
        turns = self.config.agents.chat_agent.max_history_turns
        for message in (history or [])[-turns*2:]:
            if message.get("role") in ("user", "assistant"):
                messages.append({"role": message["role"], "content": self._sanitize_input(message["content"])})
        messages.append({"role": "user", "content": self._sanitize_input(prompt)})
        return messages

    def _check_rate_limit(self):
        cfg = self.deepseek_cfg.rate_limit
        if not cfg.enabled:
            return
        with self._lock:
            now = time.monotonic()
            self._request_times = [stamp for stamp in self._request_times if now-stamp < 60]
            if len(self._request_times) >= cfg.requests_per_minute:
                raise ValueError("请求次数较多，请稍后再试")
            self._request_times.append(now)

    def _params(self, kwargs):
        params = dict(self.deepseek_cfg.default_params)
        params.pop("timeout", None)
        params.update(kwargs)
        return params

    def _log_audit(self, duration, usage, status):
        if not self.config.safety.audit_log.enabled:
            return
        audit_usage = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens",
                    "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
            value = token_count(usage.get(key))
            if value is not None:
                audit_usage[key] = value
        for key, allowed in (("prompt_tokens_details", ("cached_tokens",)),
                             ("completion_tokens_details", ("reasoning_tokens",))):
            details = usage.get(key)
            if isinstance(details, dict):
                filtered = {name: details[name] for name in allowed if token_count(details.get(name)) is not None}
                if filtered:
                    audit_usage[key] = filtered
        record = {"timestamp": datetime.now().isoformat(), "model": safe_model(self.model),
                  "duration_sec": round(duration, 3), "usage": audit_usage, "status": status}
        try:
            self.audit_log_path.parent.mkdir(exist_ok=True)
            with self.audit_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            logger.warning("审计日志暂时无法写入")

    def chat(self, prompt, system_prompt=None, history=None, **kwargs):
        system_prompt = kwargs.pop("system", system_prompt)
        self._require_client()
        messages = self._build_messages(prompt, system_prompt, history)
        cfg = self.deepseek_cfg.retry
        for attempt in range(cfg.max_attempts):
            self._check_rate_limit()
            start = time.monotonic()
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, stream=False, **self._params(kwargs))
                usage = usage_dict(getattr(response, "usage", None))
                self._record_usage(getattr(response, "model", None) or self.model, usage)
                self._log_audit(time.monotonic()-start, usage, "ok")
                return response.choices[0].message.content or ""
            except AuthenticationError:
                raise ValueError("API 密钥未通过验证，请检查本地配置") from None
            except _TRANSIENT:
                self._record_usage(self.model, {}, status="connection_error")
                self._log_audit(time.monotonic()-start, {}, "connection_error")
                if attempt+1 == cfg.max_attempts:
                    raise RuntimeError("DeepSeek 暂时无法连接，请稍后重试") from None
                time.sleep(min(cfg.initial_wait * cfg.backoff_factor**attempt, cfg.max_wait))
            except Exception:
                raise RuntimeError("DeepSeek 请求失败，请检查模型名称、余额与服务状态") from None

    def stream_chat(self, prompt, system_prompt=None, history=None, **kwargs):
        system_prompt = kwargs.pop("system", system_prompt)
        self._require_client()
        messages = self._build_messages(prompt, system_prompt, history)
        cfg = self.deepseek_cfg.retry
        for attempt in range(cfg.max_attempts):
            emitted = False
            recorded = False
            usage = {}
            response_model = self.model
            self._check_rate_limit()
            start = time.monotonic()
            try:
                params = self._params(kwargs)
                params["stream_options"] = {**(params.get("stream_options") or {}), "include_usage": True}
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, stream=True, **params)
                complete = False
                try:
                    for chunk in response:
                        response_model = getattr(chunk, "model", None) or response_model
                        if getattr(chunk, "usage", None) is not None:
                            usage = usage_dict(chunk.usage)
                        if chunk.choices and chunk.choices[0].delta.content:
                            emitted = True
                            yield chunk.choices[0].delta.content
                    complete = True
                finally:
                    try:
                        response.close()
                    finally:
                        self._record_usage(response_model, usage, stream=True,
                                           status="ok" if complete else "stream_interrupted")
                        recorded = True
                self._log_audit(time.monotonic()-start, usage, "ok")
                return
            except AuthenticationError:
                raise ValueError("API 密钥未通过验证，请检查本地配置") from None
            except _TRANSIENT:
                if not recorded:
                    self._record_usage(response_model, usage, stream=True, status="stream_interrupted")
                self._log_audit(time.monotonic()-start, {}, "stream_interrupted")
                if emitted or attempt+1 == cfg.max_attempts:
                    raise RuntimeError("报告生成中断，已收到的内容保留，请重试") from None
                time.sleep(min(cfg.initial_wait * cfg.backoff_factor**attempt, cfg.max_wait))
            except Exception:
                raise RuntimeError("报告生成失败，请检查服务状态后重试") from None

    def health_check(self):
        try:
            self.chat("请回复 OK", max_tokens=10)
            return True
        except (ValueError, RuntimeError):
            return False
