from types import SimpleNamespace

import pytest

from src.llm.billing import BillingTracker
from src.llm.client import DeepSeekClient
from src.utils.config_loader import AttrDict


@pytest.fixture
def tracker(tmp_path):
    return BillingTracker(ledger_path=tmp_path / "billing.sqlite3")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("CREDIT_RISK_PUBLIC_DEMO", raising=False)
    monkeypatch.setenv("BILLING_TEST_API_KEY", "test-placeholder")
    config = AttrDict({"deepseek": {"model": "deepseek-chat", "api_key_env": "BILLING_TEST_API_KEY",
                       "base_url": "https://api.deepseek.com", "default_params": {"timeout": 30},
                       "retry": {"max_attempts": 1}, "rate_limit": {"enabled": False}},
                       "agents": {"chat_agent": {"max_history_turns": 3}},
                       "safety": {"audit_log": {"enabled": False}}})
    instance = DeepSeekClient(config)
    instance.billing_log_path = tmp_path / "billing.sqlite3"
    return instance


def full_usage():
    return {"prompt_tokens": 1000, "completion_tokens": 200,
            "prompt_cache_hit_tokens": 400, "prompt_cache_miss_tokens": 600}


def test_cache_split_price_and_no_prompt_in_ledger(tracker):
    usage = {**full_usage(), "prompt": "private question", "response": "private answer", "api_key": "private credential"}
    record = tracker.record("deepseek-flash", usage)
    assert record["cost_cny"] == pytest.approx((400 * 0.04 + 600 * 2 + 200 * 8) / 1000000)
    assert record["billing_status"] == "estimated"
    data = tracker.ledger_path.read_bytes()
    assert all(text.encode() not in data for text in ("private question", "private answer", "private credential"))
    assert record["rates"] == {"input_cache_hit": 0.04, "input_cache_miss": 2.0, "output": 8.0}


def test_ledger_survives_new_instance(tracker):
    tracker.record("deepseek-flash", full_usage())
    another = BillingTracker(ledger_path=tracker.ledger_path)
    another.record("deepseek-flash", full_usage(), stream=True)
    summary = another.summary()
    assert summary["calls"] == 2
    assert summary["prompt_tokens"] == 2000
    assert summary["completion_tokens"] == 400
    assert len(another.records(1)) == 1


@pytest.mark.parametrize("usage", [None, {}, {"total_tokens": 9}, {"prompt_tokens": 3},
                                   {"prompt_tokens": True, "completion_tokens": 2}])
def test_missing_usage_is_not_free(tracker, usage):
    record = tracker.record("deepseek-flash", usage)
    assert record["cost_cny"] is None and record["billing_status"] == "usage_missing"
    assert tracker.summary()["pending_calls"] == 1


def test_missing_cache_split_returns_range(tracker):
    record = tracker.record("deepseek-flash", {"prompt_tokens": 1000, "completion_tokens": 200})
    assert record["cost_cny"] is None
    assert record["cost_min_cny"] < record["cost_max_cny"]
    assert record["billing_status"] == "cache_split_missing"


def test_cached_tokens_details_can_fill_split(tracker):
    record = tracker.record("deepseek-flash", {"prompt_tokens": 1000, "completion_tokens": 200,
                                              "prompt_tokens_details": {"cached_tokens": 400}})
    assert record["cache_miss_tokens"] == 600
    assert record["billing_status"] == "estimated"


def test_inconsistent_cache_counts_are_not_priced(tracker):
    record = tracker.record("deepseek-flash", {**full_usage(), "prompt_cache_hit_tokens": 700})
    assert record["billing_status"] == "usage_invalid"
    assert record["cost_cny"] is None


def test_negative_cache_tokens_are_not_inferred_as_valid(tracker):
    record = tracker.record("deepseek-flash", {**full_usage(), "prompt_cache_hit_tokens": -1})
    assert record["billing_status"] == "usage_invalid"
    assert record["cost_cny"] is None


def test_unknown_model_requires_price_config(tracker):
    record = tracker.record("deepseek-chat", full_usage())
    assert record["cost_cny"] is None
    assert record["billing_status"] == "price_unknown"


def test_configurable_off_peak_snapshot(tracker, tmp_path):
    import yaml
    config = tracker.config()
    config["period"] = "off_peak"
    path = tmp_path / "billing.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    tracker.config_path = path
    first = tracker.record("deepseek-flash", full_usage())
    config["models"]["deepseek-flash"]["off_peak"]["output"] = 99
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert tracker.records()[0]["cost_cny"] == first["cost_cny"]
    assert first["price_period"] == "off_peak"


def test_sync_uses_response_model_and_usage(client):
    response = SimpleNamespace(model="deepseek-flash", usage=full_usage(),
                               choices=[SimpleNamespace(message=SimpleNamespace(content="done"))])
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response)))
    assert client.chat("synthetic example") == "done"
    assert client.last_billing["model"] == "deepseek-flash"
    assert client.last_billing["billing_status"] == "estimated"


def test_stream_real_usage_only_counted_once(client):
    params = {}

    class Stream:
        def __iter__(self):
            yield SimpleNamespace(model="deepseek-flash", choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"))], usage=None)
            yield SimpleNamespace(model="deepseek-flash", choices=[], usage=full_usage())

        def close(self):
            self.closed = True

    stream = Stream()

    def create(**kwargs):
        params.update(kwargs)
        return stream

    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert "".join(client.stream_chat("synthetic example")) == "hi"
    assert params["stream_options"]["include_usage"] is True
    assert stream.closed
    assert client.billing.summary()["calls"] == 1
    assert client.last_billing["completion_tokens"] == 200


def test_stream_usage_can_arrive_with_final_content(client):
    class Stream:
        def __iter__(self):
            yield SimpleNamespace(model="deepseek-flash", choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"))], usage=full_usage())

        def close(self):
            pass

    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: Stream())))
    assert "".join(client.stream_chat("synthetic example")) == "hi"
    assert client.last_billing["cost_cny"] is not None


def test_interrupted_stream_keeps_unknown_fee(client):
    class Stream:
        def __iter__(self):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"))])
            raise RuntimeError("interrupted")

        def close(self):
            pass

    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: Stream())))
    with pytest.raises(RuntimeError):
        list(client.stream_chat("synthetic example"))
    assert client.billing.summary()["calls"] == 1
    assert client.last_billing["billing_status"] == "usage_missing"
    assert client.last_billing["status"] == "stream_interrupted"


def test_cancelled_stream_keeps_unknown_fee(client):
    class Stream:
        def __iter__(self):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"))])

        def close(self):
            pass

    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: Stream())))
    generator = client.stream_chat("synthetic example")
    assert next(generator) == "hi"
    generator.close()
    assert client.billing.summary()["calls"] == 1
    assert client.last_billing["billing_status"] == "usage_missing"


def test_public_demo_does_not_call_or_bill(client, monkeypatch):
    def fail(**kwargs):
        pytest.fail("public demo must not contact API")

    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    monkeypatch.setenv("APP_ENV", "public_demo")
    with pytest.raises(ValueError):
        client.chat("example")
    with pytest.raises(ValueError):
        list(client.stream_chat("example"))
    assert not client.billing_log_path.exists()


def test_accounting_error_does_not_discard_answer(client, tmp_path):
    client.billing.config_path = tmp_path / "missing.yaml"
    response = SimpleNamespace(model="deepseek-flash", usage=full_usage(),
                               choices=[SimpleNamespace(message=SimpleNamespace(content="done"))])
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response)))
    assert client.chat("example") == "done"
    assert client.billing.last_error
    assert client.last_billing["billing_status"] == "price_config_error"
    assert client.billing.summary()["pending_calls"] == 1


@pytest.mark.parametrize("broken_config", ["models: [", "- bad-root", "plain-string",
                                             "models: []", "models: {deepseek-flash: []}",
                                             "models: {deepseek-flash: {peak: []}}",
                                             "models: {deepseek-flash: {peak: {input_cache_hit: -1}}}"])
@pytest.mark.parametrize("stream", [False, True])
def test_bad_config_does_not_discard_successful_answer(client, tmp_path, broken_config, stream):
    path = tmp_path / "bad-billing.yaml"
    path.write_text(broken_config, encoding="utf-8")
    client.billing.config_path = path

    class Stream:
        def __iter__(self):
            yield SimpleNamespace(model="deepseek-flash", usage=full_usage(),
                                  choices=[SimpleNamespace(delta=SimpleNamespace(content="done"))])

        def close(self):
            pass

    response = Stream() if stream else SimpleNamespace(
        model="deepseek-flash", usage=full_usage(),
        choices=[SimpleNamespace(message=SimpleNamespace(content="done"))])
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response)))
    answer = "".join(client.stream_chat("example")) if stream else client.chat("example")
    assert answer == "done"
    assert client.last_billing["billing_status"] == "price_config_error"
    assert client.last_billing["cost_cny"] is None
    assert client.last_billing["prompt_tokens"] == 1000
    assert client.billing.summary()["pending_calls"] == 1
    assert client.billing.last_error
    with pytest.raises(ValueError):
        client.billing.config()


def test_audit_keeps_only_numeric_token_fields(client, tmp_path):
    import json
    client.config.safety.audit_log.enabled = True
    client.audit_log_path = tmp_path / "audit.log"
    usage = {**full_usage(), "provider_note": "private-provider-data",
             "completion_tokens_details": {"reasoning_tokens": 15, "provider_secret": "private-detail-data"},
             "prompt_tokens_details": {"cached_tokens": 400, "provider_note": "private-cache-data"}}
    response = SimpleNamespace(model="deepseek-flash", usage=usage,
                               choices=[SimpleNamespace(message=SimpleNamespace(content="done"))])
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response)))
    assert client.chat("private question") == "done"
    text = client.audit_log_path.read_text(encoding="utf-8")
    assert all(value not in text for value in ("private-provider-data", "private-detail-data", "private-cache-data", "private question"))
    audit = json.loads(text)
    assert audit["usage"]["completion_tokens_details"] == {"reasoning_tokens": 15}
    assert audit["usage"]["prompt_tokens_details"] == {"cached_tokens": 400}
    assert audit["usage"]["prompt_tokens"] == 1000


@pytest.mark.parametrize("public", [False, True])
def test_billing_page_protects_private_history(client, monkeypatch, public):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from app.components import runtime

    client.billing.record("deepseek-flash", full_usage())
    monkeypatch.setenv("BILLING_TEST_API_KEY", "")
    monkeypatch.setattr(runtime, "init_page", lambda *args: None)
    monkeypatch.setattr(runtime, "require_service", lambda: (None, {}))
    monkeypatch.setattr(runtime, "llm_tools", lambda: (client, None, client.config))
    monkeypatch.setattr(runtime, "saved_prediction", lambda: None)
    monkeypatch.setattr(runtime, "public_mode", lambda: public)
    if public:
        def private_read_forbidden(*args):
            pytest.fail("public page must not read the private billing ledger")

        monkeypatch.setattr(client.billing, "summary", private_read_forbidden)
        monkeypatch.setattr(client.billing, "records", private_read_forbidden)
    page = Path(__file__).resolve().parents[1] / "app/pages/6_💬_LLM智能助手.py"
    app = AppTest.from_file(str(page), default_timeout=30).run()
    assert not app.exception
    labels = [item.label for item in app.metric]
    assert ("示例费用" in labels) is public
    assert ("累计已估算费用" in labels) is not public
