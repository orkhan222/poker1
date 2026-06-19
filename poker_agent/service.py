from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from poker_agent.agents import MLPolicyAgent, RuleBasedAgent
from poker_agent.api_contract import api_contract
from poker_agent.schemas import PredictionRequest


app = FastAPI(
    title="Poker Decision Agent API",
    description="API for real-time poker action prediction using the bundled trained policy model.",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Prediction",
            "description": "Poker action prediction endpoints.",
        },
        {
            "name": "System",
            "description": "Service status and model health endpoints.",
        },
    ],
)
_agent = None
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "poker_policy.joblib"
OPTIONAL_BUNDLE_MODEL_PATH = PROJECT_ROOT / "models" / "poker_policy_bundle.joblib"
FALLBACK_MODEL_PATH = PROJECT_ROOT / "models" / "poker_policy.json"


APP_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Poker Decision Agent</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background: #0f1412;
      color: #eef4ef;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 20% 10%, rgba(59, 130, 246, 0.18), transparent 30rem),
        radial-gradient(circle at 86% 12%, rgba(34, 197, 94, 0.16), transparent 28rem),
        linear-gradient(135deg, #0f1412 0%, #17201c 50%, #101516 100%);
    }
    main {
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 36px 0;
    }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 24px;
      margin-bottom: 24px;
    }
    h1 {
      margin: 0;
      font-size: clamp(32px, 5vw, 56px);
      font-weight: 800;
      letter-spacing: 0;
    }
    .subtitle {
      max-width: 720px;
      margin: 12px 0 0;
      color: #d6e3db;
      font-size: 17px;
      line-height: 1.45;
    }
    .status {
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 8px;
      padding: 10px 14px;
      background: rgba(255, 255, 255, 0.06);
      color: #9fe6b3;
      white-space: nowrap;
      font-weight: 750;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(320px, 440px);
      gap: 18px;
    }
    section {
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 8px;
      background: rgba(16, 20, 18, 0.82);
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
    }
    .form-panel { padding: 22px; }
    .panel-title {
      margin: 0 0 16px;
      color: #f5fff8;
      font-size: 18px;
      font-weight: 850;
    }
    .result-panel {
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    label {
      display: grid;
      gap: 7px;
      color: #b6c7bd;
      font-size: 13px;
      font-weight: 650;
    }
    input, select {
      width: 100%;
      height: 42px;
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 6px;
      padding: 0 12px;
      background: #0f1714;
      color: #f4fff7;
      font: inherit;
      outline: none;
    }
    input:focus, select:focus {
      border-color: #6ee08c;
      box-shadow: 0 0 0 3px rgba(110, 224, 140, 0.16);
    }
    button {
      width: 100%;
      height: 46px;
      margin-top: 18px;
      border: 0;
      border-radius: 6px;
      background: #55c46f;
      color: #07110a;
      font-weight: 800;
      font-size: 15px;
      cursor: pointer;
    }
    button:disabled {
      cursor: wait;
      opacity: 0.65;
    }
    button:hover { background: #62d67c; }
    .action {
      min-height: 110px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      background: #e9fff0;
      color: #102016;
      font-size: clamp(38px, 6vw, 72px);
      font-weight: 900;
      text-transform: uppercase;
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      min-height: 72px;
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 8px;
      padding: 12px;
      background: rgba(255, 255, 255, 0.04);
    }
    .metric span {
      display: block;
      color: #9fb0a6;
      font-size: 12px;
      font-weight: 750;
    }
    .metric strong {
      display: block;
      margin-top: 7px;
      color: #f4fff7;
      font-size: 18px;
    }
    .bars {
      display: grid;
      gap: 10px;
    }
    .bar-row {
      display: grid;
      grid-template-columns: 76px 1fr 58px;
      align-items: center;
      gap: 10px;
      color: #d5e3d9;
      font-size: 14px;
    }
    .track {
      height: 12px;
      overflow: hidden;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.09);
    }
    .fill {
      height: 100%;
      width: 0%;
      border-radius: inherit;
      background: linear-gradient(90deg, #55c46f, #f0c14b);
      transition: width 180ms ease;
    }
    pre {
      overflow: auto;
      min-height: 120px;
      margin: 0;
      border-radius: 8px;
      padding: 14px;
      background: #0a0f0d;
      color: #cfe7d5;
      font-size: 12px;
      line-height: 1.5;
    }
    @media (max-width: 820px) {
      header, .layout { grid-template-columns: 1fr; }
      header { align-items: start; }
      .grid { grid-template-columns: 1fr; }
      .summary { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Poker Decision Agent</h1>
        <p class="subtitle">Real-time poker action prediction from game state inputs, backed by a trained policy model and exposed through a FastAPI service.</p>
      </div>
      <div class="status">Live API</div>
    </header>

    <div class="layout">
      <section class="form-panel">
        <h2 class="panel-title">Game State</h2>
        <form id="predict-form">
          <div class="grid">
            <label>Position
              <select name="position">
                <option>BTN</option>
                <option>SB</option>
                <option>BB</option>
                <option>UTG</option>
                <option>MP</option>
                <option>CO</option>
                <option>Player1_Bottom</option>
              </select>
            </label>
            <label>Street
              <select name="street">
                <option>preflop</option>
                <option>flop</option>
                <option>turn</option>
                <option>river</option>
              </select>
            </label>
            <label>Hole cards
              <input name="hole_cards" value="Ah Kd" aria-label="Hole cards, for example Ah Kd">
            </label>
            <label>Board cards
              <input name="board_cards" value="" aria-label="Board cards, for example 2c 7d Qs">
            </label>
            <label>Pot
              <input name="pot" type="number" step="0.1" value="2.5">
            </label>
            <label>To call
              <input name="to_call" type="number" step="0.1" value="1.0">
            </label>
            <label>Stack
              <input name="stack" type="number" step="0.1" value="100">
            </label>
            <label>Min raise
              <input name="min_raise" type="number" step="0.1" value="2.0">
            </label>
            <label>Players
              <input name="player_count" type="number" step="1" value="6">
            </label>
          </div>
          <button id="submit-button" type="submit">Predict action</button>
        </form>
      </section>

      <section class="result-panel">
        <div id="action" class="action">Ready</div>
        <div class="summary">
          <div class="metric"><span>Confidence</span><strong id="confidence">-</strong></div>
          <div class="metric"><span>Bet size</span><strong id="bet-size">-</strong></div>
          <div class="metric"><span>Wait time</span><strong id="wait-time">-</strong></div>
        </div>
        <div id="bars" class="bars"></div>
        <pre id="json-output">{}</pre>
      </section>
    </div>
  </main>

  <script>
    const form = document.getElementById("predict-form");
    const button = document.getElementById("submit-button");
    const actionBox = document.getElementById("action");
    const confidence = document.getElementById("confidence");
    const betSize = document.getElementById("bet-size");
    const waitTime = document.getElementById("wait-time");
    const bars = document.getElementById("bars");
    const output = document.getElementById("json-output");

    function cards(value) {
      return value.split(/[ ,]+/).map((card) => card.trim()).filter(Boolean);
    }

    function numberValue(data, name) {
      return Number(data.get(name) || 0);
    }

    function render(result, payload) {
      actionBox.textContent = result.action || "N/A";
      const probabilities = Object.values(result.probabilities || {});
      const topProbability = probabilities.length ? Math.max(...probabilities) : 0;
      confidence.textContent = `${(topProbability * 100).toFixed(1)}%`;
      betSize.textContent = Number(result.bet_size || 0).toFixed(2);
      waitTime.textContent = result.wait_time_ms ? `${result.wait_time_ms} ms` : "-";
      output.textContent = JSON.stringify(result, null, 2);
      bars.innerHTML = "";
      Object.entries(result.probabilities || {})
        .sort((a, b) => b[1] - a[1])
        .forEach(([name, value]) => {
          const row = document.createElement("div");
          row.className = "bar-row";
          row.innerHTML = `
            <strong>${name}</strong>
            <div class="track"><div class="fill" style="width:${Math.round(value * 100)}%"></div></div>
            <span>${(value * 100).toFixed(1)}%</span>
          `;
          bars.appendChild(row);
        });
    }

    async function predict(event) {
      event.preventDefault();
      button.disabled = true;
      button.textContent = "Predicting...";
      const data = new FormData(form);
      const payload = {
        position: data.get("position"),
        street: data.get("street"),
        hole_cards: cards(data.get("hole_cards") || ""),
        board_cards: cards(data.get("board_cards") || ""),
        pot: numberValue(data, "pot"),
        to_call: numberValue(data, "to_call"),
        stack: numberValue(data, "stack"),
        min_raise: numberValue(data, "min_raise"),
        player_count: Number(data.get("player_count") || 6)
      };

      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        render(await response.json(), payload);
      } catch (error) {
        actionBox.textContent = "Error";
        output.textContent = String(error);
      } finally {
        button.disabled = false;
        button.textContent = "Predict action";
      }
    }

    form.addEventListener("submit", predict);
    form.dispatchEvent(new Event("submit"));
  </script>
</body>
</html>
"""


def health_payload() -> dict[str, str]:
    model_path = resolve_model_path()
    payload = {
        "status": "ok",
        "model": str(model_path),
        "model_status": "loaded" if model_path.exists() else "fallback_rule_based",
    }
    try:
        agent = get_agent()
        model = getattr(agent, "model", None)
        metadata = getattr(model, "metadata", {}) or {}
        if metadata:
            payload["policy"] = str(metadata.get("policy", getattr(model, "model_kind", "unknown")))
            payload["split"] = str((metadata.get("split") or {}).get("split_type", "unknown"))
            valid_metrics = metadata.get("valid_metrics") or {}
            if "macro_f1" in valid_metrics:
                payload["valid_macro_f1"] = f"{float(valid_metrics['macro_f1']):.4f}"
    except Exception:
        payload["metadata_status"] = "unavailable"
    return payload


def health_html(payload: dict[str, str]) -> str:
    status = payload["status"].upper()
    model_status = payload["model_status"].replace("_", " ")
    model_path = payload["model"]
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Poker Decision Agent Status</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background: #0f1412;
      color: #eef4ef;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
      background:
        radial-gradient(circle at 20% 12%, rgba(59, 130, 246, 0.18), transparent 30rem),
        radial-gradient(circle at 85% 18%, rgba(34, 197, 94, 0.16), transparent 28rem),
        linear-gradient(135deg, #0f1412 0%, #17201c 50%, #101516 100%);
    }}
    main {{
      width: min(760px, 100%);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 8px;
      padding: 28px;
      background: rgba(16, 20, 18, 0.88);
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: start;
      gap: 18px;
      margin-bottom: 22px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(30px, 5vw, 44px);
      letter-spacing: 0;
    }}
    .badge {{
      border-radius: 999px;
      padding: 8px 12px;
      background: #e9fff0;
      color: #102016;
      font-weight: 850;
      white-space: nowrap;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .item {{
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 8px;
      padding: 14px;
      background: rgba(255, 255, 255, 0.04);
    }}
    .item span {{
      display: block;
      color: #9fb0a6;
      font-size: 12px;
      font-weight: 750;
      text-transform: uppercase;
    }}
    .item strong {{
      display: block;
      margin-top: 8px;
      overflow-wrap: anywhere;
      color: #f4fff7;
      font-size: 17px;
    }}
    .model {{
      grid-column: 1 / -1;
    }}
    nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 22px;
    }}
    a {{
      border-radius: 6px;
      padding: 10px 13px;
      background: #55c46f;
      color: #07110a;
      text-decoration: none;
      font-weight: 850;
    }}
    a.secondary {{
      border: 1px solid rgba(255, 255, 255, 0.14);
      background: rgba(255, 255, 255, 0.06);
      color: #eef4ef;
    }}
    @media (max-width: 640px) {{
      header, .grid {{ grid-template-columns: 1fr; }}
      header {{ display: grid; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Service Status</h1>
      </div>
      <div class="badge">{status}</div>
    </header>
    <section class="grid">
      <div class="item">
        <span>API</span>
        <strong>{payload["status"]}</strong>
      </div>
      <div class="item">
        <span>Model status</span>
        <strong>{model_status}</strong>
      </div>
      <div class="item model">
        <span>Model path</span>
        <strong>{model_path}</strong>
      </div>
    </section>
    <nav>
      <a href="/predict">Open application</a>
      <a class="secondary" href="/docs">API docs</a>
      <a class="secondary" href="/health.json">Raw JSON</a>
      <a class="secondary" href="/contract.json">Contract</a>
    </nav>
  </main>
</body>
</html>
"""


def get_agent():
    global _agent
    if _agent is not None:
        return _agent
    model_path = resolve_model_path()
    if model_path.exists():
        _agent = MLPolicyAgent.from_path(model_path)
    else:
        _agent = RuleBasedAgent()
    return _agent


def resolve_model_path() -> Path:
    configured = os.getenv("POKER_POLICY_PATH")
    if configured:
        return Path(configured)
    if OPTIONAL_BUNDLE_MODEL_PATH.exists():
        return OPTIONAL_BUNDLE_MODEL_PATH
    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH
    return FALLBACK_MODEL_PATH


@app.get("/health", include_in_schema=False)
def health(request: Request) -> Any:
    payload = health_payload()
    accept = request.headers.get("accept", "")
    if "text/html" in accept and "application/json" not in accept:
        return HTMLResponse(health_html(payload))
    return payload


@app.get(
    "/health.json",
    tags=["System"],
    summary="Service status",
    description="Returns API status and confirms whether the bundled policy model is loaded.",
)
def health_json() -> dict[str, str]:
    return health_payload()


@app.get(
    "/contract.json",
    tags=["System"],
    summary="API response contract",
    description="Returns the field-level contract for prediction responses and delivery status terms.",
)
def contract_json() -> dict[str, Any]:
    return api_contract()


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home_page() -> str:
    return APP_HTML


@app.get("/predict", response_class=HTMLResponse, include_in_schema=False)
def predict_page() -> str:
    return APP_HTML


@app.post(
    "/predict",
    tags=["Prediction"],
    summary="Predict poker action",
    description="Accepts a poker game state and returns the selected action with action probabilities.",
)
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    request = PredictionRequest.from_dict(payload)
    return get_agent().predict(request).to_dict()
