"""Parse natural language strategy description into structured rules using GPT."""
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set in .env")
        _client = OpenAI(api_key=key)
    return _client


SYSTEM_PROMPT = """You translate trading strategy descriptions into JSON for backtesting.

TWO MODES — pick the right one based on description:

═══ MODE A: INDICATOR-BASED RULES (math conditions on indicators) ═══

Use when description mentions things like RSI, moving averages, MACD, Bollinger bands,
crossovers, levels, thresholds.

Schema:
{
  "indicators": [{"name": "<unique>", "type": "<type>", "period": <int>, ...extra}],
  "entry_long": "<expression>",
  "exit_long": "<expression>",
  "entry_short": "<expression> or null",
  "exit_short": "<expression> or null",
  "stop_loss_pct": <float 0-1> or null,
  "take_profit_pct": <float 0-1> or null
}

INDICATOR TYPES (lowercase): rsi, sma, ema, atr, macd, stoch, bbands, adx
- rsi: period (default 14)
- sma/ema: period
- atr: period (default 14)
- macd: fast, slow, signal, component ("line"|"signal"|"hist")
- stoch: k, d, component ("k"|"d")
- bbands: period, std, component ("upper"|"middle"|"lower")
- adx: period

CRITICAL — NEVER use period: null. Always pick concrete numeric value.
When description uses words instead of numbers, use these defaults:
- "fast EMA" / "fast MA" → period 9
- "slow EMA" / "slow MA" → period 21
- "short term" → 9 or 14
- "medium term" → 20 or 50
- "long term" → 100 or 200
- "RSI" alone → period 14
- "MACD" alone → fast=12, slow=26, signal=9
- "Bollinger" alone → period 20, std 2.0

When description mentions "auto stop loss" or "with SL/TP" but no numbers, default:
- stop_loss_pct: 0.05 (5%)
- take_profit_pct: 0.10 (10%)

When entry/exit logic implies the opposite signal terminates positions but isn't explicit
("LONG on X, SHORT on opposite") — set exit_long = entry_short expression and vice versa.

EXPRESSION SYNTAX: < > <= >= == != and or — columns: open, high, low, close, volume + indicator names.

═══ MODE B: CHART PATTERN DETECTION ═══

Use when description mentions chart patterns by name: double top, double bottom,
head and shoulders, triangles, flags, wedges, etc.

Schema:
{
  "pattern": {
    "type": "<one of supported types>",
    "params": <optional dict of parameter overrides>
  }
}

SUPPORTED PATTERNS:
- "double_top" — bearish reversal, 2 peaks at similar height
- "double_bottom" — bullish reversal, 2 troughs at similar height
- "head_and_shoulders" — bearish reversal, 3 peaks (middle highest)
- "inverse_head_and_shoulders" — bullish reversal, 3 troughs (middle lowest)
- "ascending_triangle" — bullish continuation, flat top + rising bottom
- "descending_triangle" — bearish continuation, falling top + flat bottom
- "symmetrical_triangle" — neutral, converging trendlines
- "bullish_flag" — bullish continuation, strong up-move + tight consolidation
- "bearish_flag" — bearish continuation, strong down-move + tight consolidation

═══ EXAMPLES ═══

Input: "Buy when RSI(14) drops below 30, sell at RSI 70. 5% stop loss."
Output:
{
  "indicators": [{"name":"rsi_14","type":"rsi","period":14}],
  "entry_long": "rsi_14 < 30",
  "exit_long": "rsi_14 > 70",
  "entry_short": null,
  "exit_short": null,
  "stop_loss_pct": 0.05,
  "take_profit_pct": null
}

Input: "Golden cross: long when 50 SMA above 200 SMA, exit when reversed."
Output:
{
  "indicators": [
    {"name":"sma_50","type":"sma","period":50},
    {"name":"sma_200","type":"sma","period":200}
  ],
  "entry_long": "sma_50 > sma_200",
  "exit_long": "sma_50 < sma_200",
  "entry_short": null,
  "exit_short": null,
  "stop_loss_pct": null,
  "take_profit_pct": null
}

Input: "Trade double tops — short when pattern confirms."
Output:
{
  "pattern": {"type": "double_top"}
}

Input: "Bullish flag breakout strategy."
Output:
{
  "pattern": {"type": "bullish_flag"}
}

Input: "Head and shoulders reversal short."
Output:
{
  "pattern": {"type": "head_and_shoulders"}
}

Input: "Ascending triangle breakout, long on confirmation above resistance."
Output:
{
  "pattern": {"type": "ascending_triangle"}
}

═══ RULES ═══

1. If description names a chart pattern → MODE B (pattern).
2. If description gives math conditions on indicators → MODE A (indicators).
3. If both — prefer MODE B (pattern) since pattern detection includes its own SL/TP via measured-move.
4. Always reference indicator column names exactly as declared in "indicators" array.
5. ALWAYS try to produce sensible rules. For vague input, make reasonable interpretation:
   - "buy low sell high" → RSI mean reversion (RSI<30 buy, RSI>70 sell)
   - "trend following" → EMA crossover (9/21)
   - "breakout strategy" → 20-bar high breakout (use sma_20 of high as proxy)
   - "scalping" → tight RSI extremes (RSI<25 buy, RSI>75 sell, tight SL 1%)
   - Russian/Ukrainian/other-language descriptions → translate and parse
6. Only return {"error": "..."} if input is COMPLETELY empty or has zero trading content (e.g., "hello", "test", "asdf").
7. Return ONLY valid JSON. No commentary, no markdown.
"""


def parse_strategy(text: str, model: str = "gpt-4o-mini") -> dict:
    """Convert NL strategy description to structured rules dict."""
    response = client().chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.1,
    )
    raw = response.choices[0].message.content
    rules = json.loads(raw)

    if "error" in rules:
        raise ValueError(f"LLM couldn't parse: {rules['error']}")

    # Dispatch validation: pattern mode OR indicators mode
    if "pattern" in rules:
        pat = rules["pattern"]
        if not isinstance(pat, dict) or "type" not in pat:
            raise ValueError(f"pattern mode missing 'type' field. Got: {pat}")
    else:
        required = ["indicators", "entry_long", "exit_long"]
        missing = [k for k in required if k not in rules]
        if missing:
            raise ValueError(f"indicator mode missing keys: {missing}. Got: {list(rules.keys())}")

    return rules


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "Buy when RSI(14) drops below 30, sell when above 70"
    print(f"Parsing: {text!r}\n")
    rules = parse_strategy(text)
    print(json.dumps(rules, indent=2))
