# tools/evaluate_run.py
from pathlib import Path
import json, csv, datetime as dt

# ─── CONFIGURABLE CONSTANTS ───────────────────────────────────────────
LOG_DIR = Path("logs")            # folder backend writes into

MANUAL_MINUTES = 145              # human drafting time (mins)
HOURLY_RATE    = 45               # $/hour fully‑loaded cost

# GPT‑4o‑mini prices (May‑2025)
OPENAI_INPUT_COST    = 0.30 / 1_000_000
OPENAI_CACHED_COST   = 0.15 / 1_000_000
OPENAI_OUTPUT_COST   = 1.20 / 1_000_000
OPENAI_TRAINING_COST = 3.00 / 1_000_000   # rarely used in inference

# fallback averages per call
AVG_TOKENS_IN  = 700
AVG_TOKENS_OUT = 900

# Pinecone pricing
RUN_UNIT_COST       = 0.096
PINECONE_PLAN       = "starter"          # starter | standard | enterprise
MONTHLY_RUN_VOLUME  = 60
# ───────────────────────────────────────────────────────────────────────

OUT_CSV = "metrics.csv"

def read_iso(p: Path) -> dt.datetime:
    return dt.datetime.fromisoformat(p.read_text().strip())

def pinecone_monthly_cost(vol: int) -> float:
    raw = vol * RUN_UNIT_COST
    if PINECONE_PLAN == "starter":
        return raw
    if PINECONE_PLAN == "standard":     # $25 fee, $15 credit
        return max(0, raw - 15) + 25
    if PINECONE_PLAN == "enterprise":   # $500 fee, $150 credit
        return max(0, raw - 150) + 500
    raise ValueError("Unknown PINECONE_PLAN")

# ─── MAIN ─────────────────────────────────────────────────────────────
def main():
    # 1) timing --------------------------------------------------------
    start_f = LOG_DIR / "pipeline_start.txt"
    end_f   = LOG_DIR / "pipeline_end.txt"
    if not (start_f.exists() and end_f.exists()):
        raise FileNotFoundError("Run pipeline first – start/end txt missing")
    pipeline_min = (read_iso(end_f) - read_iso(start_f)).total_seconds() / 60

    # 2) labour cost ---------------------------------------------------
    manual_cost   = MANUAL_MINUTES / 60 * HOURLY_RATE
    auto_cost_lab = pipeline_min   / 60 * HOURLY_RATE

    # 3) OpenAI token usage (real totals if present) -------------------
    TOKEN_FILE = LOG_DIR / "token_usage.json"
    if TOKEN_FILE.exists():
        usage_totals = json.loads(TOKEN_FILE.read_text())
        fresh_prompt_tokens  = usage_totals["fresh_prompt"]
        cached_prompt_tokens = usage_totals["cached_prompt"]
        completion_tokens    = usage_totals["completion"]
        training_tokens      = usage_totals["training"]
    else:
        # fall back to averages × llm_calls
        ag_status = json.loads((LOG_DIR / "agent_status.json").read_text())
        llm_calls = ag_status.get("llm_usage_count") or 0
        fresh_prompt_tokens  = AVG_TOKENS_IN  * llm_calls
        cached_prompt_tokens = 0
        completion_tokens    = AVG_TOKENS_OUT * llm_calls
        training_tokens      = 0

    openai_cost = (
          fresh_prompt_tokens  * OPENAI_INPUT_COST
        + cached_prompt_tokens * OPENAI_CACHED_COST
        + completion_tokens    * OPENAI_OUTPUT_COST
        + training_tokens      * OPENAI_TRAINING_COST
    )

    # 4) Pinecone cost per run ----------------------------------------
    pinecone_cost = pinecone_monthly_cost(MONTHLY_RUN_VOLUME) / MONTHLY_RUN_VOLUME

    total_auto_cost = auto_cost_lab + openai_cost + pinecone_cost

    # 5) deltas --------------------------------------------------------
    time_saved    = MANUAL_MINUTES - pipeline_min
    percent_saved = time_saved / MANUAL_MINUTES * 100
    cost_saved    = manual_cost - total_auto_cost
    percent_cost  = cost_saved / manual_cost * 100

    # 6) console summary ----------------------------------------------
    print("─── SUMMARY ─────────────────────────────────")
    print(f"Pipeline minutes   : {pipeline_min:.1f}")
    print(f"Manual minutes     : {MANUAL_MINUTES}")
    print(f"Time saved         : {time_saved:.1f}  ({percent_saved:.0f} %)")
    print(f"OpenAI cost ($)    : {openai_cost:,.2f}")
    print(f"Pinecone cost ($)  : {pinecone_cost:,.2f}  ({PINECONE_PLAN} plan)")
    print(f"Total auto cost ($): {total_auto_cost:,.2f}")
    print(f"Cost saved ($)     : {cost_saved:,.2f}  ({percent_cost:.0f} %)")
    print("CSV saved →", OUT_CSV)
    print("──────────────────────────────────────────────")

    # 7) CSV export ----------------------------------------------------
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Metric", "Manual", "AI pipeline", "Δ", "% saving"])
        w.writerow(["Time to draft (min)",
                    MANUAL_MINUTES, f"{pipeline_min:.1f}",
                    f"{-time_saved:.1f}", f"{percent_saved:.0f} %"])
        w.writerow(["Labour cost ($)",
                    f"{manual_cost:.2f}", f"{auto_cost_lab:.2f}",
                    f"{manual_cost - auto_cost_lab:.2f}",
                    f"{(manual_cost - auto_cost_lab) / manual_cost * 100:.0f} %"])
        w.writerow(["OpenAI usage ($)", "0", f"{openai_cost:.2f}",
                    f"{openai_cost:.2f}", "—"])
        w.writerow(["Pinecone usage ($)", "0", f"{pinecone_cost:.2f}",
                    f"{pinecone_cost:.2f}", "—"])
        w.writerow(["Total cost ($)",
                    f"{manual_cost:.2f}", f"{total_auto_cost:.2f}",
                    f"{-cost_saved:.2f}", f"{percent_cost:.0f} %"])

if __name__ == "__main__":
    main()
