### Gaia agent traces synthesis

Standalone ReAct trace collection code for:
- **GAIA** (text-only subset; file-based tasks filtered out)
- **BBH (modified)** (MCQ → open-ended “GAIA-style”, with `\\boxed{}` answer instruction)
- **Reasoning Gym** (procedural task generator)

This repo supports:
- **`--sources bbh`**
- **`--sources gaia`**
- **`--sources reasoning-gym`**

### Install

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras:

```bash
pip install -e ".[e2b]"
pip install -e ".[browser]"
```

### Environment variables

- **Always required**: `OPENROUTER_API_KEY`
- **If using web_search/web_contents**: `EXA_API_KEY`
- **If using execute_python**: `E2B_API_KEY`
- **If using browser tool**: `BROWSERBASE_API_KEY`, `BROWSERBASE_PROJECT_ID`
- **For GAIA** (gated HF dataset): `HF_TOKEN` (must accept dataset terms on HF)

### Run

Example (Reasoning Gym via config file), from repo root:

```bash
python collect_traces.py \
  --sources reasoning-gym \
  --rg-config configs/rg_all_tasks_seed256_n624.yaml \
  --rg-root /home/ubuntu/workspace/reasoning-gym \
  --rg-size 624 \
  --model google/gemini-3-flash-preview \
  --reasoning-effort low \
  --max-tokens 100000 \
  --concurrency 16 \
  --max-steps 30 \
  --allowed-tools all
```

Outputs are written to `runs/collect/<run_id>/`:
- `traces/` (per-task JSON traces; filenames are sanitized by replacing `:` with `-`)
- `reasoning/` (optional **hidden reasoning archive** per task; not included in traces)
- `tasks.jsonl`, `results.jsonl`, `summary.json`, `config.json`

