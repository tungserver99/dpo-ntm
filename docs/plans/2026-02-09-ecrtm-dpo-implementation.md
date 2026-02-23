# ECRTM DPO Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add end-to-end LLM-guided preference learning (DPO) to ECRTM with automatic preference generation and DPO fine-tuning in one run.

**Architecture:** Keep ECRTM unchanged except for exposing beta logits. Add a DPO pipeline with (1) snapshot & artifacts at epoch E, (2) LLM+embedding preference builder producing JSONL artifacts, (3) DPO loss computed against frozen beta_ref logits and applied only to filtered topics.

**Tech Stack:** PyTorch, sentence-transformers (`all-mpnet-base-v2`), OpenAI API (gpt-4o, function calling), JSONL artifacts, existing evaluation utils.

### Task 1: Add DPO arguments and wiring flags

**Files:**
- Modify: `utils/config.py`

**Step 1: Write the failing test**

Create `tests/test_config_dpo_args.py` to assert DPO args parse with defaults.

```python
import utils.config as config

def test_dpo_args_defaults():
    parser = config.new_parser()
    config.add_training_argument(parser)
    args = parser.parse_args([])
    assert hasattr(args, "dpo_start_epoch")
    assert hasattr(args, "dpo_weight")
    assert hasattr(args, "dpo_alpha")
    assert hasattr(args, "dpo_topic_filter")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_dpo_args.py -v`  
Expected: FAIL (missing args)

**Step 3: Write minimal implementation**

Add args in `utils/config.py`:
- `--enable_dpo` (default False)
- `--dpo_start_epoch` (int)
- `--dpo_weight` (float)
- `--dpo_alpha` (float, default 1.0)
- `--dpo_topic_filter` (already exists)

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config_dpo_args.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add utils/config.py tests/test_config_dpo_args.py
git commit -m "feat: add DPO training arguments"
```

### Task 2: Expose beta logits in ECRTM

**Files:**
- Modify: `models/ECRTM/ECRTM.py`
- Test: `tests/test_ecrtm_beta_logits.py`

**Step 1: Write the failing test**

```python
import torch
from models.ECRTM.ECRTM import ECRTM

def test_beta_logits_shape():
    model = ECRTM(vocab_size=10, num_topics=3)
    logits = model.get_beta_logits()
    assert logits.shape == (3, 10)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ecrtm_beta_logits.py -v`  
Expected: FAIL (missing method)

**Step 3: Write minimal implementation**

Add `get_beta_logits()` returning `-dist / self.beta_temp` (shape KxV).

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ecrtm_beta_logits.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add models/ECRTM/ECRTM.py tests/test_ecrtm_beta_logits.py
git commit -m "feat: expose ECRTM beta logits"
```

### Task 3: Create DPO loss module

**Files:**
- Create: `dpo/dpo_loss.py`
- Test: `tests/test_dpo_loss.py`

**Step 1: Write the failing test**

```python
import torch
from dpo.dpo_loss import dpo_loss

def test_dpo_loss_basic():
    beta = torch.tensor([[2.0, 1.0, 0.0],
                         [0.5, 1.0, 1.5]])
    beta_ref = torch.tensor([[1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0]])
    prefs = {0: {"w_win": [0], "w_loose": [2]}}
    loss = dpo_loss(beta, beta_ref, prefs, alpha=1.0)
    assert loss.item() > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dpo_loss.py -v`  
Expected: FAIL (missing module)

**Step 3: Write minimal implementation**

Implement vectorized loss using log-sigmoid on:
`alpha * ((beta_k[w+] - beta_k[w-]) - (beta_ref_k[w+] - beta_ref_k[w-]))`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_dpo_loss.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add dpo/dpo_loss.py tests/test_dpo_loss.py
git commit -m "feat: add DPO loss module"
```

### Task 4: Build preference pipeline (LLM + embeddings + JSONL)

**Files:**
- Create: `dpo/preference_builder.py`
- Create: `dpo/llm_client.py`
- Create: `dpo/jsonl_io.py`
- Test: `tests/test_preference_builder_io.py`

**Step 1: Write the failing test**

Test JSONL schema writing for top_words_10/15/20/25 and preferences.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_preference_builder_io.py -v`  
Expected: FAIL (missing modules)

**Step 3: Write minimal implementation**

Implement:
- LLM scoring (topic score 1–3) via function calling
- LLM topic description with diversity for score 1
- JSONL writer for top_words_10/15/20/25
- Embedding-based extra 5 words using `all-mpnet-base-v2`
- LLM preference dataset builder
- Resume/skip if JSONL exists

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_preference_builder_io.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add dpo/preference_builder.py dpo/llm_client.py dpo/jsonl_io.py tests/test_preference_builder_io.py
git commit -m "feat: add preference builder pipeline"
```

### Task 5: Integrate DPO into training loop (ECRTM only)

**Files:**
- Modify: `basic_trainer.py`
- Modify: `main.py`
- Modify: `evaluate.py` (optional: for CV cache)
- Test: `tests/test_training_dpo_hook.py`

**Step 1: Write the failing test**

Stub a trainer run to verify:
- snapshot at `dpo_start_epoch`
- `beta_ref_logits.npy` saved
- DPO loss added after snapshot

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_training_dpo_hook.py -v`  
Expected: FAIL (missing hooks)

**Step 3: Write minimal implementation**

Implement:
- Snapshot at epoch E (checkpoint + beta_ref_logits + top_words)
- Auto-call preference builder
- Compute topic filter (CV below avg / llm_score / either)
- Add DPO loss to batch loss for ECRTM only

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_training_dpo_hook.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add basic_trainer.py main.py evaluate.py tests/test_training_dpo_hook.py
git commit -m "feat: integrate DPO into ECRTM training"
```

### Task 6: Smoke run and docs

**Files:**
- Modify: `README.md`

**Step 1: Add usage docs**

Document new args and end-to-end flow in `README.md`.

**Step 2: Run a smoke check (optional)**

Run: `python main.py --model ECRTM --dataset <small> --enable_dpo ...`  
Expected: training + preference build + DPO phases complete.

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document DPO pipeline usage"
```
