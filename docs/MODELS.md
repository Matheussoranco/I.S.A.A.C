# Choosing a model

I.S.A.A.C. is local-first. The question that decides how well it works is not
"which framework" but "which model, on the GPU you actually have". This page
gives four rungs, what each one costs, and — for tool calling, the thing that
most often breaks small-model agents — what was measured rather than assumed.

```bash
isaac models list             # the ladder
isaac models recommend        # picks a rung from your VRAM
isaac models show good        # settings + env block for one rung
isaac models use good         # prints the .env block to paste
```

---

## The ladder

| Preset | Model | VRAM | Constrained decoding | Test-time samples |
|---|---|---|---|---|
| `minimal` | `gemma3:1b` | ~1.5 GB | **yes** (required) | 3 |
| `small` | `qwen3.5:2b` | ~2.5 GB | no | 3 |
| `good` | `nemotron-3-nano:4b` | ~4 GB | no | 3 |
| `better` | `ornith:9b` | ~7 GB | no | 3 |
| `best` | `claude-sonnet-5` (API) | — | no | 1 |

A preset pins more than a model name. It also sets the loop settings that model
needs: a 1B model with no tool-calling support needs a grammar to be usable at
all, while a frontier model is only slowed down by the same machinery. That is
why `test_time_samples` drops to 1 at the top of the ladder — spend the budget
on a stronger model instead of on resampling a weaker one.

`best` is the only rung that leaves your machine. It needs `ANTHROPIC_API_KEY`
and sends task content to a third party. `isaac models recommend` will not
suggest it when a local rung fits, even if a key is present.

---

## Tool-call reliability, measured

An agent is only as good as its ability to call tools. The failure mode that
matters on small models is a *malformed* call: the model picks the right tool,
then emits the call as prose or a fenced code block instead of through the
provider's function-calling channel. Before 1.4.0 the loop treated that as a
final answer and stopped, handing the user a raw JSON blob.

`isaac eval-toolcalls` measures it. Twenty prompts, each with exactly one
correct tool call, across five tools:

```bash
isaac eval-toolcalls --model nemotron-3-nano:4b --mode repair
```

**Malformed rate** is attempted calls that did not arrive natively, over all
attempted calls. Choosing the *wrong* tool is a reasoning error and is counted
separately, so it never flatters the formatting number.

### Results

Measured 2026-07-29 on an RTX 3050 6 GB laptop GPU, Ollama 0.32.5,
temperature 0.2, 20 cases per model.

| Model | Params | Ollama `tools` | Requests accepted | Malformed rate | Correct tool | Median latency |
|---|---|---|---|---|---|---|
| `nemotron-3-nano:4b` | 4 B | yes | 20/20 | **0.0 %** (0/20) | 20/20 (100 %) | 8.4 s |
| `qwen3.5:2b` | 2 B | yes | 19/20 | **0.0 %** (0/19) | 19/20 (95 %) | 25.0 s |
| `gemma3:1b` — native | 1 B | **no** | **0/20** | — | 0/20 (0 %) | — |
| `gemma3:1b` — constrained | 1 B | no | 20/20 | **0.0 %** (0/20) | 8/20 (40 %) | 2.4 s |

The single `qwen3.5:2b` miss was a provider-side error
(`XML syntax error … unexpected EOF`), not a malformed call.

### What this shows

**On tools-capable local models the malformed-call rate is already zero.**
Across 39 measured attempts on two models, not one call arrived outside the
native channel. The repair layer never fired. That is a real result and it is
worth stating plainly: if you run a modern Ollama model that advertises the
`tools` capability, salvage and Reflexion are a safety net you are unlikely to
need, and 1.4.0 will not make your agent visibly better.

**The gap is models with no tool-calling support at all.** `gemma3:1b` reports
only the `completion` capability, and Ollama rejects every tools-bearing
request with HTTP 400 — `does not support tools`. Repair cannot help, because
the request never reaches the model. Under constrained decoding the tools are
never bound, the decoder is held to the envelope grammar, and all 20 requests
succeed with 20 well-formed calls. That is the capability unlock: **0/20 → 20/20
executable requests**, from a model that previously could not act at all.

**A well-formed call is not a correct one.** The same 1B model picks the right
tool 8 times in 20. The grammar guarantees shape, not judgement, and no amount
of decoding constraint will make a 1B model reason like a 4B one. The suite
scores tool *choice* separately from tool *format* precisely so this cannot be
hidden — a release note claiming "100 % well-formed" would be true and
misleading at once.

**Constraining the schema per tool matters more than expected.** With a flat
envelope (tool-name enum, generic `arguments` object) `gemma3:1b` produced only
3 executable calls out of 20 — it picked a valid tool and then invented the
argument keys. Branching the schema per tool, so each arm carries that tool's
own parameter schema, raised that to 8/20 with identical tool-choice accuracy:
every correctly-chosen tool now gets correct arguments (8/8 vs 3/8). This
contradicted the initial design assumption that small models cope badly with
branched schemas, so `per_tool=True` is now the default.

---

## Recovering from malformed calls

Three layers, cheapest first. All are on by default except the third.

1. **Repair** (`ISAAC_REPAIR_TOOL_CALLS=1`, default on). Pure text parsing, no
   extra LLM call. Recognises the dialects small models actually emit: fenced
   JSON, Hermes/Qwen `<tool_call>` tags, `args`/`parameters`/`input` spellings,
   double-encoded argument strings, Python dict syntax, trailing commas, and
   `tool(arg="x")` call expressions. Gated on the bound tool names, so prose
   that merely contains braces is never mistaken for a call.

2. **Reflexion** (`ISAAC_REFLEXION_RETRIES=2`). When the text is unparseable
   but clearly *was* an attempted call, the model is shown its own broken
   output plus the contract and asked again. Budgeted per run, because a model
   that cannot correct in two tries will not manage it in ten.

3. **Constrained decoding** (`ISAAC_CONSTRAINED_DECODING=1`, default off).
   Bypasses native function calling entirely and constrains the decoder to a
   JSON envelope — via Ollama's `format` JSON-Schema field, or a GBNF grammar
   on llama.cpp. A malformed call becomes unrepresentable rather than
   recoverable. Costs the model its free-form reasoning channel, so it is the
   right default only where native tool calling is unavailable or unreliable.

   ```json
   {"tool": "web_search", "arguments": {"query": "…"}}
   {"tool": "none", "final_answer": "…"}
   ```

When a provider exposes no constraint channel, the envelope is still requested
in the prompt and still parsed — but it is not *enforced*, and the loop logs a
warning rather than letting you assume a guarantee you do not have.

---

## Test-time compute for hard steps

Small models fail on hard steps in a way more compute can fix. ISAAC reuses the
escalation the ARC solver already uses — try cheap, verify, escalate, exit as
soon as something passes:

1. One greedy sample. Verify it. Most steps stop here for one LLM call.
2. **Best-of-N** — resample until a cheap verifier accepts. Exits on the first
   pass, so the common case costs one sample, not *n*.
3. **Self-consistency** — with no verifier available, sample *n* times at
   non-zero temperature and take the majority answer.

Set `ISAAC_TEST_TIME_SAMPLES` (a preset does this for you). `1` is the
pre-1.4.0 single-shot behaviour. Code synthesis uses this today, with a syntax
check as the verifier: a model that emits unparseable Python gets resampled
before the sandbox round-trip rather than after it.

Verifiers must be *cheap* — a parse, a compile, a schema match, a range check.
One that costs an LLM call spends the budget it exists to save. The stock set
is in `isaac.reasoning.verifiers`. They check well-formedness, not truth: valid
Python can still be wrong, which is why the ladder falls back to agreement
voting when the verifier never accepts.

---

## Reproducing these numbers

```bash
ollama serve
ollama pull nemotron-3-nano:4b
isaac eval-toolcalls --model nemotron-3-nano:4b --mode repair --out report.json
```

`--mode native` scores the same prompts under the 1.3.x policy, where a
salvageable call still counts as a failure. `--no-reflexion` isolates how much
the parser recovers without corrective retries. Results depend on the model
revision, the quantisation Ollama pulled, and the sampling temperature — treat
numbers from a different setup as unverified.
