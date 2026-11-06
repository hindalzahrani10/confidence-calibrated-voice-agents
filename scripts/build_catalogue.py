"""Build the public candidate-task catalogue the agent reasons over.

PRIVATE-SIDE tool: it reads the hash-checked Phase 1 snapshot to pool the slot-value inventory
per (intent, slot-name set), then writes only a role-free list of tasks. Gold/alternative roles,
item ids and splits are not written, and the agent-side code loads only the output.

Leakage note: the pooled inventory necessarily contains every item's gold task, so the agent's
hypothesis space always includes the truth. This is label-space (not instance-level) knowledge,
shared identically by all policies, and is reported as a threat to validity.
"""
import hashlib, itertools, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "data/phase1/snapshots/phase1.v2.0+05471eebec39"
OUT = ROOT / "data/phase3/catalogue.v1.json"

for line in (SNAP / "SHA256SUMS").read_text().splitlines():
    h, name = line.split(None, 1)
    assert hashlib.sha256((SNAP / name.strip()).read_bytes()).hexdigest() == h, name
recs = [json.loads(l) for l in (SNAP / "base_items.jsonl").read_text().splitlines()]
pools = {}
for r in recs:
    P = r["private"]
    for t in [P["gold_task"]] + [a["task"] for a in P["alternatives"]]:
        k = (t["intent"], tuple(sorted(t["slots"])))
        pool = pools.setdefault(k, {n: set() for n in t["slots"]})
        for n, v in t["slots"].items():
            pool[n].add(v)
tasks = []
for (intent, names), pool in sorted(pools.items()):
    for combo in itertools.product(*[sorted(pool[n], key=str) for n in names]):
        tasks.append({"intent": intent, "slots": dict(zip(names, combo))})
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"note": "pooled slot-value inventory, cross-product within each intent; no roles or item ids",
                           "n_tasks": len(tasks), "tasks": tasks}, indent=1, sort_keys=True) + "\n")
print(f"{len(tasks)} candidate tasks, {len({t['intent'] for t in tasks})} intents -> {OUT}")
