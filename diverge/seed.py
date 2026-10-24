"""
Seed the arena with DEMO submissions written by the DIVERGE team (flagged
demo_seed=True and prefixed 'demo:'). Scores are computed by the real scorer;
nothing here is a pre-set score. Includes deliberately bad entries (off-topic,
gibberish, conventional paraphrase) to show that the gate works.

Run:  .venv/bin/python -m diverge.seed
"""
from __future__ import annotations

from .baselines import get_baselines
from .questions import BY_ID
from .scoring import score_submission
from . import store

SEEDS = [
    ("demo:ada", "q02",
     "Congestion is a pricing problem, not a supply problem: peak road space is free, so it is overconsumed. New lanes cut the time cost of driving, which induces latent demand until speeds fall back to the old equilibrium.",
     "Because trips are priced at zero, added capacity gets absorbed (the Downs-Thomson paradox). Stockholm's congestion charge cut traffic about 20% within months, while the Katy Freeway widening in Houston was followed by longer commutes. We should predict that pricing, not lanes, changes peak speeds."),
    ("demo:turing", "q02",
     "More roads cause worse traffic because they attract more cars. Cities should build more public transport to reduce congestion.", ""),
    ("demo:noise", "q02",
     "purple elephant quantum banana sideways mirror forty clocks dissolve under velvet trains", "blorp zip carrot."),
    ("demo:hopper", "q01",
     "The underrated driver is wasted computation on redundant data: much of a web-scale corpus is near-duplicate, so a large share of training FLOPs re-learn the same patterns. The fix is data selection, not bigger chips.",
     "Deduplication studies on large text corpora found that removing near-duplicates lets models reach the same loss with fewer tokens, which means fewer GPU-hours. If this is right, we should see energy per unit of capability fall when curation improves, even on identical hardware."),
    ("demo:curie", "q03",
     "The hidden bottleneck is interconnection study throughput: clean projects wait years in queues because grid operators run power-flow studies serially with small engineering teams.",
     "Queues hold far more capacity than is installed, and most projects withdraw after years of delay. The constraint is therefore labor and process, not technology cost. For example, cluster-based studies and automation could clear more megawatts per engineer-year; we would expect queue times to drop where studies are batched."),
    ("demo:offtopic", "q03",
     "The best sourdough bread uses an active starter and a long cold fermentation in the fridge overnight.", "Because slow fermentation develops flavor."),
    ("demo:lovelace", "q05",
     "Experts lose to simple models largely because they are inconsistent: the same expert gives different judgments to identical cases, and that noise, not bias, erodes accuracy.",
     "A linear model applies the same weights every time, so it removes noise. Studies of clinical prediction found that even a model of the expert's own judgments outperforms the expert, which implies inconsistency is the main cost. We should therefore expect averaging several experts to beat any single one."),
    ("demo:feynman", "q09",
     "Models struggle to admit uncertainty because training rewards confident completion: in pre-training and human preference data, a fluent guess usually scores better than 'I don't know', so calibration is trained out.",
     "If graders penalize abstention, the optimal policy is to guess. That is an incentive problem in the evaluation, not a missing capability. We would predict that benchmarks which give partial credit for abstaining produce models that abstain more appropriately."),
]


def main():
    existing = {(r["user"], r["question_id"]) for r in store.load_all()}
    for user, qid, ans, why in SEEDS:
        if (user, qid) in existing:
            continue
        q = BY_ID[qid]
        b = get_baselines(q["text"])
        ev = score_submission(q["text"], ans, why, b["baselines"], store.prior_texts(qid))
        store.add(dict(user=user, question_id=qid, domain=q["domain"], question=q["text"], answer=ans,
                       reasoning=why, baselines=b["baselines"], baseline_meta=b["meta"],
                       evaluation=ev, demo_seed=True))
        print(f"{user:15s} {qid} score={ev['score']:5.1f} gate={ev['gate']:.2f}")


if __name__ == "__main__":
    main()
