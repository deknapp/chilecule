"""Put the agent itself on the benchmark.

Everything else in :mod:`chilecule.bench` scores a *method*: ECFP4 similarity,
QED, random ordering. This module scores the *agent* -- the same agent that
writes the reports -- on the same held-out set, with the same metrics, against
the same baselines.

Why this exists. An agent that reasons fluently about medicinal chemistry is
easy to build and impossible to trust. The usual evidence offered for one is a
transcript: the agent said sensible things, called the right tools, hedged in
the right places. That is an argument from style. It does not answer the only
question that matters, which is whether the agent's ranking would have found
the actives.

So the agent is handed exactly what the similarity baseline is handed -- a set
of known actives and a pool of unlabelled candidates -- and asked to score the
pool. Its scores go through the identical metric path. If it cannot beat a
free Tanimoto search, that is the finding, and the finding gets reported. A
benchmark that only its author's method can win is not a benchmark.

The fairness constraints are the whole design:

* **The agent never sees a label.** It gets SMILES and opaque candidate IDs.
* **The agent gets no more information than the baseline.** Same reference
  actives, same candidates. Not more reference compounds, not the target name
  in a form the baseline lacks.
* **Candidate order is shuffled with a seeded RNG**, so position carries no
  signal and the shuffle is reproducible.
* **Unscored candidates sink.** A candidate the agent omits or malforms is
  ranked below every candidate it did score, rather than dropped. Dropping
  them would let an agent inflate its own precision by declining to answer on
  everything it found hard.

This module makes real API calls and costs real money. It is deliberately not
imported by the default validation path; see ``chilecule validate --with-agent``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# Scores are requested on a 0-100 integer scale rather than 0-1 floats. Models
# are measurably more consistent on a coarse integer scale, and the metric path
# only cares about rank order, so the lost resolution costs nothing.
SCORE_MIN = 0
SCORE_MAX = 100

# Below every legitimate score, so omitted candidates sort to the bottom
# without colliding with a genuine zero.
UNSCORED = -1.0

RANKING_INSTRUCTIONS = """\
You are ranking candidate compounds for likely activity against a target, given
a set of compounds already known to be active against it.

Score every candidate from 0 to 100, where 100 means "as likely to be active as
the known actives" and 0 means "no reason to expect activity".

You may use your tools. Structural similarity to the known actives is the
obvious signal and you should use it, but it is not the only one -- scaffold
relationships, pharmacophore overlap, and whether a change sits at a position
the known series tolerates all matter.

Report nothing except the scores, one per line, in exactly this format:

CANDIDATE_ID: SCORE

Score every candidate you were given. Do not add commentary, do not group
candidates, and do not omit a candidate because you are uncertain about it --
an uncertain candidate gets a middling score, not silence.
"""


@dataclass
class AgentRanking:
    """Scores the agent returned, plus what it failed to return.

    ``n_unscored`` is kept because it is a quality signal in its own right. An
    agent that silently drops a third of the pool is not doing the task, and a
    metric computed over only what it chose to answer would hide that.
    """

    scores: dict[str, float] = field(default_factory=dict)
    n_requested: int = 0
    n_unscored: int = 0
    n_malformed: int = 0
    tools_called: list[str] = field(default_factory=list)
    raw_answers: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        if self.n_requested == 0:
            return 0.0
        return len(self.scores) / self.n_requested

    def to_dict(self) -> dict:
        return {
            "n_requested": self.n_requested,
            "n_scored": len(self.scores),
            "n_unscored": self.n_unscored,
            "n_malformed": self.n_malformed,
            "coverage": round(self.coverage, 3),
            "unique_tools": sorted(set(self.tools_called)),
        }

    def summary(self) -> str:
        return (
            f"agent scored {len(self.scores)}/{self.n_requested} candidates "
            f"({self.coverage:.1%} coverage), {self.n_malformed} malformed lines, "
            f"tools used: {', '.join(sorted(set(self.tools_called))) or 'none'}"
        )


def candidate_ids(n: int) -> list[str]:
    """Opaque, fixed-width identifiers: ``C0001`` ... ``C{n}``.

    Opaque because anything derived from the molecule -- a ChEMBL ID, an index
    into the original frame -- is a channel through which label information can
    leak. Fixed-width because ragged IDs make the model's output harder to
    parse for no benefit.
    """
    width = max(4, len(str(n)))
    return [f"C{i + 1:0{width}d}" for i in range(n)]


def build_ranking_prompt(
    candidates: dict[str, str],
    reference_actives: list[str],
    *,
    target: str | None = None,
) -> str:
    """Assemble the ranking task.

    ``candidates`` maps candidate ID to SMILES. ``reference_actives`` is the
    same reference set the similarity baseline uses -- passing a different set
    here would make the comparison meaningless.
    """
    header = "KNOWN ACTIVES"
    if target:
        header = f"KNOWN ACTIVES AGAINST {target}"

    actives_block = "\n".join(f"  {s}" for s in reference_actives)
    candidates_block = "\n".join(f"  {cid}: {smiles}" for cid, smiles in candidates.items())

    return (
        f"{RANKING_INSTRUCTIONS}\n"
        f"{header} ({len(reference_actives)} compounds):\n{actives_block}\n\n"
        f"CANDIDATES TO SCORE ({len(candidates)} compounds):\n{candidates_block}\n"
    )


# ``C0001: 73``, tolerating bullets, bold, whitespace, and a trailing period.
_SCORE_LINE = re.compile(
    r"^[\s\-\*>#]*\**\s*(C\d+)\**\s*[:=]\s*\**\s*(-?\d+(?:\.\d+)?)\s*\**\s*\.?\s*$",
    re.MULTILINE,
)


def parse_agent_scores(text: str, valid_ids: set[str]) -> tuple[dict[str, float], int]:
    """Extract ``{candidate_id: score}`` from the agent's reply.

    Returns the scores and a count of lines that looked like scores but were
    not usable -- an unknown ID, or a number outside the requested range. That
    count is reported rather than swallowed, because a model that has started
    inventing candidate IDs is failing in a way the ranking metrics cannot see.

    On a repeated ID the first occurrence wins. Models that restate a block
    tend to restate it verbatim, and taking the first keeps the result
    independent of how much the model padded its answer.
    """
    scores: dict[str, float] = {}
    malformed = 0

    for match in _SCORE_LINE.finditer(text):
        cid, raw = match.group(1), match.group(2)
        if cid not in valid_ids or cid in scores:
            if cid not in valid_ids:
                malformed += 1
            continue
        try:
            value = float(raw)
        except ValueError:  # pragma: no cover - regex guarantees a number
            malformed += 1
            continue
        if not (SCORE_MIN <= value <= SCORE_MAX):
            malformed += 1
            continue
        scores[cid] = value

    return scores, malformed


def scores_to_array(ranking: AgentRanking, ordered_ids: list[str]) -> np.ndarray:
    """Align agent scores to the candidate order the metrics expect.

    Anything unscored becomes :data:`UNSCORED`, which sorts below every real
    score. See the module docstring for why omissions must not be dropped.
    """
    return np.array(
        [ranking.scores.get(cid, UNSCORED) for cid in ordered_ids], dtype=float
    )


def shuffled_batches(
    ordered_ids: list[str],
    smiles_by_id: dict[str, str],
    *,
    batch_size: int,
    seed: int,
) -> list[dict[str, str]]:
    """Split candidates into shuffled batches of ``batch_size``.

    Shuffled so that neither position within a batch nor assignment to a batch
    correlates with anything about the compound. Seeded so a run can be
    repeated exactly.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    rng = np.random.default_rng(seed)
    shuffled = list(ordered_ids)
    rng.shuffle(shuffled)

    return [
        {cid: smiles_by_id[cid] for cid in shuffled[i : i + batch_size]}
        for i in range(0, len(shuffled), batch_size)
    ]


async def rank_with_agent(
    smiles: list[str],
    reference_actives: list[str],
    *,
    target: str | None = None,
    batch_size: int = 40,
    seed: int = 0,
    model: str = "claude-opus-5",
    max_turns: int = 30,
) -> tuple[AgentRanking, list[str]]:
    """Have the agent score ``smiles``; return the ranking and the ID order.

    The returned ID list is in the same order as ``smiles``, so
    :func:`scores_to_array` can align the result back onto the caller's frame.

    Batched because a pool of several hundred SMILES in one prompt degrades
    scoring quality well before it hits a context limit -- the model starts
    treating the list as a block to summarize rather than items to judge.
    """
    from ..agents.discovery import run_detailed

    ids = candidate_ids(len(smiles))
    smiles_by_id = dict(zip(ids, smiles, strict=True))

    ranking = AgentRanking(n_requested=len(ids))

    for batch in shuffled_batches(ids, smiles_by_id, batch_size=batch_size, seed=seed):
        prompt = build_ranking_prompt(batch, reference_actives, target=target)
        run = await run_detailed(prompt, model=model, max_turns=max_turns)

        parsed, malformed = parse_agent_scores(run.answer, set(batch))
        ranking.scores.update(parsed)
        ranking.n_malformed += malformed
        ranking.tools_called.extend(run.tools_called)
        ranking.raw_answers.append(run.answer)

    ranking.n_unscored = ranking.n_requested - len(ranking.scores)
    return ranking, ids
