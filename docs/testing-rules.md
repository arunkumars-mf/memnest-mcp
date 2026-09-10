# Six rules, each earned by a bug that got past the suite

These come out of a long review series on this codebase. Every one of them
exists because a check passed while something was wrong, so they are written as
tests you apply to your own reasoning rather than as style advice.

## 1. An explanation earns confidence only when you check a prediction it makes beyond the result you already have

A correct result with a wrong explanation is the most durable kind of error,
because passing tests protect it.

Worked example. Two stories explained the FTS behaviour: "a mixed token indexes
only if its alphabetic run is at least 2 characters" and "the tokenizer splits
at alpha↔digit boundaries, then drops short and numeric fragments". Both predict
`q7` fails and `ab7` matches, so the threshold fixture — however clean — could
never separate them. What separated them was querying `abcd` and `88` against
content `abcd88`: only the split story predicted that `abcd` matches and `88`
does not. The wrong story then produced a wrong fix (`num3300`, which collapses
to the shared token `num`).

The same shape appeared earlier when score movement was attributed to
`access_count` on correlation alone. In both cases the tell was available for
free: the explanation made an additional prediction nobody had checked.

**Corollary.** If you cannot name a prediction your explanation makes beyond the
result you already have, you do not have an explanation. You have a restatement.

## 2. Assert the test fails without the fix, for the stated reason

A test that encodes a rationale is only as good as the rationale, and a passing
test protects a wrong one indefinitely.

Worked example. A guard asserted that a superseded memory never outranks its
current version, and it passed — by coupling, not by neutrality. Centrality
favoured the stale memory by ~0.03, and the ×0.5 supersession penalty happened
to cancel it. Since the coherent replacement for that penalty under rank fusion
is a rank demotion, the guard would have silently stopped guarding the moment
the penalty changed. The replacement test asserts the property that survives a
penalty redesign, and was verified to fail with the fix reverted.

**Procedure.** After adding a guard test, revert the fix, confirm the test fails,
restore the fix. **Corollary.** If you cannot describe the failure mode the test
would catch, you have written an assertion, not a test.

## 3. Presence tests cannot find disclosure bugs

Four round-trip tests covered `memory_export`, and all four asserted what
*survived*. The bug was what *escaped*: the export embedded the user's absolute
workspace path in 39 places. "Data made it" never implies "nothing else did".

Round-trip tests are inherently sender-side — they ask *did my data arrive?* —
which is why no quantity of them would have found this, and why the bug surfaced
from **using** an export as a shared artifact rather than from testing or reading
the code. That was the first time anyone occupied the recipient's position.

**Rule.** For anything that crosses a trust boundary, write at least one
assertion from the recipient's perspective.

**Related design lesson.** Opt-in protection protects nobody, because the person
who needs it is the person not thinking about it. When redaction seems to cost
diagnostic value, check whether the diagnostics can be computed on the sending
side instead — `db_inside_workspace` and `private_to_workspace` stay valid
without printing a path, which dissolved the tradeoff rather than trading it off.

## 4. When a fix requires editing tests, say which of "wrong contract" or "wrong assumption" applies

This is distinct from rule 2, and more dangerous. A vacuous test protects
nothing. A test that passes *correctly* while defending the bug has real
coverage: it fails when you change the behaviour, and the behaviour it defends
is the defect.

Worked example. Three tests asserted that `memory_stats` returned raw filesystem
paths. They had genuine coverage and they blocked the redaction fix.

The moment a fix requires touching tests is a fork: either the tests encode a
contract you are violating, or they encode an assumption you are correcting —
and **the diff looks identical either way**. State which it is, in the commit,
so a later reader can tell whether a test was relaxed or re-pointed. Doing it
silently lets the suite quietly ratify whatever the code now does.

## 5. Enumerate emission sites from the code, not from your description of it

A release note is a derived artifact. If it merged two things, a count taken
from it inherits the merge.

Worked example. A fix was reported as covering "four hint surfaces". There were
five: the write-time hint has two branches — `near_duplicate` above
`DEDUP_THRESHOLD` and `value_disagreement` below it — and the release note had
already collapsed them into one line. The read-time pair had been enumerated
separately and both were fixed; the write-time pair was enumerated as one and
only one was fixed. Grepping for `hint` assignments gives three sites, one with
two branches: five.

This is not rule 4 and not a vacuous test. Three tests were added, each
correctly asserting its fixture fires first, and all three passed. The gap was
an **unenumerated surface**, so there was never a test to write.

## 6. Control the branch selector, not the fixture

When a branch is chosen by comparing a measured value against a threshold,
never let the fixture's measured value decide which branch runs. Move the
threshold.

Worked example. A test meant to cover both write-time hint branches used text
fixtures. Measured, they score 0.9138 and 0.7192 against a threshold of 0.92 —
so **both** landed on `value_disagreement`, the `near_duplicate` branch was
never exercised, and the test passed while that branch was broken. Confirmed by
reverting the fix and watching it still pass. The repair is to monkeypatch the
threshold (0.50 forces one branch, 0.999 the other) and to assert the measured
similarity so the test proves which branch it ran.

A straddling fixture does not fail. It reports as coverage while testing an
arbitrary branch — the same protection-by-passing-check as rule 1, one level
down: rule 1 is an explanation that may be wrong while its test passes, this is
a *branch* that may be wrong while its test passes.

## The thread joining 5 and 6

Both are the same failure at different levels: **trusting a derived artifact
where the primary source was available and cheap to read.** A release note
instead of the emission sites; a similarity number quoted from a different
fixture instead of measuring your own.

That second one is not hypothetical, and it went wrong twice. The first
write-up of rule 6 cited 0.9226 for a fixture that actually scores 0.9138 — the
conclusion held and the evidence for it did not, which is rule 1 again. The
first *correction* then attributed the gap to "a different corpus", which is
also wrong: cosine similarity is a function of the two texts and the model
alone, so the same pair yields the same number anywhere. Verified — the 0.9226
pair measures 0.9226 in a second environment, and still 0.9226 with 30
unrelated memories added.

The gap was a **wording** difference between two fixtures, not an environment
difference. That distinction is practical: fixture similarities are portable, so
`conflict_similarity` can be asserted as an exact value — which "it depends on
your corpus" would have discouraged.

This is the thread an author is least likely to notice about their own work,
because the derived artifact is usually something they wrote themselves.
