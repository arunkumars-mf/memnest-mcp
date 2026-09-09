# Four rules, each earned by a bug that got past the suite

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
