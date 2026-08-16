# learning_machine_learning

A collection of machine learning algorithms, each with an in-depth explanation of the method —
its mathematics derived rather than asserted, its behaviour worked through on real numbers, its
failure modes stated plainly — paired with a practical implementation that can be run, measured
and modified.

Each subject lives in its own directory under [topics/](topics/), carrying its own lessons, its
own shared harness, its own data preparation and its own projects. A new subject is an addition,
never an edit.

## Topics

| Topic | Subject | Status |
|---|---|---|
| [rag-retrieval-methods](topics/rag-retrieval-methods/) | Document retrieval for RAG — two approaches, BM25 and a dense bi-encoder, to be compared over one shared evaluation harness | [BM25 lesson](topics/rag-retrieval-methods/lessons/bm25.md) |

## Who these are for, and what they promise

College-level students and practitioners looking for resources to learn machine learning
algorithms — people who want to understand *why* a method works, not only which library call
invokes it.

Every algorithm is meant to arrive as a matched pair: a lesson under `topics/<topic>/lessons/`
explaining the method from first principles, and a project under `topics/<topic>/projects/`
implementing and evaluating it, so the explanation can be checked against something that actually
runs. Where a topic covers several algorithms, all of them are measured through one shared
harness — a comparison is only meaningful if both approaches pass through identical evaluation
code. The lessons come first; the implementations follow, topic by topic.

Every document describing an algorithm is written to three rules:

- **Self-contained.** As far as it can be, each document develops what it needs — including the
  mathematics — rather than sending the reader elsewhere. References exist for provenance and for
  readers who want the original treatment, never as a prerequisite for following the argument.
- **College-level algebra and calculus, and no more.** Anything beyond that — a probability
  identity, an information-theoretic quantity, what a training objective does — is stated and
  derived where it is first used.
- **Accurate, and easy to read and comprehend.** Every formula is derived rather than asserted,
  every worked number is computed rather than estimated, and every claim about how a method fails
  is stated as plainly as the claims about how it works.

## Conventions

- **Code only.** No datasets, run artifacts, embeddings or model weights are committed. Everything
  else is reconstructible from what is here.
- **One virtualenv per project**, never one for the repository. Dependency sets conflict by design.
- **Shared code is scoped to its topic.** It is promoted to a repository-level `packages/` on its
  second consumer, never in anticipation of one.
- **Licensed by content type.** Code is Apache-2.0 — see [LICENSE](LICENSE) — and that includes the
  code samples printed inside the lessons, so anything lifted from a lesson into a project carries
  the same terms as the rest of the repository. Lesson prose is
  [CC BY 4.0](topics/rag-retrieval-methods/lessons/LICENSE): copy it, adapt it, translate it, use it
  commercially, with credit. Creative Commons licenses are not meant for software and Apache-2.0 is
  not meant for prose, which is why the split follows content rather than directory.
