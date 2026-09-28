# Project Brief: Modeling How a Student Learns Math (Knowledge Tracing)

**The research question:** Given the sequence of problems a student has answered — which topics, right or wrong, in what order — can a model predict whether they'll get the *next* problem right, and can we recover what the model actually learned about the *structure* of math knowledge (which skills depend on which)?

**Why this one:** It's a real, active research area with clean public datasets, it maps directly onto your learning path (each phase unlocks a stronger model on the *same* problem), and the domain is one you know cold — Moroccan BAC-style math skills and prerequisites.

---

## Datasets

Start with the first; the others are level-ups once your pipeline works.

- **ASSISTments 2009 ("skill builder")** — *start here.* The classic benchmark, small and well-documented. Columns you care about: `user_id`, `problem_id`, `skill_id` / `skill_name`, `correct` (0/1), `order_id` (temporal order). 
  - ⚠️ **Known gotcha:** the raw file has duplicate rows (problems tagged with multiple skills get repeated). Cleaning this is a rite of passage — use a de-duplicated version or dedupe yourself, and *report which you used*. Half of "my AUC doesn't match the paper" is this.
- **Eedi** — math *diagnostic* multiple-choice questions (used in a NeurIPS 2020 competition). Closest to your BAC world; also lets you model *which wrong answer* they picked, not just right/wrong.
- **Junyi Academy** — Chinese math platform, and crucially **ships a ground-truth prerequisite graph of skills.** This is your Phase 5 evaluation set — you can check whether your model recovered the real dependency structure.
- **EdNet (Riiid)** — very large real test-prep dataset (~130M interactions). Use only once your code scales; it's for the deep-learning phase.

> Don't fetch a URL from memory — search for the current download page for each; they move.

---

## The metric (and the one discipline that matters)

- **Primary metric: ROC-AUC** on next-interaction correctness. Report accuracy too, but AUC is what papers use.
- **Split by student, and respect time.** Never shuffle a student's history so the future leaks into the past. Hold out whole students for test, and within a student always predict step *t* using only steps *< t*. Getting this wrong inflates your score and is the #1 beginner mistake in this field. If a number looks too good, you leaked.

---

## Milestones (mapped to your learning path)

Each version must produce **one AUC number that beats the previous version.** That's your ship signal.

| Version | Phase | What you build | Teaches you |
|---|---|---|---|
| **v0 — Baselines** | Phase 0–1 | Global correct-rate; per-item mean correctness; per-student mean | Loading real data, evaluation, AUC, the leakage trap |
| **v1 — IRT / logistic** | Phase 3 | Item Response Theory (Rasch/1PL): `P(correct)=σ(ability_student − difficulty_item)`, fit by MLE. It *is* logistic regression with student + item embeddings. | Maximum likelihood, optimization, embeddings — on real data |
| **v2 — BKT** *(optional)* | Phase 3 | Bayesian Knowledge Tracing: a per-skill hidden Markov model with 4 params (prior, learn, slip, guess) | Probability, HMMs, latent-state thinking |
| **v3 — DKT** | Phase 4 | Deep Knowledge Tracing: an RNN/LSTM reading the interaction sequence | Sequence models, backprop through time |
| **v4 — Attention** | Phase 4–5 | SAKT then AKT: a transformer over the student's history | Attention, on a problem you already understand |
| **v5 — Frontier** | Phase 5 | Probe the trained model to recover the skill prerequisite graph | Interpretability + real research |

The original DKT paper (Piech et al., 2015) showed you can extract skill *dependencies* by probing the trained model — asking "if the student masters skill A, how much does predicted P(correct) on skill B rise?" That gives you an influence matrix → a graph. **v5 is:** build that graph, then compare it against Junyi's ground-truth prerequisite graph. That's a genuine, publishable-flavored question, in your domain.

---

## Your exact first notebook (do this week, before you feel ready)

1. Download ASSISTments 2009. Load into a pandas DataFrame.
2. **EDA:** count students, unique problems, unique skills, total interactions; overall correct rate; plot the distribution of per-student sequence lengths.
3. **Baseline A (naive):** predict the global correct rate for every interaction. Compute accuracy and AUC.
4. **Baseline B (item difficulty):** for each problem, compute its historical mean correctness *from the training split only*; predict that value on the test split. Compute accuracy and AUC.
5. Write down both AUC numbers. **That's v0. Everything you build for the rest of the project has to beat Baseline B.**

If you finish steps 1–5 and have two honest AUC numbers with a clean student-level train/test split, you've shipped v0 — and you've already practiced loading real data, evaluation, and the leakage discipline that trips up most people.

---

## Scope discipline

The temptation will be to read ten KT papers and design v5 before v0 runs. Don't. The rule for this project: **you may only start the next version once the current version's AUC number is written down and beats the last one.** The problem itself pulls the math in — you'll learn attention properly at v4 *because you need it to beat v3*, which is the whole point of picking one problem that grows with you.

**Definition of done for the project (for now):** v3 (DKT) beating your IRT baseline on ASSISTments, with a correct student-level, time-respecting split. Reaching that means you've genuinely learned how modern sequence models work — on a problem you care about. v4 and v5 are the "keep going" frontier.
