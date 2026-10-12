The current process uses **training questions to propose changes to the solver’s Python source**, then **validation questions to select the best solver**. Model weights stay fixed. GEPA now uses a fixed **300-question validation subset** and **eight training questions per reflection**. Before/after evaluation still uses all 1,376 validation questions. The completed-run results below describe the earlier four-question/full-validation configuration, with a seed that re-ran its final SQL before returning it.

**1. Prepare and split the data**

We start with **6,601 BIRD training-source questions across 69 databases**.

The preparation script assigns **whole databases** to training or validation. With seed **42**, it orders databases using a deterministic hash, then chooses a validation group whose question count is closest to the **20% target**. Because databases stay intact, the split is only approximately 80/20.

Cleaning removes **two duplicate questions**, and **two training questions with failing reference SQL** are excluded. The usable sets are:

| Set | Questions | Databases | Purpose |
|---|---:|---:|---|
| Training | **5,221** | **54** | Supply execution traces and feedback for reflection |
| Validation | **1,376** | **15** | Score and select candidate solvers |
| Mini-dev test | **500** | **11** | Reserved for final evaluation |

I verified that **none of these sets share databases**. The test set comes from the separate mini-dev dataset. [Preparation code](bird_gepa/prepare.py)

**2. Turn questions into DSPy examples**

For every training and validation question, preparation executes the reference SQL and saves its complete answer rows.

Each example contains:

- **Solver inputs:** database ID, question, and evidence.
- **Evaluator labels:** reference SQL and expected answer rows.

The solver does **not** receive the reference SQL or answers. Those labels are available to grading and optimization feedback. Prepared examples are saved so ordinary runs do not need to execute reference queries again. Preparation makes no model calls. [Example loading](bird_gepa/data.py)

**3. Load the starting solver into Flex**

The starting implementation is the `SQLSolver` class. Its source is loaded into `dspy.Flex`, which makes that Python implementation editable by GEPA.

For one question, the current seed workflow:

1. Lists the database’s tables.
2. Fetches their schema definitions.
3. Uses **Luna** to generate draft SQL.
4. Executes the draft and collects a result preview or error.
5. Uses **Luna again** to review and revise the SQL, and returns it. Grading executes it.

The seed uses **two predictor calls per question**. Flex now allows at most
**ten**, giving optimized code room for conditional repair and selection steps.
This is a ceiling; raising it does not add calls to the seed.

Execution previews normally contain at most **20 rows**. The host database tool permits up to **200**, enforces read-only access, and gives each SQL query a **five-second deadline**. These controls live outside the editable solver source. [Seed implementation](bird_gepa/seed.py)

Flex lets GEPA change the **workflow, helper methods, prompts, and branching logic**, rather than just prompt wording. Its code runs inside the Deno/Pyodide interpreter. [DSPy Flex documentation](https://dspy.ai/current/api/modules/Flex)

**4. Grade the starting solver**

The CLI first evaluates the seed on **all 1,376 validation questions**, unless
`--skip-baseline` is supplied. That flag skips only this full baseline pass;
GEPA still scores the seed on its 300-question subset before search.

For each question, the evaluator executes the returned SQL and determines:

- **Correct:** successful execution and the same complete unordered set of rows as the reference.
- **Incorrect:** execution failure or a different result; score zero.

The current GEPA score for a correct answer is
`max(0, 1 - 0.01 * max(0, solver_query_count - 1))`. One query is free; four
queries score **0.97**, eight score **0.93**. Every host query tool attempt counts,
including schema queries and failed attempts. Reference preparation and grading
queries are excluded. `--query-penalty` changes the weight; zero restores binary
scoring. This objective can trade some aggregate accuracy for fewer queries.
Full-validation logs report raw accuracy separately from the penalized score,
along with total/average solver queries and average query tool time. Time and
model calls are not penalized in this version.

Row order and duplicate multiplicity are ignored by this set comparison. Grading uses complete results, not the solver’s 20-row preview.

In the completed run, the seed answered **881 / 1,376 correctly: 64.03%**. [Metric implementation](bird_gepa/metric.py)

**5. Run the GEPA search**

The current defaults are **5,500 metric calls**, **eight training examples per reflection batch**, **300 validation questions for candidate selection**, **eight workers**, and seed **42**. Luna runs the solver; **Astra** proposes source changes. Candidate merging is disabled. All 300 slots are allocated proportionally to each database's question count in the full validation split, using largest remainders and no fixed per-database minimum. Sampling is deterministic and without replacement. The other 1,076 validation questions remain outside training.

Each iteration works like this:

1. Select a parent candidate from the **Pareto frontier**. A candidate can remain useful because it solves particular validation questions better, even if its overall accuracy is lower.
2. Take **eight training questions** from a shuffled training pool.
3. Run the parent on those questions and collect scores, execution traces, and feedback. Feedback includes final SQL, result counts, query usage and a bounded host query sequence. Failure feedback adds reference SQL, column counts, and missing/unexpected row samples.
4. If all eight score **1.0**, skip reflection. Correct answers using additional queries score below one and can still prompt efficiency improvements.
5. Otherwise, ask Astra to propose revised Python source.
6. Run the revision on the **same eight questions**.
7. If its total score improves, evaluate it on the **same fixed 300 validation questions** and update the candidate pool.

So an ordinary accepted revision costs approximately:

**8 parent checks + 8 revision checks + 300 validation checks = 316 metric calls.**

It does **not** run through all 5,221 training questions each iteration. [Optimization setup](bird_gepa/optimization.py)

**6. What happened in the completed run**

There were **four search iterations**, but **three reflection proposals**. The first iteration’s four training answers were all correct, so reflection was skipped.

| Candidate | Correct validation answers | Accuracy |
|---|---:|---:|
| Seed | **881 / 1,376** | **64.03%** |
| First revision | **869 / 1,376** | **63.15%** |
| Second revision | **977 / 1,376** | **71.00%** |
| Third revision | **994 / 1,376** | **72.24%** |

The first revision illustrates why validation matters: improving four training answers did not guarantee better overall accuracy.

The recorded GEPA budget was:

**4 validation passes × 1,376 + 28 training checks = 5,532 metric calls.**

That means **about 99.5% of the budget went to validation**. The budget slightly exceeded 5,500 because evaluation batches finish before stopping. The run initially used 16 workers, then reduced to eight after rate limiting.

**7. Select and save the result**

GEPA returns the candidate with the best aggregate score on the fixed 300-question subset. The CLI evaluates it again on all 1,376 validation questions and saves its source as `solver.py`. The CLI’s before-and-after evaluations are outside GEPA’s optimization budget. `--skip-baseline` omits the before evaluation and reports only the final full-validation score, without importing a historical baseline. There is no periodic full validation during search. The checkpoint records the selection and refuses incompatible resumes; the changed setup needs a fresh checkpoint directory.

The selected solver improved validation accuracy by **8.21 percentage points**, fixing **147** previously wrong answers while losing **34** previously correct ones. The **500 test questions remained unused**. [Completed report](artifacts/overnight-20261005/report.txt) (local, not committed)

The seed’s Luna-based SQL review happens within each question. The three Astra reflections were separate events that rewrote the solver for subsequent questions.
