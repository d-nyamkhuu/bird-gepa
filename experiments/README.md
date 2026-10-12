# Experiments

The two runs reported in the main [README](../README.md#results). Both used the
default settings. The only difference was the query penalty.

| Folder | Query penalty | Seed → optimized accuracy |
| --- | ---: | --- |
| [query-penalty](query-penalty/) | 0.01 | 62.21% → 67.44% |
| [accuracy-only](accuracy-only/) | 0 | 61.19% → 73.55% |

## Files

Each folder holds:

| File | Contents |
| --- | --- |
| `run.json` | The command, settings and timings |
| `result.json` | Accuracy, query counts and metric calls, before and after |
| `solver.json` | The optimized solver, in DSPy's format |
| `candidates.json` | The source of every candidate GEPA accepted. The first is the seed |
| `candidate_tree.html` | GEPA's search tree. Open it in a browser |
| `run_log.json` | Each iteration's parent, training batch and scores |
| `selection.json` | The training questions, and the 300-question selection subset |
| `run.log` | The full log, including each reflection prompt and proposal |

## Notes

- **These runs used an earlier seed**, which re-ran its final SQL and so made
  four queries per question. The current seed makes three.
- **The query-penalty run was restarted** after Codex `server_is_overloaded`
  errors, which led to the retry handling in `codex.py`. Its `run.log` is
  the log of the final, complete attempt.
- **Paths in `run.json` and the logs point to `artifacts/`**, where the runs
  were written. That directory, the response cache and GEPA's binary state are
  not committed.
- **`solver.json` was saved after the runs**, when the CLI moved to DSPy's
  format. Load it with `load_solver`, as shown in the main README.

## Source changes

Each winner's source, compared with the seed both runs started from. That is
the earlier seed described above, which ends with a final
`self.execute_sql(db_id, sql)`.

<details>
<summary>Seed → query-penalised winner</summary>

```diff
--- seed.py (as run)
+++ query-penalty/solver.json
@@ -1,72 +1,46 @@
 class SQLSolver(dspy.Module):
-    def __init__(self) -> None:
+    def __init__(self):
         super().__init__()
-        self.generate = dspy.ChainOfThought("db_id, question, evidence, schema_text -> sql")
-        self.revise = dspy.ChainOfThought(
+        self.generate = dspy.ReAct(
             dspy.Signature(
-                "question, evidence, schema_text, draft_sql, execution -> sql",
-                "Review this SQLite query against the question, evidence, and execution preview. "
-                "Fix errors or incorrect logic. "
-                "An empty or truncated result is not itself an error. "
-                "Keep the SQL unchanged when it is already correct. Use the required JSON response "
-                "format with reasoning and sql fields; put only the query in the sql field.",
-            )
+                'db_id: str, question: str, evidence: str, schema_text: str -> sql: str',
+                'Write one read-only SQLite query answering the question. Return SQL only, without markdown. The schema has already been retrieved; do not retrieve it again. Use query_database only for a targeted value or relationship check that resolves a genuine ambiguity. Do not routinely execute the finished query. '
+                'Select exactly the requested attributes. A request for which customer normally means CustomerID from the order, not a person name: customers may be organizations and joining to Person can remove valid customers. Only join a name table when a name is requested. For explicitly requested personal names stored in separate columns, select first and last name separately. Use the fewest necessary joins. '
+                'Interpret evidence as guidance on columns and meanings, not as an infallible executable formula. For proportions among all entities, keep the entire population in the denominator and put the category condition inside a conditional aggregate, not in WHERE. An average number of a category among all categories means its frequency share: COUNT(CASE WHEN condition THEN entity_id END) * 1.0 / COUNT(entity_id), not SUM of numeric identifiers. For percent more of one category than another among all employees or other stated population, calculate 100.0 * (conditional_count_A - conditional_count_B) / COUNT(population_id); do not divide by category B unless the question explicitly specifies that baseline. In contrast, an average measurement of a filtered group uses SUM(measurement) * 1.0 / COUNT(entity_id) with that group in WHERE. Do not sum identifier values. Preserve the natural join grain unless distinct entities are explicitly required or joins introduce unrelated duplication. '
+                'Categorical labels in evidence can have different capitalization from stored labels. Use case-insensitive comparison (COLLATE NOCASE or LIKE without wildcards) for human-readable categories and textual boolean flags when case is not meaningful. This avoids unnecessary lookup calls. However, when an explicit evidence formula defines literal conditional buckets, preserve those bucket predicates rather than broadening their meaning. For abbreviated categories or typographic punctuation differences that case-insensitive matching cannot resolve, inspect the relevant distinct values or a distinctive prefix with a small lookup. Do not guess alternative labels. Exact descriptive prose supplied by evidence can be matched directly. '
+                'For a singular superlative request (the product with the highest review rating, the longest recipe, the highest subtotal order), prefer ORDER BY the relevant measurement DESC LIMIT 1 on the joined result. This convention returns one result, not all ties, even when evidence describes the measurement as MAX or gives its maximum numeric value. Do not replace this with equality to a MAX subquery unless all tied results are explicitly requested. Apply this top-one convention even when the selected attribute is from a detail table joined to the ranked entity. Avoid DISTINCT before top-one selection unless needed. For highest number of related records, GROUP BY the parent entity and ORDER BY COUNT(...) DESC LIMIT 1. For youngest people, sort birthdate descending in the eligible population. '
+                'When a question attaches multiple loosely phrased superlatives to a transaction and its associated role, use evidence to identify the primary ranked measure and avoid inventing an independent global-maximum restriction on the associated role. If evidence defines only the transaction maximum and the role is merely described as assigned, join to that role and rank the transactions; independently restrict to a maximum role attribute only when that restriction is explicitly defined or clearly required. '
+                'Use evidence for explicit join mappings even if names differ from the natural-language concept. Explicit numeric bounds, dates and quantities in the question override conflicting evidence values. Preserve direct numeric column comparisons; do not add CAST, currency stripping or arithmetic coercion unless explicitly requested. For division of counts use a REAL numerator to avoid integer division. Respect inclusive endpoints. Quote reserved identifiers and names containing spaces. Never use a preview limit as an answer limit. Before returning, check projection, denominator population, category case, join multiplicity, ranking scope and requested result cardinality.'
+            ),
+            tools=[query_database],
+            max_iters=5,
         )
 
-    def list_tables(self, db_id: str) -> list[str]:
-        tables: list[str] = []
-        while True:
+    def forward(self, **inputs):
+        db_id = inputs['db_id']
+        definitions = []
+        for page in range(5):
             result = query_database(
                 db_id,
-                "SELECT name FROM sqlite_master WHERE type = 'table' "
-                "AND name NOT GLOB 'sqlite_*' ORDER BY name LIMIT ? OFFSET ?",
-                parameters=[200, len(tables)],
+                "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name NOT GLOB 'sqlite_*' ORDER BY name LIMIT ? OFFSET ?",
+                parameters=[200, page * 200],
                 row_limit=200,
             )
-            if result["error"]:
-                raise ValueError(result["error"])
-            tables.extend(row[0] for row in result["rows"])
-            if len(result["rows"]) < 200:
-                return tables
-
-    def inspect_schema(self, db_id: str, table_names: list[str]) -> str:
-        definitions: list[str] = []
-        for start in range(0, len(table_names), 200):
-            table_batch = table_names[start : start + 200]
-            placeholders = ",".join("?" for _ in table_batch)
-            result = query_database(
-                db_id,
-                "SELECT sql FROM sqlite_master WHERE type = 'table' "
-                "AND name IN (" + placeholders + ") ORDER BY name",
-                parameters=table_batch,
-                row_limit=200,
-            )
-            if result["error"]:
-                raise ValueError(result["error"])
-            definitions.extend(row[0] for row in result["rows"] if row[0])
-        return "\n".join(definitions)
-
-    def execute_sql(self, db_id: str, sql: str) -> dict:
-        result = query_database(db_id, sql, row_limit=20)
-        return {
-            "rows": repr(result["rows"]),
-            "truncated": result["truncated"],
-            "error": result["error"],
-        }
-
-    def forward(self, db_id: str, question: str, evidence: str) -> dspy.Prediction:
-        tables = self.list_tables(db_id)
-        schema_text = self.inspect_schema(db_id, tables)
-        sql = self.generate(
-            db_id=db_id, question=question, evidence=evidence, schema_text=schema_text
-        ).sql
-        execution = self.execute_sql(db_id, sql)
-        sql = self.revise(
-            question=question,
-            evidence=evidence,
-            schema_text=schema_text,
-            draft_sql=sql,
-            execution=execution,
-        ).sql
-        self.execute_sql(db_id, sql)
+            if result.get('error'):
+                definitions.append('Schema retrieval error: ' + str(result['error']))
+                break
+            rows = result.get('rows', [])
+            definitions.extend(str(row[1]) for row in rows if row[1])
+            if len(rows) < 200:
+                break
+        out = self.generate(
+            db_id=db_id,
+            question=inputs['question'],
+            evidence=inputs['evidence'],
+            schema_text='\n'.join(definitions),
+        )
+        sql = str(out.sql).strip()
+        if sql.startswith('```'):
+            lines = sql.splitlines()
+            sql = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:]).strip()
         return dspy.Prediction(sql=sql)
```

</details>

<details>
<summary>Seed → accuracy-only winner</summary>

```diff
--- seed.py (as run)
+++ accuracy-only/solver.json
@@ -1,72 +1,95 @@
 class SQLSolver(dspy.Module):
-    def __init__(self) -> None:
+    def __init__(self):
         super().__init__()
-        self.generate = dspy.ChainOfThought("db_id, question, evidence, schema_text -> sql")
-        self.revise = dspy.ChainOfThought(
-            dspy.Signature(
-                "question, evidence, schema_text, draft_sql, execution -> sql",
-                "Review this SQLite query against the question, evidence, and execution preview. "
-                "Fix errors or incorrect logic. "
-                "An empty or truncated result is not itself an error. "
-                "Keep the SQL unchanged when it is already correct. Use the required JSON response "
-                "format with reasoning and sql fields; put only the query in the sql field.",
-            )
+        rules = (
+            "Generate a read-only SQLite query answering the question, using only the supplied actual schema. "
+            "Return executable SQL only, without Markdown. Select only requested columns in requested order. "
+            "Full names mapped to first_name and last_name mean two separate columns, not concatenation. "
+            "Use evidence mappings and formulas unless they contradict the explicit requested concept. "
+            "For a named categorical type use equality or LIKE without wildcards; use substring LIKE only "
+            "when the question asks for labels containing a term or about a topic. Ground uncertain literals "
+            "in stored values rather than inventing spelling or spacing. Do not add unsupported filters. "
+            "Resolve the requested counting grain before choosing an expression. 'How many' normally asks "
+            "for COUNT of matching records or distinct entities, not a numeric-looking field. In a "
+            "user-to-compliment association, counting compliments of a specified type means COUNT of matching "
+            "joined compliment records; do not select, cast, or sum number_of_compliments unless the request "
+            "or evidence explicitly asks for that stored quantity. A column name alone does not override "
+            "the requested row-count operation. Never cast a text value to integer just to make an answer numeric. "
+            "Use COUNT(*) for occurrences and COUNT(DISTINCT key) for unique entities where fanout is possible. "
+            "For percentages use floating-point division with the full filtered population as denominator "
+            "and SUM(CASE WHEN condition THEN 1 ELSE 0 END) as numerator, not COUNT(condition). "
+            "Use event-specific foreign keys: an order destination must join its address key, not every "
+            "address belonging to the customer. Avoid join fanout. "
+            "Distinguish individual extrema from grouped extrema: MIN(num_sales) refers to an individual "
+            "sales record unless totals are requested; most films by year requires grouping by year and count. "
+            "Respect answer cardinality. A singular 'which game/entity has the fewest/most' asks for one "
+            "winner: ORDER BY the measure ASC/DESC LIMIT 1. Do not return all tied entities via equality "
+            "to MIN/MAX unless all ties are requested. For such single-winner joins start from the "
+            "requested entity and follow its relationship chain to the measure; avoid DISTINCT before LIMIT. "
+            "For any N items use LIMIT N and no arbitrary additional filters. Determine whether requested "
+            "IDs denote row identities or category memberships: category IDs represented in a subcategory "
+            "table are its parent category foreign keys and should be DISTINCT, whereas explicit individual "
+            "subcategory row identifiers use the subcategory primary key. Do not assume every ID request "
+            "must project the source table's primary key. Quote identifiers when needed. "
+            "Do not select ordering or explanatory columns."
         )
+        self.generate = dspy.ChainOfThought(dspy.Signature(
+            "db_id: str, question: str, evidence: str, schema_text: str -> sql: str", rules
+        ))
+        self.review = dspy.ReAct(dspy.Signature(
+            "db_id: str, question: str, evidence: str, schema_text: str, draft_sql: str, execution: str -> sql: str",
+            rules + " Review semantics as well as syntax. Specifically check count versus stored quantity, "
+            "single-winner versus all-ties cardinality, and identifier versus category-membership projection. "
+            "A successful execution does not establish correctness. Use query_database to inspect uncertain "
+            "stored literals or schema relationships and to resolve errors. Inspect DISTINCT stored values "
+            "with a broad search for uncertain place names. Zero results may signal a bad literal, but do "
+            "not change meaning just to obtain rows. Truncation is not an error, though thousands of rows "
+            "for a singular winner indicate a cardinality mistake. Keep correct drafts unchanged. Validate "
+            "any changed query using the tool. Return only final SQL."
+        ), tools=[query_database], max_iters=8)
 
-    def list_tables(self, db_id: str) -> list[str]:
-        tables: list[str] = []
-        while True:
-            result = query_database(
-                db_id,
-                "SELECT name FROM sqlite_master WHERE type = 'table' "
-                "AND name NOT GLOB 'sqlite_*' ORDER BY name LIMIT ? OFFSET ?",
-                parameters=[200, len(tables)],
-                row_limit=200,
-            )
-            if result["error"]:
-                raise ValueError(result["error"])
-            tables.extend(row[0] for row in result["rows"])
-            if len(result["rows"]) < 200:
-                return tables
+    def forward(self, **inputs):
+        def clean_sql(text: str) -> str:
+            """Remove optional SQL Markdown fencing."""
+            text = str(text).strip()
+            if text.startswith('```'):
+                text = '\n'.join(text.splitlines()[1:])
+                if text.rstrip().endswith('```'):
+                    text = text.rstrip()[:-3].rstrip()
+            return text
 
-    def inspect_schema(self, db_id: str, table_names: list[str]) -> str:
-        definitions: list[str] = []
-        for start in range(0, len(table_names), 200):
-            table_batch = table_names[start : start + 200]
-            placeholders = ",".join("?" for _ in table_batch)
-            result = query_database(
-                db_id,
-                "SELECT sql FROM sqlite_master WHERE type = 'table' "
-                "AND name IN (" + placeholders + ") ORDER BY name",
-                parameters=table_batch,
-                row_limit=200,
-            )
-            if result["error"]:
-                raise ValueError(result["error"])
-            definitions.extend(row[0] for row in result["rows"] if row[0])
-        return "\n".join(definitions)
+        def execute(sql: str) -> dict:
+            """Preview a query without aborting on database errors."""
+            try:
+                return query_database(inputs['db_id'], sql, row_limit=20)
+            except Exception as exc:
+                return {'error': str(exc), 'rows': []}
 
-    def execute_sql(self, db_id: str, sql: str) -> dict:
-        result = query_database(db_id, sql, row_limit=20)
-        return {
-            "rows": repr(result["rows"]),
-            "truncated": result["truncated"],
-            "error": result["error"],
-        }
-
-    def forward(self, db_id: str, question: str, evidence: str) -> dspy.Prediction:
-        tables = self.list_tables(db_id)
-        schema_text = self.inspect_schema(db_id, tables)
-        sql = self.generate(
-            db_id=db_id, question=question, evidence=evidence, schema_text=schema_text
-        ).sql
-        execution = self.execute_sql(db_id, sql)
-        sql = self.revise(
-            question=question,
-            evidence=evidence,
-            schema_text=schema_text,
-            draft_sql=sql,
-            execution=execution,
-        ).sql
-        self.execute_sql(db_id, sql)
-        return dspy.Prediction(sql=sql)
+        definitions = []
+        for page in range(20):
+            try:
+                result = query_database(
+                    inputs['db_id'],
+                    "SELECT sql FROM sqlite_master WHERE type = 'table' AND name NOT GLOB 'sqlite_*' ORDER BY name LIMIT ? OFFSET ?",
+                    parameters=[200, page * 200], row_limit=200
+                )
+                if result.get('error'):
+                    definitions.append('Schema inspection error: ' + str(result['error']))
+                    break
+                rows = result.get('rows', [])
+                definitions.extend(str(row[0]) for row in rows if row[0])
+                if len(rows) < 200:
+                    break
+            except Exception as exc:
+                definitions.append('Schema inspection error: ' + str(exc))
+                break
+        schema_text = '\n'.join(definitions)
+        draft = self.generate(db_id=inputs['db_id'], question=inputs['question'],
+                              evidence=inputs['evidence'], schema_text=schema_text)
+        sql = clean_sql(draft.sql)
+        execution = execute(sql)
+        reviewed = self.review(db_id=inputs['db_id'], question=inputs['question'],
+                               evidence=inputs['evidence'], schema_text=schema_text,
+                               draft_sql=sql, execution=str(execution))
+        revised = clean_sql(reviewed.sql)
+        return dspy.Prediction(sql=revised or sql)
```

</details>

## Licence

These files contain questions, evidence, reference SQL and database rows from
[BIRD](https://bird-bench.github.io/), which is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). That licence,
not the repository's MIT licence, applies to that content.
