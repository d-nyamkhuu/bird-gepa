"""The seed solver that GEPA rewrites. Flex executes this class's source in its sandbox."""

from typing import TYPE_CHECKING

import dspy

if TYPE_CHECKING:
    from bird_gepa.database import QueryResult

    # Flex injects this host tool into the sandbox, not into this Python module.
    # Annotations inside the class must stay builtin, since only its source is sent.
    def query_database(
        db_id: str, sql: str, parameters: list | None = None, row_limit: int = 20
    ) -> QueryResult:
        raise NotImplementedError


class SQLSolver(dspy.Module):
    # A class attribute, since only the class source reaches the sandbox.
    PAGE_SIZE = 200

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought("db_id, question, evidence, schema_text -> sql")
        self.revise = dspy.ChainOfThought(
            dspy.Signature(
                "question, evidence, schema_text, draft_sql, execution -> sql",
                "Review this SQLite query against the question, evidence, and execution preview. "
                "Fix errors or incorrect logic. "
                "An empty or truncated result is not itself an error. "
                "Keep the SQL unchanged when it is already correct. Use the required JSON response "
                "format with reasoning and sql fields; put only the query in the sql field.",
            )
        )

    def list_tables(self, db_id: str) -> list[str]:
        tables: list[str] = []
        while True:
            result = query_database(
                db_id,
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT GLOB 'sqlite_*' ORDER BY name LIMIT ? OFFSET ?",
                parameters=[self.PAGE_SIZE, len(tables)],
                row_limit=self.PAGE_SIZE,
            )
            if result["error"]:
                raise ValueError(result["error"])
            tables.extend(row[0] for row in result["rows"])
            if len(result["rows"]) < self.PAGE_SIZE:
                return tables

    def inspect_schema(self, db_id: str, table_names: list[str]) -> str:
        definitions: list[str] = []
        for start in range(0, len(table_names), self.PAGE_SIZE):
            table_batch = table_names[start : start + self.PAGE_SIZE]
            placeholders = ",".join("?" for _ in table_batch)
            result = query_database(
                db_id,
                "SELECT sql FROM sqlite_master WHERE type = 'table' "
                "AND name IN (" + placeholders + ") ORDER BY name",
                parameters=table_batch,
                row_limit=self.PAGE_SIZE,
            )
            if result["error"]:
                raise ValueError(result["error"])
            definitions.extend(row[0] for row in result["rows"] if row[0])
        return "\n".join(definitions)

    def preview_sql(self, db_id: str, sql: str) -> dict:
        result = query_database(db_id, sql, row_limit=20)
        return {
            "rows": repr(result["rows"]),
            "truncated": result["truncated"],
            "error": result["error"],
        }

    def forward(self, db_id: str, question: str, evidence: str) -> dspy.Prediction:
        tables = self.list_tables(db_id)
        schema_text = self.inspect_schema(db_id, tables)
        sql = self.generate(
            db_id=db_id, question=question, evidence=evidence, schema_text=schema_text
        ).sql
        execution = self.preview_sql(db_id, sql)
        sql = self.revise(
            question=question,
            evidence=evidence,
            schema_text=schema_text,
            draft_sql=sql,
            execution=execution,
        ).sql
        return dspy.Prediction(sql=sql)
