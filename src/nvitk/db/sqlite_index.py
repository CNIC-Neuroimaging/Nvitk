"""
SQLite mirror of catalog Parquet tables for fast filtered queries (rebuilt from Parquet).

:class:`SQLiteIndex` materializes whole tables and supports :func:`~nvitk.db.filters.build_sql_where`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import click
except Exception:
    click = None

from nvitk.core.exceptions import BackendUnavailableError

from .catalog import DatasetCatalog
from .filters import build_sql_where, escape_identifier
from .storage import coerce_dataframe_to_manifest, read_parquet_table


def _cli_decorator(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


_click_command = click.command if click is not None else _cli_decorator
_click_option = click.option if click is not None else _cli_decorator


class SQLiteIndex:
    """Path to ``catalog.sqlite`` (or configured index); :meth:`build` refreshes from Parquet."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def exists(self) -> bool:
        return self.db_path.exists()

    def build(self, catalog: DatasetCatalog, *, tables: list[str] | None = None) -> Path:
        selected_tables = tables or catalog.list_tables()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as connection:
            for table_name in selected_tables:
                definition = catalog.get_table(table_name)
                if not definition.path.exists():
                    continue
                df = read_parquet_table(definition.path)
                df = coerce_dataframe_to_manifest(df, definition.columns)
                df.to_sql(table_name, connection, index=False, if_exists="replace")
                for column in tuple(definition.key_columns) + tuple(definition.index_columns):
                    if column not in df.columns:
                        continue
                    index_name = f"idx_{table_name}_{column}".replace("-", "_")
                    connection.execute(
                        f"CREATE INDEX IF NOT EXISTS {escape_identifier(index_name)} "
                        f"ON {escape_identifier(table_name)} ({escape_identifier(column)})"
                    )

            connection.execute(
                "CREATE TABLE IF NOT EXISTS _dataset_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute("DELETE FROM _dataset_meta")
            connection.executemany(
                "INSERT INTO _dataset_meta (key, value) VALUES (?, ?)",
                [
                    ("schema_version", catalog.repository_manifest["schema_version"]),
                    ("dataset_name", catalog.repository_manifest["dataset_name"]),
                ],
            )
            connection.commit()
        return self.db_path

    def query_table(
        self,
        table_name: str,
        *,
        columns: list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        select_clause = ", ".join(escape_identifier(column) for column in columns) if columns else "*"
        sql = f"SELECT {select_clause} FROM {escape_identifier(table_name)}"
        where_clause, params = build_sql_where(filters)
        if where_clause:
            sql += f" WHERE {where_clause}"

        with sqlite3.connect(self.db_path) as connection:
            return pd.read_sql_query(sql, connection, params=params)


@_click_command()
@_click_option(
    "--dataset-root",
    type=click.Path(path_type=Path) if click is not None else None,
    default=Path("dataset"),
    show_default=True,
    help="Dataset root containing the catalog manifests.",
)
@_click_option(
    "--tables",
    type=str,
    default=None,
    help="Comma-separated subset of tables to index. Defaults to all tables.",
)
def main(dataset_root: Path, tables: str | None) -> None:
    if click is None:
        raise BackendUnavailableError('click is not installed. Please install it with "pip install click".')

    catalog = DatasetCatalog(dataset_root)
    table_list = [item.strip() for item in tables.split(",") if item.strip()] if tables else None
    index = SQLiteIndex(catalog.sqlite_index_path)
    out = index.build(catalog, tables=table_list)
    click.echo(str(out))


if __name__ == "__main__":
    main()
