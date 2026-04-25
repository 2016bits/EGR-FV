from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Iterable

from .io import ensure_parent_dir


def setup_logger(name: str, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(ensure_parent_dir(log_file), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class CSVLogger:
    def __init__(self, path: str):
        self.path = ensure_parent_dir(path)
        self.fieldnames: list[str] | None = None

    def log(self, row: Dict[str, object]) -> None:
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerow(row)
            return

        new_fields = [field for field in row.keys() if field not in self.fieldnames]
        if new_fields:
            self._rewrite_with_new_fields(row, new_fields)
            return

        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(row)

    def _rewrite_with_new_fields(self, row: Dict[str, object], new_fields: Iterable[str]) -> None:
        assert self.fieldnames is not None
        existing_rows = []
        if self.path.exists():
            with self.path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                existing_rows = list(reader)

        self.fieldnames = self.fieldnames + list(new_fields)
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            for existing_row in existing_rows:
                writer.writerow(existing_row)
            writer.writerow(row)
