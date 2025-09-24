# backend/state.py
from __future__ import annotations

import threading
import pandas as pd
from typing import Dict, Any, Optional


class AppState:
    """Thread-safe in-memory store for the latest dataframes and metadata."""
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.tables: Dict[str, pd.DataFrame] = {}
        self.meta: Dict[str, Any] = {}

    def set_table(self, name: str, df: pd.DataFrame) -> None:
        with self._lock:
            self.tables[name] = df

    def get_table(self, name: str) -> Optional[pd.DataFrame]:
        with self._lock:
            return self.tables.get(name)

    def list_tables(self) -> Dict[str, int]:
        with self._lock:
            return {k: len(v) for k, v in self.tables.items()}

    def set_meta(self, **kwargs: Any) -> None:
        with self._lock:
            self.meta.update(kwargs)

    def get_meta(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.meta)


STATE = AppState()
