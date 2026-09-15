"""
Single place responsible for knowing where every project dataframe lives on
disk and how to load it. Right now that's master_df / shot_df_clean (xG) and
events_df (event classification), but the registry pattern means adding a
future dataframe never touches the models code — you only add a loader
function here.

Usage:
    loader = DataLoader()
    shots_df = loader.load('shot_df_clean')
    events_df = loader.load('events_df')

Adding a new dataframe later:
    loader.register('player_ratings_df', build_player_ratings_df)
    loader.load('player_ratings_df')   # runs the function, caches result
"""
import os
from typing import Callable, Optional

import pandas as pd

from src.runners.feature_build_runner import FeatureBuildRunner
from src.runners.shot_clean_runner import ShotCleaningRunner
from src.runners.event_clean_runner import EventCleaningRunner

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')


def _load_master_df(path: Optional[str] = None) -> pd.DataFrame:
    path = path or os.path.join(PROCESSED_DIR, 'master_df.parquet')
    if not os.path.exists(path):
        raise FileNotFoundError(f"master_df.parquet not found at {path} — run FeatureBuildRunner().run() first.")
    return pd.read_parquet(path)


def _load_shot_df_clean(path: Optional[str] = None) -> pd.DataFrame:
    runner = ShotCleaningRunner(output_path=path)
    if not os.path.exists(runner.output_path):
        # Fall back to building it from master_df rather than failing outright.
        return runner.run()
    return runner.load_output()


def _load_events_df(path: Optional[str] = None) -> pd.DataFrame:
    runner = EventCleaningRunner(output_path=path)
    if not os.path.exists(runner.output_path):
        return runner.run()
    return runner.load_output()


class DataLoader:
    """Registry-based loader: name -> loader function, with per-instance caching."""

    def __init__(self):
        self._registry: dict[str, Callable[..., pd.DataFrame]] = {
            'master_df': _load_master_df,
            'shot_df_clean': _load_shot_df_clean,
            'events_df': _load_events_df,
        }
        self._cache: dict[str, pd.DataFrame] = {}

    def register(self, name: str, loader_fn: Callable[..., pd.DataFrame]) -> None:
        """Plug in a new dataframe without touching any existing code."""
        self._registry[name] = loader_fn

    def load(self, name: str, path: Optional[str] = None, use_cache: bool = True) -> pd.DataFrame:
        if name not in self._registry:
            raise KeyError(
                f"No loader registered for '{name}'. "
                f"Known dataframes: {list(self._registry)}. "
                "Use DataLoader.register(name, loader_fn) to add a new one."
            )
        if use_cache and name in self._cache:
            return self._cache[name]

        df = self._registry[name](path) if path is not None else self._registry[name]()
        if use_cache:
            self._cache[name] = df
        return df

    def clear_cache(self, name: Optional[str] = None) -> None:
        if name is None:
            self._cache.clear()
        else:
            self._cache.pop(name, None)