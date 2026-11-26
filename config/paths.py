"""
Centralized path configuration for quant_alpha_lab.

Structure:
  data/
    raw/{source}/{asset}/{region}/{universe}/{freq}/{type}/
    clean/{asset}/{region}/{universe}/{freq}/
      base/           # ohlcv.parquet, features.parquet (shared)
      {stage}/{run_id}/  # signals, signal_eval, en_scores, en_model, risk_model, weights, backtest
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import json
import os

ROOT = Path(__file__).resolve().parents[1]


def _config_file() -> Path:
    '''
    The dataset config. Default: config/dataset_config.json.
    The environment variable QAL_DATASET_CONFIG selects another file (path relative to the repo root, or absolute).
    Example, the synthetic market:  QAL_DATASET_CONFIG=config/dataset_synthetic.json python scripts/pipeline/02_etl_prices.py
    Every stage reads its paths through this file, so one variable switches the whole pipeline to another dataset.
    '''
    env = os.environ.get('QAL_DATASET_CONFIG')
    if not env:
        return ROOT / 'config' / 'dataset_config.json'
    p = Path(env)
    return p if p.is_absolute() else ROOT / p


CONFIG_FILE = _config_file()

_DEFAULTS = {
    'source': 'yfinance',
    'asset': 'equities',
    'region': 'US',
    'universe': 'SP500',
    'freq': 'daily',
}


def _load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


_cfg = _load_config()
DEFAULTS = {k: _cfg.get(k, v) for k, v in _DEFAULTS.items()}

# Point-in-time universe (stage 02b). In the dataset config file:
#   "universe_mode": "all"         every ticker is in the universe on every date it trades (the old behaviour)
#                    "rule"        top N by dollar volume, re-ranked every k days (src/universe/point_in_time.py)
#                    "membership"  the vendor list data/raw/.../membership.csv
#   "universe_rule": {"top_n": 1000, ...}   overrides of DEFAULT_RULE
UNIVERSE_MODE = _cfg.get('universe_mode', 'all')
UNIVERSE_RULE = _cfg.get('universe_rule', {})


@dataclass
class DataPaths:
    """Minimal path config - base paths + run_id for versioned outputs."""
    source: str = DEFAULTS['source']
    asset: str = DEFAULTS['asset']
    region: str = DEFAULTS['region']
    universe: str = DEFAULTS['universe']
    freq: str = DEFAULTS['freq']
    run_id: Optional[str] = None

    @property
    def raw_base(self) -> Path:
        return ROOT / 'data' / 'raw' / self.source / self.asset / self.region / self.universe / self.freq

    @property
    def clean_base(self) -> Path:
        return ROOT / 'data' / 'clean' / self.asset / self.region / self.universe / self.freq

    # Raw
    @property
    def raw_ohlcv_dir(self) -> Path:
        return self.raw_base / 'ohlcv'

    # Clean base tables (shared)
    @property
    def ohlcv_file(self) -> Path:
        return self.clean_base / 'ohlcv' / 'ohlcv.parquet'

    @property
    def universe_file(self) -> Path:
        '''[date, ticker, in_universe]: who was investable on each date (written by 02b_build_universe.py).'''
        return self.clean_base / 'universe' / 'universe.parquet'

    @property
    def raw_membership_file(self) -> Path:
        '''Optional vendor index list [ticker, start, end] (see docs/survivorship.md).'''
        return self.raw_base / 'membership.csv'

    @property
    def features_file(self) -> Path:
        return self.clean_base / 'features' / 'features.parquet'

    # Pipeline outputs (versioned by run_id)
    def stage_dir(self, stage: str) -> Path:
        """Get directory for a pipeline stage (signals, en_scores, etc.)."""
        if self.run_id is None:
            raise ValueError("run_id required for pipeline output paths")
        return self.clean_base / stage / self.run_id


def get_paths(
    source: str = None,
    asset: str = None,
    region: str = None,
    universe: str = None,
    freq: str = None,
    run_id: Optional[str] = None,
) -> DataPaths:
    """Create DataPaths with optional overrides from config/dataset_config.json."""
    return DataPaths(
        source=source or DEFAULTS['source'],
        asset=asset or DEFAULTS['asset'],
        region=region or DEFAULTS['region'],
        universe=universe or DEFAULTS['universe'],
        freq=freq or DEFAULTS['freq'],
        run_id=run_id,
    )


def run_id_from_args(date: str, tag: str = 'default') -> str:
    return f'{date}_{tag}'
