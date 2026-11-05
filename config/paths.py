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

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / 'config' / 'dataset_config.json'

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
