"""
Centralized path configuration for the quant_alpha_lab pipeline.

Structure:
  data/
    raw/      # Immutable, source-format
      {source}/{asset}/{region}/{universe}/{freq}/{type}/
    clean/    # Research-ready, single parquet per base table
      {asset}/{region}/{universe}/{freq}/
        {base}/           # ohlcv, features (one file each)
        {pipeline_stage}/ # signals, signal_eval, en_scores, en_model, risk_model, weights, backtest
          {run_id}/       # versioned by date_tag (e.g., 2026-10-29_default)
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import json

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / 'config' / 'dataset_config.json'

# Hardcoded defaults (used if config file missing)
DEFAULT_SOURCE = 'yfinance'
DEFAULT_ASSET = 'equities'
DEFAULT_REGION = 'US'
DEFAULT_UNIVERSE = 'SP500'
DEFAULT_FREQ = 'daily'

def _load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}

_cfg = _load_config()
DEFAULT_SOURCE = _cfg.get('source', DEFAULT_SOURCE)
DEFAULT_ASSET = _cfg.get('asset', DEFAULT_ASSET)
DEFAULT_REGION = _cfg.get('region', DEFAULT_REGION)
DEFAULT_UNIVERSE = _cfg.get('universe', DEFAULT_UNIVERSE)
DEFAULT_FREQ = _cfg.get('freq', DEFAULT_FREQ)


@dataclass
class DataConfig:
    """Configuration for data paths."""
    source: str = DEFAULT_SOURCE
    asset: str = DEFAULT_ASSET
    region: str = DEFAULT_REGION
    universe: str = DEFAULT_UNIVERSE
    freq: str = DEFAULT_FREQ
    run_id: Optional[str] = None

    @property
    def raw_base(self) -> Path:
        return ROOT / 'data' / 'raw' / self.source / self.asset / self.region / self.universe / self.freq

    @property
    def clean_base(self) -> Path:
        return ROOT / 'data' / 'clean' / self.asset / self.region / self.universe / self.freq

    # Raw data
    @property
    def raw_ohlcv_dir(self) -> Path:
        return self.raw_base / 'ohlcv'

    # Clean base tables (shared across runs)
    @property
    def clean_ohlcv_file(self) -> Path:
        return self.clean_base / 'ohlcv' / 'ohlcv.parquet'

    @property
    def clean_features_file(self) -> Path:
        return self.clean_base / 'features' / 'features.parquet'

    # Pipeline outputs (versioned by run_id)
    def _run_dir(self, stage: str) -> Path:
        if self.run_id is None:
            raise ValueError("run_id required for pipeline output paths")
        return self.clean_base / stage / self.run_id

    @property
    def signals_dir(self) -> Path:
        return self._run_dir('signals')

    @property
    def signals_file(self) -> Path:
        return self.signals_dir / 'signals.parquet'

    @property
    def signals_config_file(self) -> Path:
        return self.signals_dir / 'config.json'

    @property
    def signal_eval_dir(self) -> Path:
        return self._run_dir('signal_eval')

    @property
    def en_scores_dir(self) -> Path:
        return self._run_dir('en_scores')

    @property
    def en_scores_file(self) -> Path:
        return self.en_scores_dir / 'en_scores.parquet'

    @property
    def en_scores_config_file(self) -> Path:
        return self.en_scores_dir / 'config.json'

    @property
    def en_model_dir(self) -> Path:
        return self._run_dir('en_model')

    @property
    def en_model_file(self) -> Path:
        return self.en_model_dir / 'model.json'

    @property
    def en_model_stability_file(self) -> Path:
        return self.en_model_dir / 'stability.parquet'

    @property
    def risk_model_dir(self) -> Path:
        return self._run_dir('risk_model')

    @property
    def weights_dir(self) -> Path:
        return self._run_dir('weights')

    @property
    def weights_file(self) -> Path:
        return self.weights_dir / 'weights.parquet'

    @property
    def backtest_dir(self) -> Path:
        return self._run_dir('backtest')


def get_data_config(
    source: str = None,
    asset: str = None,
    region: str = None,
    universe: str = None,
    freq: str = None,
    run_id: Optional[str] = None,
) -> DataConfig:
    """Create a DataConfig with optional overrides from config/dataset_config.json."""
    return DataConfig(
        source=source or DEFAULT_SOURCE,
        asset=asset or DEFAULT_ASSET,
        region=region or DEFAULT_REGION,
        universe=universe or DEFAULT_UNIVERSE,
        freq=freq or DEFAULT_FREQ,
        run_id=run_id,
    )


def run_id_from_args(date: str, tag: str = 'default') -> str:
    return f'{date}_{tag}'
