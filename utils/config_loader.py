"""
Configuration Loader
=====================

Loads and validates configuration from YAML files.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Configuration error."""
    pass


@dataclass
class ApiConfig:
    """API configuration."""
    polymarket_rest_url: str = "https://clob.polymarket.com"
    polymarket_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    gamma_api_url: str = "https://gamma-api.polymarket.com"
    kalshi_api_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""
    private_key: str = ""
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class TradingConfig:
    """Trading configuration."""
    markets: list[str] = field(default_factory=list)
    min_edge: float = 0.01
    bundle_arb_enabled: bool = True
    min_spread: float = 0.05
    tick_size: float = 0.01
    mm_enabled: bool = True
    default_order_size: float = 50.0
    min_order_size: float = 5.0
    max_order_size: float = 200.0
    slippage_tolerance: float = 0.02
    order_timeout_seconds: float = 60.0


@dataclass
class RiskConfig:
    """Risk configuration."""
    max_position_per_market: float = 200.0
    max_global_exposure: float = 5000.0
    max_daily_loss: float = 500.0
    max_drawdown_pct: float = 0.10
    trade_only_high_volume: bool = True
    min_24h_volume: float = 10000.0
    whitelist: list[str] = field(default_factory=list)
    blacklist: list[str] = field(default_factory=list)
    kill_switch_enabled: bool = True
    auto_unwind_on_breach: bool = False


@dataclass
class ModeConfig:
    """Trading mode configuration."""
    trading_mode: str = "dry_run"  # "live" or "dry_run"
    data_mode: str = "real"  # "real" or "simulation" - use simulation for demos
    cross_platform_enabled: bool = True  # Enable cross-platform arbitrage (Polymarket + Kalshi)
    kalshi_enabled: bool = True  # Enable Kalshi market monitoring
    min_match_similarity: float = 0.6  # Minimum similarity score for market matching (0-1)
    dry_run_initial_balance: float = 10000.0
    simulate_fills: bool = True
    fill_probability: float = 0.8


@dataclass
class LoggingConfig:
    """Logging configuration."""
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    log_dir: str = "logs"
    main_log_file: str = "bot.log"
    trades_log_file: str = "trades.log"
    opportunities_log_file: str = "opportunities.log"
    max_log_size_mb: int = 50
    backup_count: int = 5


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    snapshot_interval: float = 60.0
    heartbeat_interval: float = 30.0
    track_latency: bool = True
    track_fill_rates: bool = True


_VALID_NEWS_PROVIDERS = frozenset({"router", "gdelt", "newsapi", "mediastack"})
_VALID_FALLBACK_PROVIDERS = frozenset({"gdelt", "newsapi", "mediastack"})
_VALID_CATEGORIES = frozenset({
    "internal_arb", "cross_market_arb", "crypto_price", "macro_release",
    "fed_rates", "weather", "sports", "legal_regulatory", "company_sec",
    "politics_election", "geopolitics", "generic_news", "unknown",
})


@dataclass
class AIEngineConfig:
    """LLM-News strategy configuration."""
    enabled: bool = False
    allow_live_trading: bool = False
    anthropic_api_key: str = ""
    news_api_key: str = ""
    mediastack_api_key: str = ""
    model: str = "claude-sonnet-4-6"
    check_interval_seconds: float = 60.0
    max_markets_per_cycle: int = 20
    calls_per_minute: int = 10
    news_page_size: int = 10
    news_max_age_minutes: int = 360
    min_edge: float = 0.08
    min_confidence: str = "high"
    max_spread: float = 0.03
    default_order_size: float = 5.0
    max_order_size: float = 25.0
    shadow_mode: bool = True
    long_only: bool = True
    submit_in_dry_run: bool = False
    news_provider: str = "router"
    fallback_news_provider: str = "gdelt"
    source_router_enabled: bool = True
    enabled_categories: list = field(default_factory=lambda: [
        "crypto_price", "macro_release", "fed_rates", "weather",
        "legal_regulatory", "company_sec", "generic_news",
    ])
    disabled_categories: list = field(default_factory=lambda: [
        "politics_election", "geopolitics", "sports",
    ])


@dataclass
class BotConfig:
    """Complete bot configuration."""
    api: ApiConfig = field(default_factory=ApiConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    mode: ModeConfig = field(default_factory=ModeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    ai_engine: AIEngineConfig = field(default_factory=AIEngineConfig)
    
    @property
    def is_dry_run(self) -> bool:
        return self.mode.trading_mode.lower() == "dry_run"
    
    @property
    def is_live(self) -> bool:
        return self.mode.trading_mode.lower() == "live"
    
    @property
    def use_simulation(self) -> bool:
        """Use simulated data (for demos/screenshots)."""
        return self.mode.data_mode.lower() == "simulation"


def load_config(config_path: str = "config.yaml") -> BotConfig:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        BotConfig instance with loaded values
        
    Raises:
        ConfigError: If the config file cannot be loaded or is invalid
    """
    path = Path(config_path)
    
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")
    
    try:
        with open(path, "r") as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in config file: {e}")
    
    if raw_config is None:
        raw_config = {}
    
    # Parse sections
    api_data = raw_config.get("api", {})
    trading_data = raw_config.get("trading", {})
    risk_data = raw_config.get("risk", {})
    mode_data = raw_config.get("mode", {})
    logging_data = raw_config.get("logging", {})
    monitoring_data = raw_config.get("monitoring", {})
    ai_engine_data = raw_config.get("ai_engine", {})

    # Handle environment variable overrides
    api_data = _apply_env_overrides(api_data, {
        "api_key": "POLYMARKET_API_KEY",
        "api_secret": "POLYMARKET_API_SECRET",
        "passphrase": "POLYMARKET_PASSPHRASE",
        "private_key": "POLYMARKET_PRIVATE_KEY",
    })
    ai_engine_data = _apply_env_overrides(ai_engine_data, {
        "anthropic_api_key": "ANTHROPIC_API_KEY",
        "news_api_key": "NEWS_API_KEY",
        "mediastack_api_key": "MEDIASTACK_API_KEY",
        "news_provider": "NEWS_PROVIDER",
    })

    # Build config objects
    config = BotConfig(
        api=_build_dataclass(ApiConfig, api_data),
        trading=_build_dataclass(TradingConfig, trading_data),
        risk=_build_dataclass(RiskConfig, risk_data),
        mode=_build_dataclass(ModeConfig, mode_data),
        logging=_build_dataclass(LoggingConfig, logging_data),
        monitoring=_build_dataclass(MonitoringConfig, monitoring_data),
        ai_engine=_build_dataclass(AIEngineConfig, ai_engine_data),
    )
    
    # Validate
    _validate_config(config)
    
    return config


def _apply_env_overrides(data: dict, env_map: dict[str, str]) -> dict:
    """Apply environment variable overrides to config data."""
    result = data.copy()
    for key, env_var in env_map.items():
        env_value = os.environ.get(env_var)
        if env_value:
            result[key] = env_value
    return result


def _build_dataclass(cls, data: dict):
    """Build a dataclass from a dictionary, ignoring unknown keys."""
    import dataclasses
    
    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered_data = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered_data)


def _validate_config(config: BotConfig) -> None:
    """Validate configuration values."""
    errors = []
    
    # Trading validation
    if config.trading.min_edge < 0 or config.trading.min_edge > 1:
        errors.append("trading.min_edge must be between 0 and 1")
    
    if config.trading.min_spread < 0 or config.trading.min_spread > 1:
        errors.append("trading.min_spread must be between 0 and 1")
    
    if config.trading.tick_size <= 0:
        errors.append("trading.tick_size must be positive")
    
    if config.trading.default_order_size <= 0:
        errors.append("trading.default_order_size must be positive")
    
    # Risk validation
    if config.risk.max_position_per_market <= 0:
        errors.append("risk.max_position_per_market must be positive")
    
    if config.risk.max_global_exposure <= 0:
        errors.append("risk.max_global_exposure must be positive")
    
    if config.risk.max_daily_loss < 0:
        errors.append("risk.max_daily_loss must be non-negative")
    
    if config.risk.max_drawdown_pct < 0 or config.risk.max_drawdown_pct > 1:
        errors.append("risk.max_drawdown_pct must be between 0 and 1")
    
    # Mode validation
    if config.mode.trading_mode.lower() not in ("live", "dry_run"):
        errors.append("mode.trading_mode must be 'live' or 'dry_run'")
    
    # Live mode checks
    if config.is_live:
        if not config.api.api_key or config.api.api_key == "YOUR_API_KEY_HERE":
            errors.append("api.api_key is required for live trading")
        if not config.api.private_key or config.api.private_key == "YOUR_PRIVATE_KEY_HERE":
            errors.append("api.private_key is required for live trading")
    
    # AI Engine validation
    _RETIRED_MODELS = {"claude-3-5-sonnet-20241022", "claude-3-opus-20240229"}
    if config.ai_engine.enabled:
        ai = config.ai_engine
        if not (0.0 <= ai.min_edge <= 1.0):
            errors.append("ai_engine.min_edge must be between 0 and 1")
        if not (0.0 <= ai.max_spread <= 1.0):
            errors.append("ai_engine.max_spread must be between 0 and 1")
        if ai.default_order_size <= 0:
            errors.append("ai_engine.default_order_size must be positive")
        if ai.max_order_size < ai.default_order_size:
            errors.append("ai_engine.max_order_size must be >= default_order_size")
        if ai.min_confidence not in ("low", "medium", "high"):
            errors.append("ai_engine.min_confidence must be 'low', 'medium', or 'high'")
        if ai.check_interval_seconds <= 0:
            errors.append("ai_engine.check_interval_seconds must be positive")
        if ai.max_markets_per_cycle <= 0:
            errors.append("ai_engine.max_markets_per_cycle must be positive")
        if ai.calls_per_minute <= 0:
            errors.append("ai_engine.calls_per_minute must be positive")
        if ai.news_page_size <= 0:
            errors.append("ai_engine.news_page_size must be positive")
        if ai.news_max_age_minutes <= 0:
            errors.append("ai_engine.news_max_age_minutes must be positive")
        if ai.model in _RETIRED_MODELS:
            errors.append(
                f"ai_engine.model '{ai.model}' is retired — "
                "use 'claude-sonnet-4-6' or 'claude-opus-4-7'"
            )
        if config.is_live and not ai.allow_live_trading:
            logger.warning(
                "[AIEngine] Live mode detected but allow_live_trading=false — shadow mode forced"
            )
        # News-Key requirement depends on provider
        if ai.news_provider not in _VALID_NEWS_PROVIDERS:
            errors.append(
                f"ai_engine.news_provider '{ai.news_provider}' invalid — "
                f"must be one of: {sorted(_VALID_NEWS_PROVIDERS)}"
            )
        if ai.fallback_news_provider not in _VALID_FALLBACK_PROVIDERS:
            errors.append(
                f"ai_engine.fallback_news_provider '{ai.fallback_news_provider}' invalid — "
                f"must be one of: {sorted(_VALID_FALLBACK_PROVIDERS)}"
            )
        unknown_enabled = set(ai.enabled_categories) - _VALID_CATEGORIES
        if unknown_enabled:
            errors.append(
                f"ai_engine.enabled_categories contains unknown values: {sorted(unknown_enabled)}"
            )
        unknown_disabled = set(ai.disabled_categories) - _VALID_CATEGORIES
        if unknown_disabled:
            errors.append(
                f"ai_engine.disabled_categories contains unknown values: {sorted(unknown_disabled)}"
            )
        if config.is_live and not ai.anthropic_api_key:
            errors.append("ai_engine: ANTHROPIC_API_KEY required in live mode")
        if config.is_live and ai.news_provider == "newsapi" and not ai.news_api_key:
            errors.append("ai_engine: NEWS_API_KEY required when news_provider=newsapi in live mode")
        if config.is_live and ai.news_provider == "mediastack" and not ai.mediastack_api_key:
            errors.append(
                "ai_engine: MEDIASTACK_API_KEY required when news_provider=mediastack in live mode"
            )
        if ai.shadow_mode and ai.submit_in_dry_run:
            logger.warning(
                "[AIEngine] shadow_mode=true overrides submit_in_dry_run — no signals will be submitted"
            )

    if errors:
        raise ConfigError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


def save_config(config: BotConfig, config_path: str = "config.yaml") -> None:
    """Save configuration to a YAML file."""
    import dataclasses
    
    def to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        return obj
    
    data = to_dict(config)
    
    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_default_config() -> BotConfig:
    """Get a default configuration."""
    return BotConfig()

