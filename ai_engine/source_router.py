import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MarketCategory(str, Enum):
    INTERNAL_ARB = "internal_arb"
    CROSS_MARKET_ARB = "cross_market_arb"
    CRYPTO_PRICE = "crypto_price"
    MACRO_RELEASE = "macro_release"
    FED_RATES = "fed_rates"
    WEATHER = "weather"
    SPORTS = "sports"
    LEGAL_REGULATORY = "legal_regulatory"
    COMPANY_SEC = "company_sec"
    POLITICS_ELECTION = "politics_election"
    GEOPOLITICS = "geopolitics"
    GENERIC_NEWS = "generic_news"
    UNKNOWN = "unknown"


@dataclass
class SourceRoute:
    category: MarketCategory
    primary_provider: str
    fallback_providers: list[str]
    confidence: float
    reason: str


_CRYPTO_KEYWORDS = frozenset({
    "btc", "eth", "sol", "xrp", "bnb", "ada", "avax", "doge", "matic",
    "bitcoin", "ethereum", "solana", "crypto", "cryptocurrency",
    "coinbase", "binance", "kraken", "defi", "nft", "blockchain",
    "stablecoin", "altcoin", "memecoin", "token", "web3",
    "satoshi", "halving", "on-chain", "onchain", "hashrate",
})

_MACRO_KEYWORDS = frozenset({
    "cpi", "inflation", "unemployment", "jobs", "payrolls", "gdp",
    "pce", "ppi", "nonfarm", "jobless", "consumer price",
    "producer price", "retail sales", "housing starts", "ism",
    "pmi", "trade deficit", "current account", "fiscal",
})

_FED_KEYWORDS = frozenset({
    "fed", "fomc", "federal reserve", "interest rate", "rate hike",
    "rate cut", "jerome powell", "powell", "basis points", "bps",
    "monetary policy", "quantitative", "qe", "qt", "taper",
    "hawkish", "dovish", "fed funds",
})

_WEATHER_KEYWORDS = frozenset({
    "hurricane", "storm", "temperature", "rain", "rainfall", "snowfall",
    "noaa", "weather", "tornado", "flood", "drought", "wildfire",
    "earthquake", "tsunami", "cyclone", "typhoon", "blizzard",
    "heat wave", "frost", "wind speed",
})

_SPORTS_KEYWORDS = frozenset({
    "nba", "nfl", "mlb", "nhl", "nascar", "soccer", "football",
    "basketball", "baseball", "tennis", "golf", "ufc", "mma",
    "match", "game", "championship", "playoffs", "tournament",
    "lakers", "celtics", "warriors", "patriots", "cowboys",
    "super bowl", "world series", "stanley cup", "finals",
    "wimbledon", "us open", "french open", "australian open",
})

_LEGAL_KEYWORDS = frozenset({
    "sec", "lawsuit", "court", "supreme court", "judge", "indictment",
    "charges", "verdict", "trial", "settlement", "injunction",
    "regulatory", "compliance", "violation", "penalty", "fine",
    "ftc", "doj", "cftc", "occ", "fincen", "cfpb",
    "etf approval", "bitcoin etf", "crypto etf",
})

_COMPANY_KEYWORDS = frozenset({
    "earnings", "stock", "shares", "ipo", "merger", "acquisition",
    "sec filing", "10-k", "10-q", "8-k", "annual report",
    "revenue", "profit", "eps", "guidance", "quarterly",
    "dividend", "buyback", "nasdaq", "nyse", "s&p",
    "apple", "microsoft", "tesla", "amazon", "google", "meta",
    "nvidia", "openai", "anthropic",
})

_ELECTION_KEYWORDS = frozenset({
    "election", "president", "senate", "congress", "house",
    "poll", "vote", "ballot", "candidate", "primary",
    "republican", "democrat", "gop", "midterm", "electoral",
    "approval rating", "polling", "swing state", "caucus",
    "biden", "trump", "harris",
})

_GEOPOLITICS_KEYWORDS = frozenset({
    "war", "ceasefire", "attack", "sanctions", "invasion",
    "missile", "conflict", "troops", "military", "nato",
    "ukraine", "russia", "china", "taiwan", "iran", "israel",
    "palestine", "hamas", "hezbollah", "coup", "nuclear",
    "treaty", "embargo", "diplomat", "un security",
})


def _tokenize(text: str) -> set[str]:
    """Lowercase, strip punctuation, extract word-level tokens + bigrams."""
    text = text.lower()
    words = [w.strip("?.,'\"!()[]%$#@&*/\\:;") for w in text.split()]
    words = [w for w in words if w]
    tokens: set[str] = set(words)
    # add bigrams
    for i in range(len(words) - 1):
        tokens.add(f"{words[i]} {words[i+1]}")
    return tokens


def _match_score(tokens: set[str], keywords: frozenset) -> int:
    return sum(1 for kw in keywords if kw in tokens)


class MarketSourceRouter:
    """
    Rule-based router that classifies a market question into a category
    and returns the appropriate primary news provider + fallbacks.
    No LLM, no external NLP library.
    """

    _PROVIDER_MAP: dict[MarketCategory, tuple[str, list[str]]] = {
        MarketCategory.CRYPTO_PRICE:      ("crypto_price_feed",   ["gdelt"]),
        MarketCategory.MACRO_RELEASE:     ("official_macro",      ["gdelt"]),
        MarketCategory.FED_RATES:         ("federal_reserve",     ["gdelt"]),
        MarketCategory.WEATHER:           ("noaa_nws",            ["gdelt"]),
        MarketCategory.SPORTS:            ("sports_data",         ["gdelt"]),
        MarketCategory.LEGAL_REGULATORY:  ("official_legal_sec",  ["gdelt"]),
        MarketCategory.COMPANY_SEC:       ("sec_edgar",           ["gdelt"]),
        MarketCategory.POLITICS_ELECTION: ("official_election",   ["gdelt"]),
        MarketCategory.GEOPOLITICS:       ("gdelt",               []),
        MarketCategory.GENERIC_NEWS:      ("gdelt",               ["newsapi"]),
        MarketCategory.UNKNOWN:           ("gdelt",               []),
    }

    def classify(
        self, market_question: str, market_description: str = ""
    ) -> SourceRoute:
        combined = f"{market_question} {market_description}"
        tokens = _tokenize(combined)

        scores: dict[MarketCategory, int] = {
            MarketCategory.FED_RATES:         _match_score(tokens, _FED_KEYWORDS),
            MarketCategory.CRYPTO_PRICE:      _match_score(tokens, _CRYPTO_KEYWORDS),
            MarketCategory.MACRO_RELEASE:     _match_score(tokens, _MACRO_KEYWORDS),
            MarketCategory.WEATHER:           _match_score(tokens, _WEATHER_KEYWORDS),
            MarketCategory.SPORTS:            _match_score(tokens, _SPORTS_KEYWORDS),
            MarketCategory.LEGAL_REGULATORY:  _match_score(tokens, _LEGAL_KEYWORDS),
            MarketCategory.COMPANY_SEC:       _match_score(tokens, _COMPANY_KEYWORDS),
            MarketCategory.POLITICS_ELECTION: _match_score(tokens, _ELECTION_KEYWORDS),
            MarketCategory.GEOPOLITICS:       _match_score(tokens, _GEOPOLITICS_KEYWORDS),
        }

        # FED_RATES must beat MACRO_RELEASE if both fire
        best_cat = max(scores, key=lambda c: (scores[c], _category_priority(c)))
        best_score = scores[best_cat]

        if best_score == 0:
            category = MarketCategory.GENERIC_NEWS
            reason = "no specific keyword match"
            confidence = 0.3
        else:
            category = best_cat
            total_tokens = max(len(tokens), 1)
            confidence = min(1.0, best_score / max(total_tokens * 0.15, 1))
            reason = f"matched {best_score} keyword(s) in {category.value}"

        primary, fallbacks = self._PROVIDER_MAP.get(
            category, ("gdelt", [])
        )

        logger.debug(
            f"[SOURCE_ROUTER] question={market_question[:60]!r} "
            f"→ category={category.value} primary={primary} "
            f"confidence={confidence:.2f} reason={reason}"
        )

        return SourceRoute(
            category=category,
            primary_provider=primary,
            fallback_providers=fallbacks,
            confidence=confidence,
            reason=reason,
        )


def _category_priority(cat: MarketCategory) -> int:
    """Higher = preferred when scores are equal (more specific first)."""
    order = [
        MarketCategory.FED_RATES,
        MarketCategory.CRYPTO_PRICE,
        MarketCategory.MACRO_RELEASE,
        MarketCategory.WEATHER,
        MarketCategory.SPORTS,
        MarketCategory.LEGAL_REGULATORY,
        MarketCategory.COMPANY_SEC,
        MarketCategory.POLITICS_ELECTION,
        MarketCategory.GEOPOLITICS,
        MarketCategory.GENERIC_NEWS,
    ]
    try:
        return len(order) - order.index(cat)
    except ValueError:
        return 0
