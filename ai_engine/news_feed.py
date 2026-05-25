import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from utils.config_loader import AIEngineConfig

logger = logging.getLogger(__name__)

_GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_NEWSAPI_URL = "https://newsapi.org/v2/everything"
_MEDIASTACK_URL = "http://api.mediastack.com/v1/news"

_STOP_WORDS = frozenset({
    "will", "the", "a", "an", "in", "on", "at", "to", "for", "of",
    "and", "or", "is", "be", "by", "with", "this", "that", "it",
    "was", "are", "has", "have", "had", "does", "did", "not", "no",
    "if", "as", "but", "than", "when", "who", "what", "which",
    "from", "its", "been", "more", "over", "under", "into", "about",
})

_GDELT_DATE_FORMATS = [
    "%Y%m%dT%H%M%SZ",
    "%Y%m%d%H%M%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
]


@dataclass
class NewsEvent:
    headline: str
    source: str
    published_at: datetime
    url: str
    description: str = ""
    relevance_score: float = 0.0


def _extract_keywords(question: str, max_keywords: int = 8) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for w in question.split():
        w = w.strip("?.,'\"!()[]").strip()
        lw = w.lower()
        if len(w) >= 3 and lw not in _STOP_WORDS and lw not in seen:
            seen.add(lw)
            result.append(w)
        if len(result) >= max_keywords:
            break
    return result


def _score_relevance(article: dict, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    text = f"{article.get('title', '')} {article.get('description', '')}".lower()
    return sum(1 for kw in keywords if kw.lower() in text) / len(keywords)


def _redact(text: str, secret: str) -> str:
    if secret and secret in text:
        return text.replace(secret, "***")
    return text


def _parse_gdelt_date(raw: str) -> datetime | None:
    for fmt in _GDELT_DATE_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


class BaseNewsProvider:
    async def fetch_for_market(
        self, market_question: str, market_id: str
    ) -> list[NewsEvent]:
        raise NotImplementedError


class GDELTNewsProvider(BaseNewsProvider):
    def __init__(self, cfg: "AIEngineConfig") -> None:
        self._cfg = cfg

    async def fetch_for_market(
        self, market_question: str, market_id: str
    ) -> list[NewsEvent]:
        keywords = _extract_keywords(market_question, max_keywords=8)
        if not keywords:
            logger.debug(f"[GDELT_NEWS_FETCH] No keywords for market {market_id}")
            return []

        query = " ".join(keywords[:6])
        logger.info(f"[GDELT_NEWS_FETCH] market_id={market_id} query={query!r}")

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(_GDELT_URL, params={
                    "query": query,
                    "mode": "artlist",
                    "format": "json",
                    "maxrecords": self._cfg.news_page_size,
                    "sort": "datedesc",
                })
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.warning(f"[GDELT_NEWS_ERROR] market_id={market_id} error={e}")
            return []

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=self._cfg.news_max_age_minutes)
        seen_urls: set[str] = set()
        events: list[NewsEvent] = []

        for article in data.get("articles") or []:
            url = article.get("url") or ""
            title = article.get("title") or ""
            if not url or not title or url in seen_urls:
                continue
            seen_urls.add(url)

            raw_date = article.get("seendate") or ""
            pub_dt = _parse_gdelt_date(raw_date) if raw_date else None
            if pub_dt is None:
                logger.debug(
                    f"[GDELT_NEWS_FETCH] unparseable date {raw_date!r} for {url} — keeping"
                )
                pub_dt = now
            elif pub_dt < cutoff:
                continue

            source = (
                article.get("domain")
                or article.get("sourceCountry")
                or "gdelt"
            )
            events.append(NewsEvent(
                headline=title,
                source=str(source),
                published_at=pub_dt,
                url=url,
                description=article.get("snippet") or "",
                relevance_score=_score_relevance(article, keywords),
            ))

        events.sort(key=lambda e: (e.relevance_score, e.published_at), reverse=True)
        logger.info(
            f"[GDELT_NEWS_RESULT] market_id={market_id} count={len(events)}"
        )
        return events


class NewsAPINewsProvider(BaseNewsProvider):
    def __init__(self, cfg: "AIEngineConfig") -> None:
        self._cfg = cfg

    async def fetch_for_market(
        self, market_question: str, market_id: str
    ) -> list[NewsEvent]:
        if not self._cfg.news_api_key:
            logger.warning(
                f"[NEWSAPI_FETCH] Skipping market {market_id}: NEWS_API_KEY not set"
            )
            return []

        keywords = _extract_keywords(market_question, max_keywords=4)
        if not keywords:
            return []

        query = " ".join(keywords)
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=self._cfg.news_max_age_minutes)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(_NEWSAPI_URL, params={
                    "q": query,
                    "sortBy": "publishedAt",
                    "pageSize": self._cfg.news_page_size,
                    "apiKey": self._cfg.news_api_key,
                    "language": "en",
                    "from": cutoff.strftime("%Y-%m-%dT%H:%M:%S"),
                })
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            safe_err = _redact(str(e), self._cfg.news_api_key)
            logger.warning(
                f"[NEWSAPI_FETCH] NewsAPI error for market {market_id}: {safe_err}"
            )
            return []

        events: list[NewsEvent] = []
        seen_urls: set[str] = set()
        for article in data.get("articles", []):
            url = article.get("url") or ""
            title = article.get("title") or ""
            if not url or not title or url in seen_urls:
                continue
            seen_urls.add(url)

            pub_str = article.get("publishedAt") or ""
            try:
                pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            if pub_dt < cutoff:
                continue

            events.append(NewsEvent(
                headline=title,
                source=(article.get("source") or {}).get("name") or "newsapi",
                published_at=pub_dt,
                url=url,
                description=article.get("description") or "",
                relevance_score=_score_relevance(article, keywords),
            ))

        events.sort(key=lambda e: (e.relevance_score, e.published_at), reverse=True)
        logger.info(
            f"[NEWSAPI_FETCH] market_id={market_id} query={query!r} count={len(events)}"
        )
        return events


class MediastackNewsProvider(BaseNewsProvider):
    def __init__(self, cfg: "AIEngineConfig") -> None:
        self._cfg = cfg

    async def fetch_for_market(
        self, market_question: str, market_id: str
    ) -> list[NewsEvent]:
        if not self._cfg.mediastack_api_key:
            logger.warning(
                f"[MEDIASTACK_FETCH] Skipping market {market_id}: MEDIASTACK_API_KEY not set"
            )
            return []

        keywords = _extract_keywords(market_question, max_keywords=5)
        if not keywords:
            return []

        query = ",".join(keywords)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(_MEDIASTACK_URL, params={
                    "access_key": self._cfg.mediastack_api_key,
                    "keywords": query,
                    "languages": "en",
                    "limit": self._cfg.news_page_size,
                    "sort": "published_desc",
                })
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            safe_err = _redact(str(e), self._cfg.mediastack_api_key)
            logger.warning(
                f"[MEDIASTACK_FETCH] Error for market {market_id}: {safe_err}"
            )
            return []

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=self._cfg.news_max_age_minutes)
        events: list[NewsEvent] = []
        seen_urls: set[str] = set()

        for article in (data.get("data") or []):
            url = article.get("url") or ""
            title = article.get("title") or ""
            if not url or not title or url in seen_urls:
                continue
            seen_urls.add(url)

            pub_str = article.get("published_at") or ""
            try:
                pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            if pub_dt < cutoff:
                continue

            events.append(NewsEvent(
                headline=title,
                source=article.get("source") or "mediastack",
                published_at=pub_dt,
                url=url,
                description=article.get("description") or "",
                relevance_score=_score_relevance(article, keywords),
            ))

        events.sort(key=lambda e: (e.relevance_score, e.published_at), reverse=True)
        logger.info(
            f"[MEDIASTACK_FETCH] market_id={market_id} count={len(events)}"
        )
        return events


# Providers that are on the roadmap but not yet implemented in Phase 1.
# They gracefully return [] with a log message.
class _UnimplementedProvider(BaseNewsProvider):
    def __init__(self, name: str, gdelt_fallback: "GDELTNewsProvider") -> None:
        self._name = name
        self._gdelt = gdelt_fallback

    async def fetch_for_market(
        self, market_question: str, market_id: str
    ) -> list[NewsEvent]:
        logger.info(
            f"[SOURCE_ROUTE_UNIMPLEMENTED] provider={self._name} market={market_id} "
            f"— falling back to GDELT"
        )
        return await self._gdelt.fetch_for_market(market_question, market_id)


class NewsFeedPoller:
    def __init__(self, cfg: "AIEngineConfig") -> None:
        self._cfg = cfg
        gdelt = GDELTNewsProvider(cfg)

        self._providers: dict[str, BaseNewsProvider] = {
            "gdelt": gdelt,
            "newsapi": NewsAPINewsProvider(cfg),
            "mediastack": MediastackNewsProvider(cfg),
            # Roadmap providers — fall back to GDELT until implemented
            "crypto_price_feed": _UnimplementedProvider("crypto_price_feed", gdelt),
            "official_macro":    _UnimplementedProvider("official_macro",    gdelt),
            "federal_reserve":   _UnimplementedProvider("federal_reserve",   gdelt),
            "noaa_nws":          _UnimplementedProvider("noaa_nws",          gdelt),
            "sports_data":       _UnimplementedProvider("sports_data",       gdelt),
            "official_legal_sec":_UnimplementedProvider("official_legal_sec",gdelt),
            "sec_edgar":         _UnimplementedProvider("sec_edgar",         gdelt),
            "official_election": _UnimplementedProvider("official_election", gdelt),
        }

        if cfg.source_router_enabled:
            from ai_engine.source_router import MarketSourceRouter
            self._router: MarketSourceRouter | None = MarketSourceRouter()
        else:
            self._router = None

    async def fetch_for_market(
        self, market_question: str, market_id: str
    ) -> list[NewsEvent]:
        provider_name = self._cfg.news_provider

        if provider_name == "router" and self._router is not None:
            return await self._fetch_via_router(market_question, market_id)

        provider = self._providers.get(provider_name)
        if provider is None:
            logger.warning(
                f"[NEWS_POLLER] Unknown provider '{provider_name}' — falling back to GDELT"
            )
            provider = self._providers["gdelt"]

        return await provider.fetch_for_market(market_question, market_id)

    async def _fetch_via_router(
        self, market_question: str, market_id: str
    ) -> list[NewsEvent]:
        assert self._router is not None
        route = self._router.classify(market_question)

        logger.info(
            f"[SOURCE_ROUTE] market={market_id} "
            f"category={route.category.value} "
            f"primary={route.primary_provider} "
            f"confidence={route.confidence:.2f}"
        )

        # Category gate
        disabled = set(self._cfg.disabled_categories)
        if route.category.value in disabled:
            logger.info(
                f"[SOURCE_ROUTE_DISABLED] market={market_id} "
                f"category={route.category.value} — skipped by disabled_categories"
            )
            return []

        # Try primary provider
        primary = self._providers.get(route.primary_provider)
        if primary is not None:
            results = await primary.fetch_for_market(market_question, market_id)
            if results:
                return results

        # Try route fallbacks
        for fb_name in route.fallback_providers:
            fb = self._providers.get(fb_name)
            if fb is not None:
                results = await fb.fetch_for_market(market_question, market_id)
                if results:
                    return results

        # Final fallback from config
        final_fb = self._providers.get(self._cfg.fallback_news_provider)
        if final_fb is not None:
            return await final_fb.fetch_for_market(market_question, market_id)

        return []
