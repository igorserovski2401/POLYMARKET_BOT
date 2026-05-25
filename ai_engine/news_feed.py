import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import httpx

logger = logging.getLogger(__name__)

_NEWS_API_URL = "https://newsapi.org/v2/everything"

_STOP_WORDS = frozenset({
    "will", "the", "a", "an", "in", "on", "at", "to", "for", "of",
    "and", "or", "is", "be", "by", "with", "this", "that", "it",
    "was", "are", "has", "have", "had", "does", "did", "not", "no",
    "if", "as", "but", "than", "when", "who", "what", "which",
    "from", "its", "been", "more", "over", "under", "into", "about",
})


@dataclass
class NewsEvent:
    headline: str
    source: str
    published_at: datetime
    url: str
    description: str = ""
    relevance_score: float = 0.0


class NewsFeedPoller:
    def __init__(self, api_key: str, page_size: int = 10, max_age_minutes: int = 360):
        self._api_key = api_key
        self._page_size = page_size
        self._max_age = timedelta(minutes=max_age_minutes)
        self._seen_urls: set[str] = set()

    async def fetch_for_market(self, market_question: str, market_id: str) -> list[NewsEvent]:
        keywords = self._extract_keywords(market_question)
        if not keywords:
            logger.debug(f"[LLM_NEWS_FETCH] No keywords for market {market_id}")
            return []

        query = " ".join(keywords[:4])
        now = datetime.now(timezone.utc)
        cutoff = now - self._max_age

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(_NEWS_API_URL, params={
                    "q": query,
                    "sortBy": "publishedAt",
                    "pageSize": self._page_size,
                    "apiKey": self._api_key,
                    "language": "en",
                    "from": cutoff.strftime("%Y-%m-%dT%H:%M:%S"),
                })
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            # Mask full URL to prevent apiKey from appearing in logs
            safe_err = str(e)
            if self._api_key and self._api_key in safe_err:
                safe_err = safe_err.replace(self._api_key, "***")
            logger.warning(f"[LLM_NEWS_FETCH] NewsAPI error for market {market_id}: {safe_err}")
            return []

        events: list[NewsEvent] = []
        for article in data.get("articles", []):
            url = article.get("url") or ""
            title = article.get("title") or ""
            if not url or not title or url in self._seen_urls:
                continue

            pub_str = article.get("publishedAt") or ""
            try:
                pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue

            if pub_dt < cutoff:
                continue

            self._seen_urls.add(url)
            events.append(NewsEvent(
                headline=title,
                source=(article.get("source") or {}).get("name") or "unknown",
                published_at=pub_dt,
                url=url,
                description=article.get("description") or "",
                relevance_score=self._score_relevance(article, keywords),
            ))

        events.sort(key=lambda e: (e.relevance_score, e.published_at), reverse=True)
        logger.info(
            f"[LLM_NEWS_FETCH] market={market_id} query={query!r} "
            f"got={len(events)} fresh articles"
        )
        return events

    def _extract_keywords(self, question: str) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for w in question.split():
            w = w.strip("?.,'\"!()[]").strip()
            lw = w.lower()
            if len(w) >= 3 and lw not in _STOP_WORDS and lw not in seen:
                seen.add(lw)
                result.append(w)
        return result

    def _score_relevance(self, article: dict, keywords: list[str]) -> float:
        if not keywords:
            return 0.0
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        return sum(1 for kw in keywords if kw.lower() in text) / len(keywords)
