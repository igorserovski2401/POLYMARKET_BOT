import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import anthropic

from .news_feed import NewsEvent
from polymarket_client.models import Market, OrderBook

logger = logging.getLogger(__name__)

_ASSESSMENT_TOOL = {
    "name": "submit_assessment",
    "description": "Submit your probability estimate for this binary prediction market.",
    "input_schema": {
        "type": "object",
        "properties": {
            "probability_yes": {
                "type": "number",
                "description": "Estimated probability of YES outcome (0.0 to 1.0).",
            },
            "confidence": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Confidence level based on news evidence quality.",
            },
            "should_trade": {
                "type": "boolean",
                "description": (
                    "True only if recent news is concrete, directional, and materially "
                    "different from what the current market price implies."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "One-sentence reasoning for the estimate (max 200 chars).",
            },
        },
        "required": ["probability_yes", "confidence", "should_trade", "reasoning"],
    },
}


@dataclass
class ClaudeAssessment:
    market_id: str
    probability_yes: float
    confidence: str
    should_trade: bool
    reasoning: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AIAgent:
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        calls_per_minute: int = 10,
        max_tokens: int = 512,
    ):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._calls_per_minute = calls_per_minute
        self._call_times: list[datetime] = []
        # Cache: cache_key -> (assessment, cached_at)
        self._cache: dict[str, tuple[ClaudeAssessment, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)

    async def assess_market(
        self,
        market: Market,
        orderbook: OrderBook,
        news_events: list[NewsEvent],
    ) -> Optional[ClaudeAssessment]:
        # Cache check — keyed by market + news count to invalidate on new articles
        cache_key = f"{market.market_id}:{len(news_events)}"
        cached = self._cache.get(cache_key)
        if cached:
            assessment, cached_at = cached
            if datetime.now(timezone.utc) - cached_at < self._cache_ttl:
                return assessment

        await self._apply_rate_limit()

        prompt = self._build_prompt(market, orderbook, news_events)
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                tools=[_ASSESSMENT_TOOL],
                tool_choice={"type": "any"},
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            logger.warning(f"[LLM_ASSESSMENT] Claude API error for {market.market_id}: {e}")
            return None

        assessment = self._parse_response(market.market_id, response)
        if assessment:
            self._cache[cache_key] = (assessment, datetime.now(timezone.utc))
            logger.info(
                f"[LLM_ASSESSMENT] market={market.market_id} "
                f"prob_yes={assessment.probability_yes:.3f} "
                f"conf={assessment.confidence} trade={assessment.should_trade} | "
                f"{assessment.reasoning[:100]}"
            )
        return assessment

    def _build_prompt(
        self, market: Market, orderbook: OrderBook, news: list[NewsEvent]
    ) -> str:
        yes_bid = f"{orderbook.best_bid_yes:.3f}" if orderbook.best_bid_yes is not None else "n/a"
        yes_ask = f"{orderbook.best_ask_yes:.3f}" if orderbook.best_ask_yes is not None else "n/a"
        no_bid = f"{orderbook.best_bid_no:.3f}" if orderbook.best_bid_no is not None else "n/a"
        no_ask = f"{orderbook.best_ask_no:.3f}" if orderbook.best_ask_no is not None else "n/a"
        end_date = market.end_date.strftime("%Y-%m-%d") if market.end_date else "unknown"

        news_lines = "\n".join(
            f"  - [{e.source} {e.published_at.strftime('%m-%d %H:%M')}] "
            f"{e.headline}"
            for e in news[:5]
        ) or "  No recent news found."

        return (
            "You are a prediction market analyst.\n\n"
            f"QUESTION: {market.question}\n"
            f"DESCRIPTION: {market.description or 'None'}\n"
            f"RESOLVES: {end_date}\n"
            f"24H VOLUME: ${market.volume_24h:,.0f}\n\n"
            "ORDER BOOK:\n"
            f"  YES bid={yes_bid}  ask={yes_ask}\n"
            f"  NO  bid={no_bid}  ask={no_ask}\n\n"
            "RECENT NEWS (last 6h):\n"
            f"{news_lines}\n\n"
            "INSTRUCTIONS:\n"
            "- Estimate probability_yes (0.0–1.0).\n"
            "- Set confidence=high only if news is unambiguous and recent.\n"
            "- Set should_trade=true ONLY if news is concrete, directional, and "
            "materially changes the probability versus current market prices.\n"
            "- If news is vague, old, or consistent with current price: should_trade=false.\n"
            "- Do not invent facts. Use the submit_assessment tool."
        )

    def _parse_response(self, market_id: str, response) -> Optional[ClaudeAssessment]:
        for block in response.content:
            if block.type != "tool_use" or block.name != "submit_assessment":
                continue
            inp = block.input
            prob = inp.get("probability_yes")
            conf = inp.get("confidence")
            trade = inp.get("should_trade")
            reason = inp.get("reasoning", "")

            if not isinstance(prob, (int, float)) or not (0.0 <= float(prob) <= 1.0):
                logger.warning(
                    f"[LLM_ASSESSMENT] Invalid probability_yes={prob} for {market_id}"
                )
                return None
            if conf not in ("low", "medium", "high"):
                logger.warning(
                    f"[LLM_ASSESSMENT] Invalid confidence={conf!r} for {market_id}"
                )
                return None

            return ClaudeAssessment(
                market_id=market_id,
                probability_yes=float(prob),
                confidence=conf,
                should_trade=bool(trade),
                reasoning=str(reason)[:200],
            )

        logger.warning(
            f"[LLM_ASSESSMENT] No submit_assessment tool_use block for {market_id}"
        )
        return None

    async def _apply_rate_limit(self) -> None:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=60)
        self._call_times = [t for t in self._call_times if t > cutoff]
        if len(self._call_times) >= self._calls_per_minute:
            oldest = self._call_times[0]
            wait = 60.0 - (now - oldest).total_seconds()
            if wait > 0:
                logger.debug(f"[LLM_ASSESSMENT] Rate limit: sleeping {wait:.1f}s")
                await asyncio.sleep(wait)
        self._call_times.append(datetime.now(timezone.utc))
