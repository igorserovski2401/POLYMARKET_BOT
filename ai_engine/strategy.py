import asyncio
import logging
from typing import Optional
from uuid import uuid4

from polymarket_client.models import Market, OrderBook, Signal, OrderSide, TokenType

logger = logging.getLogger(__name__)

_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


class LLMNewsStrategy:
    def __init__(self, ai_agent, news_poller, cfg) -> None:
        self._agent = ai_agent
        self._poller = news_poller
        self._cfg = cfg

    async def evaluate_market(
        self,
        market: Optional[Market],
        orderbook: Optional[OrderBook],
    ) -> Optional[Signal]:
        try:
            # Gate 1 — Market status
            if market is None:
                logger.debug("[LLM_REJECTED] reason=no_market")
                return None
            market_id = market.market_id
            if not market.active:
                logger.debug(f"[LLM_REJECTED] market={market_id} reason=inactive")
                return None
            if market.closed:
                logger.debug(f"[LLM_REJECTED] market={market_id} reason=closed")
                return None
            if market.resolved:
                logger.debug(f"[LLM_REJECTED] market={market_id} reason=resolved")
                return None
            if orderbook is None:
                logger.debug(f"[LLM_REJECTED] market={market_id} reason=no_orderbook")
                return None

            # Gate 2 — News
            news = await self._poller.fetch_for_market(market.question, market_id)
            if not news:
                logger.info(f"[LLM_REJECTED] market={market_id} reason=no_news")
                return None

            # Gate 3 — Claude assessment
            assessment = await self._agent.assess_market(market, orderbook, news)
            if assessment is None:
                logger.info(f"[LLM_REJECTED] market={market_id} reason=no_assessment")
                return None
            if not assessment.should_trade:
                logger.info(f"[LLM_REJECTED] market={market_id} reason=llm_no_trade")
                return None

            # Gate 4 — Confidence
            sig_rank = _CONFIDENCE_RANK.get(assessment.confidence, 0)
            min_rank = _CONFIDENCE_RANK.get(self._cfg.min_confidence, 2)
            if sig_rank < min_rank:
                logger.info(
                    f"[LLM_REJECTED] market={market_id} reason=low_confidence "
                    f"got={assessment.confidence} required={self._cfg.min_confidence}"
                )
                return None

            prob_yes = assessment.probability_yes

            # Gate 5 — Orderbook checks per direction (separate)
            yes_candidate = None
            no_candidate = None

            yes_bid = orderbook.best_bid_yes
            yes_ask = orderbook.best_ask_yes
            if yes_bid is not None and yes_ask is not None:
                yes_spread = yes_ask - yes_bid
                if yes_spread <= self._cfg.max_spread:
                    yes_edge = prob_yes - yes_ask
                    if yes_edge >= self._cfg.min_edge:
                        yes_size_avail = orderbook.yes.best_ask_size if orderbook.yes else None
                        yes_candidate = (yes_edge, TokenType.YES, yes_ask, yes_size_avail)
                else:
                    logger.debug(
                        f"[LLM_REJECTED] market={market_id} reason=yes_spread_too_wide "
                        f"spread={yes_spread:.4f} max={self._cfg.max_spread}"
                    )

            if self._cfg.long_only:
                no_bid = orderbook.best_bid_no
                no_ask = orderbook.best_ask_no
                if no_bid is not None and no_ask is not None:
                    no_spread = no_ask - no_bid
                    if no_spread <= self._cfg.max_spread:
                        no_edge = (1.0 - prob_yes) - no_ask
                        if no_edge >= self._cfg.min_edge:
                            no_size_avail = orderbook.no.best_ask_size if orderbook.no else None
                            no_candidate = (no_edge, TokenType.NO, no_ask, no_size_avail)
                    else:
                        logger.debug(
                            f"[LLM_REJECTED] market={market_id} reason=no_spread_too_wide "
                            f"spread={no_spread:.4f} max={self._cfg.max_spread}"
                        )

            if yes_candidate is None and no_candidate is None:
                yes_edge_val = (prob_yes - yes_ask) if yes_ask is not None else None
                no_edge_val = ((1.0 - prob_yes) - no_ask) if no_ask is not None else None
                logger.info(
                    f"[LLM_REJECTED] market={market_id} reason=insufficient_edge "
                    f"yes_edge={yes_edge_val} no_edge={no_edge_val} min_edge={self._cfg.min_edge}"
                )
                return None

            # Choose the candidate with higher edge
            if yes_candidate and no_candidate:
                best = yes_candidate if yes_candidate[0] >= no_candidate[0] else no_candidate
            elif yes_candidate:
                best = yes_candidate
            else:
                best = no_candidate

            edge, token_type, price, available_size = best

            # Gate 6 — Price sanity
            if price <= 0 or price >= 1:
                logger.warning(
                    f"[LLM_REJECTED] market={market_id} reason=invalid_price price={price}"
                )
                return None

            # Size calculation
            notional = min(self._cfg.default_order_size, self._cfg.max_order_size)
            size = notional / price
            if available_size is not None:
                size = min(size, available_size)
                if size * price < notional * 0.1:
                    logger.info(
                        f"[LLM_REJECTED] market={market_id} reason=size_limited_by_liquidity "
                        f"available={available_size:.2f} price={price:.4f}"
                    )
                    return None
            if size <= 0:
                logger.warning(
                    f"[LLM_REJECTED] market={market_id} reason=invalid_size size={size}"
                )
                return None

            signal = Signal(
                signal_id=f"llm_news_{uuid4().hex[:8]}",
                action="place_orders",
                market_id=market_id,
                orders=[{
                    "token_type": token_type,
                    "side": OrderSide.BUY,
                    "price": price,
                    "size": size,
                    "strategy_tag": "llm_news",
                }],
                priority=3,
            )

            logger.info(
                f"[LLM_SIGNAL_CREATED] signal={signal.signal_id} market={market_id} "
                f"token={token_type.value} price={price:.4f} size={size:.2f} "
                f"edge={edge:.4f} confidence={assessment.confidence}"
            )
            return signal

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(
                f"[LLM_STRATEGY] Unexpected error for market={getattr(market, 'market_id', '?')}: {e}",
                exc_info=True,
            )
            return None
