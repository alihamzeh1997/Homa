import os
import asyncio
import datetime
from typing import List, Dict, Any

from cachetools import TTLCache, cached
from cachetools.keys import hashkey
from dotenv import load_dotenv
from loguru import logger

from services.agents.agents_schema import Article, NewsSummary
from services.agents.state_schema import GraphState

load_dotenv()

# 30-minute module-level cache — shared across all NewsService instances
_news_cache: TTLCache = TTLCache(maxsize=100, ttl=1800)


class _EventRegistryClient:
    """Thin wrapper around the eventregistry SDK that matches the NewsService interface."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def get_articles(self, keyword: str) -> List[Dict]:
        from eventregistry import EventRegistry, QueryArticlesIter

        er = EventRegistry(apiKey=self._api_key, allowUseOfArchive=False)
        
        # Remove the invalid keyword arguments from the constructor
        q = QueryArticlesIter(
            keywords=keyword,
            lang="eng"
        )
        
        # Use the .set_social_score_sort() or similar if needed, 
        # but for standard execution, most sorting is now default or handled by er.execQuery
        articles: List[Dict] = []
        
        # Pass the sorting parameters to execQuery instead
        for art in q.execQuery(er, maxItems=20, sortBy="date", sortByAsc=False):
            articles.append(art)
        return articles


class NewsService:
    def __init__(self, api_client: _EventRegistryClient) -> None:
        self.api_client = api_client

    # key=lambda ignores `self` so the TTL cache works across instances
    @cached(cache=_news_cache, key=lambda self, symbol: hashkey(symbol))
    def fetch_and_process_news(self, symbol: str) -> NewsSummary:
        """Fetches raw news and parses it into the NewsSummary schema."""
        raw_articles = self.api_client.get_articles(keyword=symbol)
        return self._aggregate_news(raw_articles)

    def _aggregate_news(self, raw_articles: List[Dict]) -> NewsSummary:
        count = len(raw_articles)
        now = datetime.datetime.now(datetime.timezone.utc)

        if count == 0:
            return NewsSummary(
                article_count=0,
                average_sentiment=0.5,
                market_bias="NEUTRAL",
                top_headlines=[],
                last_fetched=now,
            )

        internal_articles: List[Article] = []
        for raw in raw_articles:
            pub_date_str = raw.get("dateTime", now.isoformat())
            raw_val = raw.get("sentiment", 0.5)
            clamped_sentiment = max(0.0, min(1.0, raw_val))
            try:
                pub_date = datetime.datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
            except ValueError:
                pub_date = now

            internal_articles.append(
                Article(
                    title=raw.get("title", "Unknown"),
                    body=raw.get("body", ""),
                    source=raw.get("source", {}).get("title", "Unknown"),
                    published_at=pub_date,
                    raw_sentiment=clamped_sentiment,
                )
            )

        avg_sentiment = sum(a.raw_sentiment for a in internal_articles) / count

        if avg_sentiment > 0.6:
            bias = "BULLISH"
        elif avg_sentiment < 0.4:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        sorted_raw = sorted(raw_articles, key=lambda x: x.get("relevance", 0), reverse=True)
        top_headlines = [a.get("title", "Unknown") for a in sorted_raw[:3]]

        return NewsSummary(
            article_count=count,
            average_sentiment=avg_sentiment,
            market_bias=bias,
            top_headlines=top_headlines,
            last_fetched=now,
        )


# ------------------------------------------------------------------ node
async def news_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph Node: Fetches and aggregates crypto news via EventRegistry.
    Runs in parallel with market_data_node. Falls back to None gracefully
    so the rest of the pipeline continues without news context.
    """
    symbol = state.get("symbol", "BTC")
    api_key = os.getenv("EVENTREGISTRY_API_KEY")

    if not api_key:
        logger.warning("📰 No EVENTREGISTRY_API_KEY found. Skipping news fetch.")
        return {"news": None}

    try:
        client = _EventRegistryClient(api_key=api_key)
        service = NewsService(api_client=client)

        # EventRegistry SDK is synchronous — offload to thread pool
        summary: NewsSummary = await asyncio.to_thread(
            service.fetch_and_process_news, symbol
        )

        logger.info(
            f"📰 News fetched: {summary.article_count} articles | "
            f"bias={summary.market_bias} | sentiment={summary.average_sentiment:.2f}"
        )
        return {"news": summary}

    except Exception as e:
        logger.error(f"📰 News fetch failed: {e}. Continuing without news context.")
        return {"news": None}
