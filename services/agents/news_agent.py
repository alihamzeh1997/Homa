import datetime
from typing import List, Dict
from cachetools import TTLCache, cached
from agents_schema import Article, NewsSummary

# 30-minute cache (1800 seconds)
news_cache = TTLCache(maxsize=100, ttl=1800)

class NewsService:
    def __init__(self, api_client):
        self.api_client = api_client

    @cached(cache=news_cache)
    def fetch_and_process_news(self, symbol: str) -> NewsSummary:
        """Fetches raw news and parses it strictly into the NewsSummary schema."""
        raw_articles = self.api_client.get_articles(keyword=symbol)
        return self._aggregate_news(raw_articles)

    def _aggregate_news(self, raw_articles: List[Dict]) -> NewsSummary:
        count = len(raw_articles)
        now = datetime.datetime.now(datetime.timezone.utc)
        
        # Handle empty results gracefully
        if count == 0:
            return NewsSummary(
                article_count=0, 
                average_sentiment=0.5, 
                market_bias="NEUTRAL",
                top_headlines=[],
                last_fetched=now
            )

        # 1. Parse raw data into the internal Article schema
        internal_articles = []
        for raw in raw_articles:
            # Handle ISO datetime parsing from EventRegistry (e.g., '2026-04-17T20:57:35Z')
            pub_date_str = raw.get("dateTime", now.isoformat())
            try:
                pub_date = datetime.datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
            except ValueError:
                pub_date = now

            internal_articles.append(Article(
                title=raw.get("title", "Unknown"),
                body=raw.get("body", ""),
                source=raw.get("source", {}).get("title", "Unknown"),
                published_at=pub_date,
                raw_sentiment=raw.get("sentiment", 0.5)
            ))

        # 2. Compute aggregations for the NewsSummary
        avg_sentiment = sum(a.raw_sentiment for a in internal_articles) / count
        
        if avg_sentiment > 0.6:
            bias = "BULLISH"
        elif avg_sentiment < 0.4:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        # Grab top 3 titles based on API's relevance score
        sorted_raw = sorted(raw_articles, key=lambda x: x.get("relevance", 0), reverse=True)
        top_headlines = [a.get("title", "Unknown") for a in sorted_raw[:3]]

        return NewsSummary(
            article_count=count,
            average_sentiment=avg_sentiment,
            market_bias=bias,
            top_headlines=top_headlines,
            last_fetched=now
        )