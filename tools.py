## tools.py - external tools the Researcher agent can call
import re
import feedparser
import yfinance as yf
from urllib.parse import quote
from langchain_core.tools import tool


## NOTE: the text inside """ """ is the tool description. The LLM reads it to decide which tool to use.
@tool
def get_stock_price(ticker: str) -> str:
    """Get the latest closing price and daily % change of a stock. Use US tickers like AAPL, TSLA or Indian ones like TCS.NS, RELIANCE.NS."""
    try:
        hist = yf.Ticker(ticker.strip().upper()).history(period="5d")
        if hist.empty:  ## wrong ticker or no data -> return an error the agent can learn from
            return f"ERROR: no price data for '{ticker}'. Check the ticker (Indian stocks need .NS, e.g. TCS.NS)."
        last = hist["Close"].iloc[-1]
        prev = hist["Close"].iloc[-2] if len(hist) > 1 else last
        change = (last - prev) / prev * 100
        return f"{ticker.upper()}: last close {last:.2f}, previous close {prev:.2f}, change {change:+.2f}% (date {hist.index[-1].date()})"
    except Exception as e:
        return f"ERROR: price lookup failed ({e})"


@tool
def get_financial_news(topic: str) -> str:
    """Search the latest financial news (last 7 days) about a company, market or topic. Returns top 5 headlines with links."""
    try:
        url = f"https://news.google.com/rss/search?q={quote(topic)}+when:7d&hl=en-IN&gl=IN&ceid=IN:en"
        entries = feedparser.parse(url).entries[:5]
        if not entries:  ## nothing found -> tell the agent to try a broader topic
            return f"ERROR: no recent news found for '{topic}'. Try a broader or simpler topic."
        return "\n".join(f"- {e.title} | {e.get('published', '')[:16]} | {e.link}" for e in entries)
    except Exception as e:
        return f"ERROR: news search failed ({e})"


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression, e.g. percentage change: (250-200)/200*100. Only numbers and + - * / ( ) allowed."""
    ## simple safety check: only allow digits and basic operators (no code, no powers)
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expression) or "**" in expression:
        return "ERROR: only numbers and + - * / ( ) are allowed."
    try:
        return str(round(eval(expression, {"__builtins__": {}}, {}), 4))
    except Exception as e:
        return f"ERROR: could not calculate ({e})"


TOOLS = [get_stock_price, get_financial_news, calculator]
TOOL_MAP = {t.name: t for t in TOOLS}  ## lets us find a tool by the name the LLM gives


def run_tool(name, args):
    ## runs one tool safely - any crash becomes an ERROR text so the agent can recover
    if name not in TOOL_MAP:
        return f"ERROR: tool '{name}' does not exist. Use one of {list(TOOL_MAP)}."
    try:
        return str(TOOL_MAP[name].invoke(args))
    except Exception as e:
        return f"ERROR: tool crashed ({e})"
