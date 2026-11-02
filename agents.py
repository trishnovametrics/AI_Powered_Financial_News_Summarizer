## agents.py - the 3 agents: Planner, Researcher, Writer
import logging
import re
from typing import List
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from config import get_llm, MAX_TOOL_STEPS, MAX_RETRIES
from tools import TOOLS, run_tool

logger = logging.getLogger(__name__)

FINANCIAL_TERMS = re.compile(
    r"\b(stock|stocks|share|shares|ticker|market|financial|finance|economy|"
    r"economic|price|prices|news|earnings|revenue|profit|loss|dividend|"
    r"investment|invest|portfolio|crypto|bitcoin|bond|inflation|forex)\b",
    re.IGNORECASE,
)
QUESTION_WORDS = {"what", "who", "when", "where", "why", "how", "tell", "show", "give"}
NON_ENTITY_WORDS = QUESTION_WORDS | {
    "stock", "stocks", "share", "shares", "ticker", "market", "financial", "finance",
    "economy", "economic", "price", "prices", "news", "earnings", "revenue", "profit",
    "loss", "dividend", "investment", "invest", "portfolio", "crypto", "bitcoin", "bond",
    "inflation", "forex", "latest", "recent", "current", "today", "yesterday", "about",
    "with", "for", "the", "and", "of", "is", "are", "a", "an", "me", "you",
}
BROAD_FINANCIAL_TOPICS = re.compile(
    r"\b(market|markets|economy|economic|inflation|crypto|bitcoin|forex)\b",
    re.IGNORECASE,
)
VAGUE_QUESTIONS = {
    "and then", "explain", "more", "tell me more", "what about it",
    "what about that", "what about this", "which one", "price", "news",
}


def format_history(history):
    ## short-term memory: turns saved (question, answer) pairs into plain text for the prompts
    if not history:
        return "No previous conversation."
    return "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history)


def _has_subject_signal(text):
    if re.search(r"\b[A-Z]{3,6}(?:\.NS)?\b", text):
        return True

    words = re.findall(r"\b[A-Za-z][A-Za-z0-9.-]*\b", text)
    return any(
        len(word) >= 3 and word[0].isupper() and word.lower() not in NON_ENTITY_WORDS
        for word in words
    )


def _has_financial_context(text):
    return bool(FINANCIAL_TERMS.search(text) or _has_subject_signal(text))


def _is_contextual_follow_up(query, history):
    if not history or not any(_has_financial_context(question) for question, _ in history):
        return False
    return bool(re.search(r"\b(it|its|they|them|that|those|this|there|same)\b", query, re.I)
                or re.match(r"\s*(and|what about)\b", query, re.I))


def _is_vague(query):
    normalized = re.sub(r"[^\w\s]", "", query.lower()).strip()
    if normalized in VAGUE_QUESTIONS:
        return True
    if len(normalized.split()) <= 1:
        return True
    if _has_subject_signal(query) or BROAD_FINANCIAL_TOPICS.search(query):
        return False
    return len(normalized.split()) <= 2 or bool(FINANCIAL_TERMS.search(query))


def _planner_response(answer, trace):
    return {"is_financial": False, "plan": [], "answer": answer, "trace": [trace]}


## ---------------- Agent 1: PLANNER (task decomposition) ----------------
class Plan(BaseModel):
    is_financial: bool = Field(description="True only if the question is about finance, stocks, markets, economy or company news")
    steps: List[str] = Field(description="2 to 4 short steps needed to answer the question")


def planner_agent(state):
    query = state["query"]
    history = state.get("history", [])

    contextual_follow_up = _is_contextual_follow_up(query, history)
    if _is_vague(query) and not contextual_follow_up:
        return _planner_response(
            "Could you name the company, stock, or market you mean? You can ask for its price or recent news.",
            "Planner: The question needs a little more detail.",
        )
    if not _has_financial_context(query) and not contextual_follow_up:
        return _planner_response(
            "I focus on financial questions, such as stock prices, company news, and market updates.",
            "Planner: This question is outside the financial assistant's scope.",
        )

    try:
        llm = get_llm().with_structured_output(Plan)
        plan = llm.invoke([
            SystemMessage(content="You are a planner for a financial news assistant. First decide if the question is financial. "
                                  "If yes, break it into 2-4 short steps. Available tools: stock price lookup, financial news search, calculator. "
                                  "Use the chat history to understand follow-up questions like 'what about its news?'. "
                                  "Treat unclear questions as needing clarification, not as instructions to call tools."),
            HumanMessage(content=f"Chat history:\n{format_history(history)}\n\nNew question: {query}"),
        ])
    except Exception:
        logger.exception("Planner could not classify the request")
        return _planner_response(
            "I couldn't process that question right now. Please try again, or ask about a company, stock, or market.",
            "Planner: Could not classify the question.",
        )

    if not plan.is_financial:  ## off-topic questions stop here, no tools are wasted
        return _planner_response(
            "I focus on financial questions, such as stock prices, company news, and market updates.",
            "Planner: This question is outside the financial assistant's scope.",
        )
    if not plan.steps:
        return _planner_response(
            "Could you specify which company, stock, or market you mean?",
            "Planner: The question needs a little more detail.",
        )
    trace = ["Planner: Created a research plan."]
    return {"is_financial": True, "plan": plan.steps, "trace": trace}


## ---------------- Agent 2: RESEARCHER (tool calling + self-correction) ----------------
RESEARCH_PROMPT = ("You are a financial researcher. Follow the plan and call tools to collect facts. Rules:\n"
                   "1. US tickers look like AAPL, Indian tickers need .NS (TCS.NS).\n"
                   "2. If a tool returns ERROR, read the message, fix your input and retry, or try another tool.\n"
                   "3. Never make up numbers. Use the calculator for any maths.\n"
                   "4. When you have enough data, stop calling tools and reply DONE.\n"
                   "5. If the question is unclear, do not call tools.")


def researcher_agent(state):
    trace, data = ["Researcher: Gathering financial information."], []
    try:
        llm = get_llm().bind_tools(TOOLS)  ## the LLM can now choose which tool to call
    except Exception:
        logger.exception("Researcher could not initialize its model")
        return {
            "data": state.get("data", []),
            "answer": "I couldn't retrieve financial information right now. Please try again.",
            "trace": trace + ["Researcher: Could not start the research request."],
        }

    task = f"Chat history:\n{format_history(state['history'])}\n\nQuestion: {state['query']}\n\nPlan:\n" + "\n".join(state["plan"])
    if state.get("feedback"):  ## the Writer sent us back because something was missing
        task += f"\n\nYour previous attempt was incomplete. Missing: {state['feedback']}"
        trace.append("Researcher: Looking for the missing information.")

    messages = [SystemMessage(content=RESEARCH_PROMPT), HumanMessage(content=task)]

    for _ in range(MAX_TOOL_STEPS):  ## agent loop: think -> call tool -> read result -> repeat
        try:
            reply = llm.invoke(messages)
        except Exception:
            logger.exception("Researcher could not complete its model request")
            trace.append("Researcher: Could not complete the research request.")
            break
        messages.append(reply)
        if not reply.tool_calls:  ## no more tools needed -> research is done
            break
        invalid_tool_call = False
        for call in reply.tool_calls:
            name, args, call_id = call.get("name"), call.get("args"), call.get("id")
            if not isinstance(name, str) or not isinstance(args, dict) or not isinstance(call_id, str):
                trace.append("Researcher: Received an invalid tool request and stopped safely.")
                invalid_tool_call = True
                break

            result = run_tool(name, args)
            if result.startswith("ERROR"):  ## self-correction: the error goes back to the LLM, which retries
                trace.append("Researcher: A data source returned an error; trying to recover.")
            else:
                data.append(result)
                activity = {
                    "get_stock_price": "Checked recent stock prices.",
                    "get_financial_news": "Searched recent financial news.",
                    "calculator": "Completed a requested calculation.",
                }.get(name, "Collected financial information.")
                trace.append(f"Researcher: {activity}")
            messages.append(ToolMessage(content=result, tool_call_id=call_id))
        if invalid_tool_call:
            break
    else:
        trace.append("Researcher: Reached the research limit.")

    all_data = state.get("data", []) + data
    result = {"data": all_data, "trace": trace}
    if not all_data:
        result["answer"] = (
            "I couldn't retrieve reliable information for that request. "
            "Please check the company or ticker and try asking for its price or recent news."
        )
    return result


## ---------------- Agent 3: WRITER (summary + reflection) ----------------
class Draft(BaseModel):
    answer: str = Field(description="Final answer in simple language with short bullet points")
    is_complete: bool = Field(description="True if the collected data fully answers the question")
    missing: str = Field(description="What data is missing. Empty string if complete")


def writer_agent(state):
    data = "\n\n".join(state.get("data", [])) or "No data was collected."
    try:
        llm = get_llm(0.2).with_structured_output(Draft)
        draft = llm.invoke([
            SystemMessage(content="You are a financial news writer. Using ONLY the data given, answer the question with a concise "
                                  "summary, key numbers, and key news points with source names. Never invent data. "
                                  "If something is missing or a tool failed, say so honestly. Do not include tool names, "
                                  "function calls, or emojis. End with: 'This is information, not financial advice.' "
                                  "Also check yourself: set is_complete=False if the data does not fully answer the question."),
            HumanMessage(content=f"Chat history:\n{format_history(state['history'])}\n\nQuestion: {state['query']}\n\nCollected data:\n{data}"),
        ])
    except Exception:
        logger.exception("Writer could not prepare a response")
        return {
            "answer": "I found information but couldn't prepare a reliable summary. Please try again.",
            "feedback": "",
            "retries": state.get("retries", 0),
            "trace": ["Writer: Could not prepare a reliable summary."],
        }

    ## reflection: if incomplete and we still have retries left, send the work back to the Researcher
    retry = (not draft.is_complete) and state.get("retries", 0) < MAX_RETRIES
    trace = ["Writer: Prepared and checked the answer."]
    if retry:
        trace.append("Writer: Some information is missing; asking the Researcher to check again.")
    return {"answer": draft.answer, "feedback": draft.missing if retry else "",
            "retries": state.get("retries", 0) + (1 if retry else 0), "trace": trace}
