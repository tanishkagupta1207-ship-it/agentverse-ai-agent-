import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

try:
    from agent.claims import AnswerResult
    from agent.indexer import search_code
    from agent.verifier import verify_existence, verify_semantic
    from agent.code_tools import get_callers, get_symbol
except ImportError:
    from claims import AnswerResult
    from indexer import search_code
    from verifier import verify_existence, verify_semantic
    from code_tools import get_callers, get_symbol

load_dotenv()
MAX_STEPS = 14
MAX_TOOL_STEPS = 6   # tool rounds before final_answer is forced
MAX_TOOL_CHARS = 3500

SYSTEM_PROMPT = """You are Oracle Onboard, an assistant that answers questions about a Python repository.
Use the tools to find evidence. Do not answer from memory.
Use ONLY the tools provided (search_code, get_symbol, get_callers, git_history, get_issue_or_pr, list_issues, final_answer). Never call any other tool, such as open_file or browsing tools.
Make at most 5 tool calls in total before answering.
Do not repeat a search for something you already found. Once get_symbol or search_code returns the code you need, move on or answer.

When you are done, call final_answer exactly once. Rules:
- Each claim is ONE fact. Split compound statements into separate claims.
- label "verified": backed by code evidence: type "code", path, and start/end = a search_code/get_symbol result's start_line/end_line, or a get_callers entry's path and line (use line for both start and end).
- label "history_backed": backed by a commit, issue or pr copied from git_history / get_issue_or_pr results.
- label "assumption": your own inference. It must have an empty evidence list.
- Only cite evidence that appeared in tool results. Never invent paths, line numbers, SHAs or numbers.
- Keep it short: at most 6 claims, each under 25 words, valid JSON only.
- If the tools find nothing relevant, call final_answer with not_found=true and no claims.
- For "why" questions, call git_history on the relevant lines. If it finds no explanatory history, say so
  with an assumption claim instead of guessing a reason."""

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "search_code",
        "description": "Search the indexed repository for code relevant to a query. Returns chunks with path and line ranges.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "k": {"type": "integer", "description": "number of results, default 5"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_symbol",
        "description": "Find where a function, method or class is defined, e.g. 'Session.send' or 'resolve_redirects'.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "get_callers",
        "description": "Find call sites of a function or method, with the function each call sits in.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "git_history",
        "description": "Commits (and linked issues/PRs) that touched a line range of a file. Use for 'why' questions.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"}},
            "required": ["path", "start_line", "end_line"]}}},
    {"type": "function", "function": {
        "name": "get_issue_or_pr",
        "description": "Details of one issue or pull request by number.",
        "parameters": {"type": "object", "properties": {
            "number": {"type": "integer"}}, "required": ["number"]}}},
    {"type": "function", "function": {
        "name": "list_issues",
        "description": "List issues. state: open/closed/all. Optional label filter.",
        "parameters": {"type": "object", "properties": {
            "state": {"type": "string"},
            "labels": {"type": "array", "items": {"type": "string"}},
            "include_prs": {"type": "boolean"},
            "limit": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "final_answer",
        "description": "Submit the final structured answer. Call once, when done.",
        "parameters": {"type": "object", "properties": {
            "claims": {"type": "array", "items": {"type": "object", "properties": {
                "id": {"type": "string"},
                "text": {"type": "string"},
                "label": {"type": "string", "enum": ["verified", "history_backed", "assumption"]},
                "evidence": {"type": "array", "items": {"type": "object", "properties": {
                    "type": {"type": "string", "enum": ["code", "commit", "issue", "pr"]},
                    "path": {"type": "string"},
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                    "sha": {"type": "string"},
                    "number": {"type": "integer"}},
                    "required": ["type"]}}},
                "required": ["id", "text", "label", "evidence"]}},
            "not_found": {"type": "boolean"},
            "note": {"type": "string"}},
            "required": ["claims"]}}},
]


def make_client():
    from openai import OpenAI
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        sys.exit("No API key. Put LLM_API_KEY=... in your .env file.")
    return OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL") or None)


def default_tools():
    # B's tools live in data_layer/ and import each other by bare name
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_layer"))
    from why_trace import git_history
    from issue_tools import get_issue_or_pr, list_issues
    return {"search_code": search_code, "get_symbol": get_symbol,
            "get_callers": get_callers, "git_history": git_history,
            "get_issue_or_pr": get_issue_or_pr, "list_issues": list_issues}


def _clean(obj):
    """Drop None values so the pydantic evidence union validates."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def _compact(name, r):
    """Shrink tool output so each request stays small (free tiers have token limits)."""
    if not isinstance(r, dict):
        return r
    if name == "get_callers":
        return {"found": r.get("found"), "note": r.get("note"),
                "callers": r.get("callers", [])[:12]}
    if name in ("search_code", "get_symbol"):
        return {"found": r.get("found"), "note": r.get("note"),
                "results": [{"path": x["path"], "symbol": x["symbol"],
                             "start_line": x["start_line"], "end_line": x["end_line"],
                             "text": x["text"][:500]} for x in r.get("results", [])]}
    if name == "git_history":
        return {"found": r.get("found"), "note": r.get("note"),
                "timeline": r.get("timeline", [])[:10],
                "evidence": r.get("evidence", [])[:20]}
    if name == "get_issue_or_pr" and r.get("body"):
        return {**r, "body": r["body"][:800]}
    if name == "list_issues":
        return {**r, "issues": r.get("issues", [])[:10]}
    return r


def _run_tool(tools, name, args):
    try:
        if name == "search_code":
            args = {**args, "k": min(int(args.get("k", 5)), 5)}
        out = json.dumps(_compact(name, tools[name](**args)), default=str)
    except Exception as e:  # tool errors go back to the model, not a crash
        out = json.dumps({"error": f"{type(e).__name__}: {e}"})
    return out[:MAX_TOOL_CHARS]


def run_agent(question, client=None, tools=None, model=None, verify_kwargs=None,
              verbose=False, semantic=None, semantic_kwargs=None):
    """Returns (AnswerResult after verification, downgrade log, tool_trace)."""
    client = client or make_client()
    tools = tools or default_tools()
    model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    trace = []
    waits = 0
    bad_calls = 0
    tool_steps = 0
    nudged = False
    force_ok=True

    for step in range(1, MAX_STEPS + 1):
        over_budget = tool_steps >= MAX_TOOL_STEPS
        if over_budget and not nudged:
            nudged = True
            messages.append({"role": "user", "content": (
                "You have enough evidence. Call final_answer now with what you found. "
                "If the evidence only partly answers the question, give the claims you can "
                "support and mark the rest as assumption. Use not_found only if nothing "
                "relevant was found, and then explain why in note.")})
        # ALWAYS send every tool spec: the API rejects calls to tools that are
        # not in the request. The budget is enforced below, not by hiding tools.
        choice = ({"type": "function", "function": {"name": "final_answer"}}
            if over_budget and force_ok else "auto")
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, tools=TOOL_SPECS, tool_choice=choice)
        except Exception as e:
            err = str(e).lower()
            if "tool choice is required" in err or "did not call a tool" in err:
                if verbose:
                    print(f"step {step}: forced final_answer failed, retrying without forcing")
                force_ok = False
                messages.append({"role": "user", "content": (
                    "Call the final_answer tool now. Keep it short: at most 4 claims, "
                    "each under 20 words, valid JSON only.")})
                continue
            malformed = any(k in err for k in (
                "tool call validation failed", "parsing failed",
                "failed to parse", "could not be parsed"))
            if malformed:
                if bad_calls < 4:
                    bad_calls += 1
                    if verbose:
                        print(f"step {step}: malformed tool call, asking the model to retry")
                    messages.append({"role": "user", "content": (
                        "Your last output was invalid. Call one tool with valid JSON arguments. "
                        "Use only: search_code, get_symbol, get_callers, git_history, "
                        "get_issue_or_pr, list_issues, final_answer.")})
                    continue
                return (AnswerResult(not_found=True,
                                     note="The model kept producing invalid tool calls."),
                        [], trace)
            if any(k in err for k in ("rate limit", "tokens per minute", "429", "413")):
                if waits < 3:
                    waits += 1
                    if verbose:
                        print(f"step {step}: rate limit hit, waiting 30s ({waits}/3)")
                    time.sleep(30)
                    continue
            raise
        msg = resp.choices[0].message

        if not msg.tool_calls:  # model replied with text: push it to submit
            if verbose:
                print(f"step {step}: text reply, asking for final_answer")
            messages.append({"role": "assistant", "content": msg.content or ""})
            messages.append({"role": "user",
                             "content": "Submit your answer by calling final_answer."})
            continue

        messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls]})

        if verbose:
            print(f"step {step}: {[tc.function.name for tc in msg.tool_calls]}")
        final = None
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if name == "final_answer":
                try:
                    final = AnswerResult.model_validate(_clean(args))
                    result = "ok"
                except ValidationError as e:
                    final = None
                    result = f"Invalid answer, fix and call final_answer again: {e}"
            elif name in tools:
                if over_budget:
                    result = json.dumps({"error": "Tool budget used up. Call final_answer now."})
                else:
                    trace.append({"tool": name, "args": args})
                    result = _run_tool(tools, name, args)
                    if verbose:
                        print(f"    {name}({args}) -> {result[:200]}")
            else:
                result = json.dumps({"error": f"unknown tool {name}"})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        tool_steps += 1
        if final is not None:
            checked, log = verify_existence(final, **(verify_kwargs or {}))
            if semantic is None:
                semantic = os.getenv("SKIP_SEMANTIC") != "1"
            if semantic and checked.claims:
                if verbose:
                    print("verifying claims against their evidence...")
                checked, log2 = verify_semantic(checked, **(semantic_kwargs or {}))
                log = log + log2
            return checked, log, trace

    return (AnswerResult(not_found=True, note="Agent did not finish within the step limit."),
            [], trace)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How does requests handle redirects?"
    answer, log, trace = run_agent(q, verbose=True)
    print("Tools used:", [t["tool"] for t in trace])
    if answer.not_found:
        print("NOT FOUND:", answer.note or "(model gave no reason)")
    for c in answer.claims:
        print(f"\n[{c.label}] {c.text}")
        for e in c.evidence:
            print("   ", e.model_dump())
        if c.note:
            print("    note:", c.note)
    if log:
        print("\nVerifier log:")
        for entry in log:
            print("  ", entry)