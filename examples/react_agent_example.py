"""GoodMem + DSPy ReAct agent example.

Four scenarios that drive the GoodMem tools through ``dspy.ReAct``:

    Scenario 1 -- Conversational memory agent
        A ReAct agent handles a sequence of turns. It stores facts the user
        shares with ``goodmem_remember`` and recalls them with
        ``goodmem_search`` when asked follow-up questions. Each turn is an
        independent ReAct call -- memory lives in GoodMem, not in the agent.

    Scenario 2 -- Cross-agent memory persistence
        A brand-new ReAct agent, with no prior calls and no conversation
        history, answers questions about the user by searching GoodMem.
        Memory outlives the agent instance.

    Scenario 3 -- Metadata-scoped memories
        Writes memories tagged with a ``category`` directly through
        ``GoodMemClient``, then hands an agent a tool set whose search is
        scoped server-side to ``category == "hobby"``. The developer chooses
        the scope; the model only supplies the query.

    Scenario 4 -- Trajectory inspection
        Prints the thought/action/observation trajectory ReAct produced, to
        show the agent actually reached for GoodMem rather than answering
        from its weights.

Prerequisites:
    - A running GoodMem server (see https://docs.goodmem.ai)
    - An OpenAI API key (or any LiteLLM-supported provider)
    - At least one embedder registered on your GoodMem server

Usage::

    export OPENAI_API_KEY="sk-..."
    export GOODMEM_API_KEY="gm_..."
    export GOODMEM_BASE_URL="https://localhost:8080"
    python examples/react_agent_example.py

Optional:

    GOODMEM_VERIFY_SSL   "true" by default. Set to "false" only for a local
                         server with a self-signed certificate.
    GOODMEM_EMBEDDER_ID  Pin the embedder; otherwise the first one the server
                         lists is used.
"""

from __future__ import annotations

import json
import os
import sys
import time

import dspy

from dspy_goodmem import GoodMemClient, GoodMemError, make_goodmem_tools

try:  # python-dotenv is optional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

REQUIRED_ENV_VARS = [
    ("GOODMEM_API_KEY", "GoodMem API key (sent as X-API-Key)."),
    ("GOODMEM_BASE_URL", "Base URL of your GoodMem server, e.g. https://localhost:8080."),
    ("OPENAI_API_KEY", "OpenAI API key used by the default dspy.LM."),
]

SPACE_NAME = "dspy-goodmem-react-example"

SCENARIO_1_TURNS = [
    "I live in Austin, Texas.",
    "My favorite database is AcmeDB.",
    "What's my favorite database?",
    "And where do I live?",
]
SCENARIO_2_QUESTION = "Tell me everything you know about the user."
TAGGED_FACTS = [
    ("I work as a senior engineer at Acme Corp.", "work"),
    ("My manager is named Sarah.", "work"),
    ("I play guitar every Saturday morning.", "hobby"),
    ("I run 5 miles every Sunday.", "hobby"),
    ("I have a black cat named Luna.", "personal"),
]
SCENARIO_3_QUESTION = "What do you know about the user's hobbies?"


def check_env_vars() -> None:
    missing = [(n, d) for n, d in REQUIRED_ENV_VARS if not os.environ.get(n)]
    if missing:
        lines = ["Error: missing required environment variables:", ""]
        lines += [f"  - {n}: {d}" for n, d in missing]
        sys.exit("\n".join(lines))


class MemoryAssistant(dspy.Signature):
    """You are a personal assistant with a semantic memory store.

    When the user shares a fact about themselves, call goodmem_remember to
    store it. When the user asks a question about themselves, call
    goodmem_search before answering. Always use a GoodMem tool rather than
    relying on your own memory.
    """

    user_message: str = dspy.InputField(desc="What the user said or asked.")
    assistant_response: str = dspy.OutputField(desc="Your grounded reply to the user.")


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def setup_space(client: GoodMemClient, space_name: str) -> str:
    """Pick an embedder and create (or reuse) the demo space."""
    try:
        embedders = client.list_embedders()
    except GoodMemError as error:
        if error.status_code in (401, 403):
            sys.exit(f"Error: GoodMem rejected the request ({error.status_code}). Check GOODMEM_API_KEY.")
        sys.exit(f"Error: could not reach GoodMem at {client.base_url}: {error}")
    if not embedders:
        sys.exit("Error: no embedders on the GoodMem server. Register one first.")
    embedder_id = os.environ.get("GOODMEM_EMBEDDER_ID") or embedders[0]["embedderId"]
    space = client.create_space(space_name, embedder_id)
    print(f"  Space '{space_name}' ({'reused' if space['reused'] else 'created'}): {space['spaceId']}")
    return space["spaceId"]


def cleanup(client: GoodMemClient, space_ids: list[str]) -> None:
    """Delete every memory in each space, then the space, and verify."""
    for space_id in space_ids:
        for memory in client.list_memories(space_id):
            client.delete_memory(memory["memoryId"])
        client.delete_space(space_id)
    remaining = {s["spaceId"] for s in client.list_spaces()} & set(space_ids)
    print("  Cleanup complete." if not remaining else f"  WARNING: not deleted: {remaining}")


def wait_until_searchable(tools: list, probe: str, timeout: float = 60.0) -> None:
    """Poll on the *write* path until the seeded memory is retrievable."""
    search = tools[0]
    deadline = time.time() + timeout
    while time.time() < deadline:
        if search(probe, 3)["totalResults"]:
            return
        time.sleep(2)


def scenario_1_conversational_agent(tools: list) -> None:
    section("Scenario 1: Conversational memory agent (multi-turn)")
    agent = dspy.ReAct(MemoryAssistant, tools=[dspy.Tool(t) for t in tools], max_iters=6)
    for turn, user_message in enumerate(SCENARIO_1_TURNS, start=1):
        print(f"\n  Turn {turn}\n  User:  {user_message}")
        result = agent(user_message=user_message)
        print(f"  Agent: {result.assistant_response}")
        if turn == 2:
            wait_until_searchable(tools, "favorite database")


def scenario_2_cross_agent_memory(tools: list) -> dspy.Prediction:
    section("Scenario 2: Cross-agent memory persistence")
    print("  (A fresh ReAct agent with no prior calls.)")
    reader = dspy.ReAct(MemoryAssistant, tools=[dspy.Tool(t) for t in tools], max_iters=6)
    print(f"\n  User:  {SCENARIO_2_QUESTION}")
    result = reader(user_message=SCENARIO_2_QUESTION)
    print(f"  Agent: {result.assistant_response}")
    return result


def scenario_3_metadata_scope(client: GoodMemClient) -> tuple[str, dspy.Prediction]:
    section("Scenario 3: Metadata-scoped memories")
    tagged_space_id = setup_space(client, f"{SPACE_NAME}-tagged")
    print(f"\n  Ingesting {len(TAGGED_FACTS)} tagged memories...")
    for content, category in TAGGED_FACTS:
        client.create_memory(tagged_space_id, text_content=content, metadata={"category": category})
        print(f"    [{category:>8}] {content}")

    # The developer scopes the search server-side; the model never sees or
    # composes the filter. Escaping and type casts are handled by the builder.
    hobby_tools = make_goodmem_tools(client, [tagged_space_id], metadata_filter={"category": "hobby"})
    wait_until_searchable(hobby_tools, "guitar")
    agent = dspy.ReAct(MemoryAssistant, tools=[dspy.Tool(t) for t in hobby_tools], max_iters=6)
    print(f"\n  User:  {SCENARIO_3_QUESTION}")
    result = agent(user_message=SCENARIO_3_QUESTION)
    print(f"  Agent: {result.assistant_response}")
    return tagged_space_id, result


def scenario_4_inspect_trajectory(result: dspy.Prediction) -> None:
    section("Scenario 4: Trajectory inspection")
    trajectory = getattr(result, "trajectory", None) or {}
    steps = sum(1 for k in trajectory if k.startswith("thought_"))
    print(f"\n  ReAct steps: {steps}")
    for i in range(steps):
        observation = str(trajectory.get(f"observation_{i}", ""))
        print(f"\n    Step {i + 1}")
        print(f"      Thought: {trajectory.get(f'thought_{i}', '')}")
        print(f"      Tool:    {trajectory.get(f'tool_name_{i}', '')}({json.dumps(trajectory.get(f'tool_args_{i}', {}))})")
        print(f"      Result:  {observation[:120]}{'...' if len(observation) > 120 else ''}")


def main() -> None:
    check_env_vars()
    verify_ssl = os.environ.get("GOODMEM_VERIFY_SSL", "true").lower() != "false"
    dspy.configure(lm=dspy.LM("openai/gpt-5-mini"))

    client = GoodMemClient(verify_ssl=verify_ssl)
    print(f"  GoodMem: {client.base_url}")

    space_id = setup_space(client, SPACE_NAME)
    tools = make_goodmem_tools(client, [space_id])
    print(f"  Tools the model sees: {[t.__name__ for t in tools]}")

    tagged_space_id: str | None = None
    try:
        scenario_1_conversational_agent(tools)
        scenario_2_cross_agent_memory(tools)
        tagged_space_id, analyst_result = scenario_3_metadata_scope(client)
        scenario_4_inspect_trajectory(analyst_result)
    finally:
        section("Cleanup")
        cleanup(client, [s for s in (space_id, tagged_space_id) if s])
        client.close()


if __name__ == "__main__":
    main()
