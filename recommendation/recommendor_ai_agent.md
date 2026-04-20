# Building AI Agents with PydanticAI

This document explains what PydanticAI is, why it was chosen for this project, how its agent model works conceptually, and how each building block maps to code — using the recommender system itself as the working example throughout.

---

## What is PydanticAI?

PydanticAI is a Python framework for building AI agents. An **agent** is a program that uses a Large Language Model (LLM) as its reasoning engine to decide what actions to take, executes those actions by calling Python functions (called **tools**), and returns a final structured response.

It was built by the Pydantic team — the same team behind Pydantic v2, which is the de facto standard for data validation in Python. This lineage means PydanticAI is deeply integrated with Python's type system: inputs and outputs are validated automatically, and the code reads like ordinary Python rather than a framework-specific DSL.

### Why PydanticAI for this project?

| Need | How PydanticAI satisfies it |
|---|---|
| Route user query to CB or CF based on context | Tools dispatch Python functions; the LLM decides which one to call |
| Pass the loaded TF-IDF matrix, ALS model, and DataFrames into tools | Dependency injection (`deps`) — no global state needed |
| Return a structured list of recommendations | `output_type=` enforces a typed Pydantic model as the final response |
| Switch from OpenAI to Claude or a local model later | Model-agnostic — change one string, not the architecture |
| Avoid writing a manual tool-calling loop | The agent loop is fully managed internally |

---

## Core Concepts

Before writing any code, it helps to understand the four building blocks PydanticAI uses and how they relate to each other.

### 1. The Agent

The agent is the central object. It holds:
- Which LLM to use
- The static instructions (system prompt)
- What type the deps object will be
- What type the output should be
- All registered tools

Think of it as the "brain + configuration" of the system. In this project, the agent is the recommender router: it receives a natural language query and decides whether to call content-based or collaborative filtering.

### 2. Tools

Tools are plain Python functions the LLM is allowed to call. The LLM reads their name, docstring, and parameter types to understand what each tool does and what arguments to pass. When the LLM decides to call a tool, PydanticAI executes the function and feeds the result back to the LLM so it can continue reasoning.

In this project, the two tools are `cb_recommend` and `cf_recommend` — the actual filtering functions.

### 3. Dependencies (Deps)

Real-world tools need access to heavy objects that should not be reloaded on every request — in this project, the TF-IDF matrix (89k × 10k), the trained ALS model, and the product metadata DataFrame. Rather than using global variables, PydanticAI lets you bundle these into a `deps` object that is created once at startup and passed in at call time. Tools access deps through a `RunContext` parameter.

### 4. Structured Output

By default, the agent returns a plain string. When you set `output_type=` to a Pydantic model, PydanticAI forces the LLM to respond in that schema and validates the result. If the LLM returns something invalid, it is automatically retried. In this project, the output is a validated list of recommended products with metadata.

---

## The Agent Loop — How It Works at Runtime

Understanding the internal loop is key to understanding why agents behave the way they do.

```
User: "recommend me apps similar to Notion, I'm a new user"
      │
      ▼
 LLM receives: system prompt + user message + tool schemas for cb_recommend & cf_recommend
      │
      ├── LLM reasons: new user, no user_id, query-based → call cb_recommend
      │         │
      │         ▼
      │    LLM calls cb_recommend(query="Notion-like productivity apps", n=10)
      │         │
      │         ▼
      │    PydanticAI executes the Python function (TF-IDF cosine similarity)
      │         │
      │         ▼
      │    Tool returns: list of 10 matching products with scores
      │         │
      │         └──► LLM receives results, formats final structured response
      │
      └── LLM is satisfied — produces final answer
                │
                ▼
         PydanticAI validates against RecommendationResponse
                │
                ▼
         result.output  ← typed, validated Pydantic model
```

For a known power user the flow branches differently:

```
User: "what should I try next?" (user_id="ae2222frpdmnomyomcwiantxp7uq", 15 reviews)
      │
      ▼
 LLM reasons: known user_id, 15 reviews → sufficient history → call cf_recommend
      │
      ▼
 cf_recommend(user_id="ae2222frpdmnomyomcwiantxp7uq", n=10)
      │
      ▼
 ALS model returns top-N items → LLM formats response
```

The loop continues until the LLM stops requesting tool calls and produces a final response. You do not write this loop — PydanticAI manages it entirely.

---

## Step-by-Step: Building the Recommender Agent

---

### Step 1 — Define Your Dependencies

Dependencies are the runtime objects your tools will need. Define them as a dataclass so PydanticAI knows their shape at type-check time.

These objects are expensive to create — the TF-IDF matrix is 89k × 10k, the ALS model takes minutes to train — so they are loaded once at startup and injected into every agent call.

```python
from dataclasses import dataclass
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares


@dataclass
class RecDeps:
    tfidf_matrix: csr_matrix              # shape: (89246 items × 10000 words)
    tfidf_index: pd.Index                 # maps row index → parent_asin
    als_model: AlternatingLeastSquares    # trained ALS model
    user_to_idx: dict                     # user_id → integer index in sparse matrix
    item_to_idx: dict                     # parent_asin → integer index
    idx_to_item: dict                     # integer index → parent_asin
    meta_df: pd.DataFrame                 # product catalog (title, category, price, is_free)
    user_item_matrix: csr_matrix          # shape: (2.6M users × 89k items)
    user_review_counts: dict              # user_id → number of reviews they have
```

This object is created once when the application starts. Nothing is global, nothing is hardcoded inside the tools.

---

### Step 2 — Define Your Output Schema

Decide what shape the final response should have and model it as a Pydantic class. The agent will always return a validated instance of this.

```python
from pydantic import BaseModel, Field


class RecommendedItem(BaseModel):
    asin: str = Field(description="Product ASIN (parent_asin)")
    title: str = Field(description="Product title from metadata")
    category: str = Field(description="Product category")
    score: float = Field(description="Relevance score — higher is better", ge=0.0, le=1.0)
    is_free: bool = Field(description="Whether the product is free")


class RecommendationResponse(BaseModel):
    method_used: str = Field(description="'content_based' or 'collaborative_filtering'")
    reason: str = Field(description="Why this method was chosen for this user and query")
    recommendations: list[RecommendedItem] = Field(description="Ordered list of recommended products")
```

`Field(description=...)` is important — these descriptions are passed to the LLM inside the output schema so it knows what each field means and how to populate it correctly.

---

### Step 3 — Create the Agent

Instantiate the agent with the model, deps type, output type, and static instructions. The instructions are where you encode the routing logic — when the LLM should call CB vs CF.

```python
from pydantic_ai import Agent

rec_agent = Agent(
    "openai:gpt-4o",                 # model — swap to "anthropic:claude-opus-4-6" anytime
    deps_type=RecDeps,
    output_type=RecommendationResponse,
    instructions=(
        "You are a software recommendation assistant for the Amazon Software catalogue. "
        "You have two tools: cb_recommend (content-based filtering) and cf_recommend (collaborative filtering). "
        "\n\n"
        "Use cf_recommend when: the user has a known user_id AND has more than 10 reviews in their history. "
        "CF leverages patterns from similar users and discovers cross-category items. "
        "\n\n"
        "Use cb_recommend when: the user is new, has fewer than 10 reviews, provides an item name or "
        "description to search by, or no user_id is available. "
        "CB uses TF-IDF similarity over product text to find matching items. "
        "\n\n"
        "Always explain your choice of method in the 'reason' field of your response. "
        "Never recommend items the user has already interacted with."
    ),
)
```

The `instructions` string is what the LLM reads first in every conversation. It defines the agent's role, the routing rules (CB vs CF), and the constraints.

---

### Step 4 — Add Dynamic System Prompts

Some context is only known at runtime — specifically, how many reviews the current user has. This information should be injected into the system prompt so the LLM can make an informed routing decision without needing to call a separate tool to look it up.

```python
from pydantic_ai import RunContext


@rec_agent.system_prompt
def inject_user_profile(ctx: RunContext[RecDeps]) -> str:
    """Appended at runtime with the current user's review history count."""
    user_id = ctx.deps.current_user_id   # see: deps carries the request's user_id too
    review_count = ctx.deps.user_review_counts.get(user_id, 0)

    if review_count == 0:
        profile = "This is a new user with no review history."
    elif review_count < 10:
        profile = f"This user has {review_count} reviews — occasional user, prefer content-based."
    else:
        profile = f"This user has {review_count} reviews — power user, collaborative filtering is reliable."

    return f"Current user_id: {user_id}. {profile}"


@rec_agent.system_prompt
def inject_catalogue_size() -> str:
    """Reminds the LLM of the catalogue scope — no ctx needed."""
    return "The product catalogue contains 89,246 software items across categories like productivity, games, utilities, and media."
```

When the agent runs, the LLM sees all of these appended together after the static instructions. Multiple `@agent.system_prompt` decorators are allowed and appended in definition order.

---

### Step 5 — Define Tools

Each tool is a Python function decorated with `@agent.tool`. The LLM reads the function name, docstring, and parameter annotations to understand what the tool does and constructs the call arguments automatically.

**Rules:**
- The first parameter must always be `ctx: RunContext[YourDepsType]`
- The remaining parameters are what the LLM fills in from the user query
- The docstring is the tool's description — write it clearly, as the LLM reads it to decide when to call this tool
- Return anything serialisable — string, dict, list, Pydantic model

```python
from pydantic_ai import RunContext
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


@rec_agent.tool
def cb_recommend(ctx: RunContext[RecDeps], query: str, n: int = 10) -> list[dict]:
    """
    Content-based recommendation using TF-IDF cosine similarity.
    Use this when: the user is new, has few reviews, or provides a description/item name to search by.
    The query should describe what the user is looking for (e.g. 'productivity timer apps', 'Notion-like tools').
    Returns the top-n most similar products from the catalogue.
    """
    deps = ctx.deps

    # Vectorise the query using the same TF-IDF vocabulary as the item matrix
    query_vec = deps.tfidf_vectorizer.transform([query])

    # Cosine similarity between query vector and all 89k item vectors
    scores = cosine_similarity(query_vec, deps.tfidf_matrix).flatten()

    # Get top n*5 candidates for re-ranking
    top_indices = np.argsort(scores)[::-1][: n * 5]

    results = []
    for idx in top_indices:
        asin = deps.tfidf_index[idx]
        row = deps.meta_df[deps.meta_df["parent_asin"] == asin].iloc[0]
        results.append({
            "asin": asin,
            "title": row["title"],
            "category": row.get("main_category", "unknown"),
            "score": round(float(scores[idx]), 4),
            "is_free": bool(row.get("is_free", False)),
        })
        if len(results) == n:
            break

    return results


@rec_agent.tool
def cf_recommend(ctx: RunContext[RecDeps], user_id: str, n: int = 10) -> list[dict]:
    """
    Collaborative filtering recommendation using a trained ALS model.
    Use this when: the user has a known user_id with more than 10 reviews in their history.
    ALS finds latent taste patterns shared across similar users.
    Returns the top-n items the user has not yet interacted with.
    """
    deps = ctx.deps

    if user_id not in deps.user_to_idx:
        return []   # unknown user — caller should fall back to cb_recommend

    user_idx = deps.user_to_idx[user_id]
    user_row = deps.user_item_matrix[user_idx]

    # ALS model returns (item_indices, scores) already excluding seen items
    item_indices, scores = deps.als_model.recommend(
        user_idx, user_row, N=n, filter_already_liked_items=True
    )

    results = []
    for item_idx, score in zip(item_indices, scores):
        asin = deps.idx_to_item[item_idx]
        row = deps.meta_df[deps.meta_df["parent_asin"] == asin].iloc[0]
        results.append({
            "asin": asin,
            "title": row["title"],
            "category": row.get("main_category", "unknown"),
            "score": round(float(score), 4),
            "is_free": bool(row.get("is_free", False)),
        })

    return results
```

**When a tool does not need deps at all**, use `@agent.tool_plain` and omit `ctx` entirely. For example, a helper that returns the list of valid categories:

```python
@rec_agent.tool_plain
def list_categories() -> list[str]:
    """Return the top-level software categories available in the catalogue."""
    return [
        "Productivity", "Games", "Utilities", "Education",
        "Media & Entertainment", "Security", "Development Tools",
    ]
```

---

### Step 6 — Run the Agent

Pass the user's message and the deps object. The agent handles the full reasoning loop and returns when it has a validated `RecommendationResponse`.

```python
import asyncio


async def get_recommendations(user_id: str, query: str, deps: RecDeps) -> RecommendationResponse:
    result = await rec_agent.run(query, deps=deps)
    return result.output   # validated RecommendationResponse instance


# Example calls
async def main():
    # Load heavy objects once at startup
    deps = RecDeps(
        tfidf_matrix=...,
        tfidf_index=...,
        als_model=...,
        user_to_idx=...,
        item_to_idx=...,
        idx_to_item=...,
        meta_df=...,
        user_item_matrix=...,
        user_review_counts=...,
        current_user_id="ae2222frpdmnomyomcwiantxp7uq",
    )

    # Case 1 — New user, query-based → agent will call cb_recommend
    result = await rec_agent.run(
        "Recommend me apps similar to Notion for note-taking and organisation.",
        deps=deps,
    )
    print(result.output.method_used)   # "content_based"
    print(result.output.reason)        # "User is new with no review history..."
    for item in result.output.recommendations:
        print(item.title, item.score)

    # Case 2 — Power user → agent will call cf_recommend
    deps.current_user_id = "ae2222frpdmnomyomcwiantxp7uq"   # 15 reviews
    result = await rec_agent.run(
        "What should I try next based on my history?",
        deps=deps,
    )
    print(result.output.method_used)   # "collaborative_filtering"


asyncio.run(main())
```

What happens internally during the Case 1 `.run()` call:

```
1. LLM reads instructions + dynamic prompts ("new user, no review history") + 3 tool schemas
2. LLM reasons: new user + item name given → call cb_recommend
3. LLM calls: cb_recommend(query="Notion note-taking organisation apps", n=10)
4. PydanticAI executes cb_recommend → TF-IDF cosine similarity runs → returns list of 10 dicts
5. LLM receives the list, formats it into a RecommendationResponse
6. PydanticAI validates the response against the schema
7. result.output returned — a fully typed RecommendationResponse instance
```

For synchronous code (scripts, notebooks), use `run_sync` instead:

```python
result = rec_agent.run_sync(
    "Recommend free productivity apps.",
    deps=deps,
)
print(result.output.recommendations[0].title)
```

---

### Step 7 — Multi-turn Conversations (Optional)

If you want the agent to remember what was said earlier in the same session — for example, a user refining their request — pass `message_history` from the previous result:

```python
# Turn 1 — initial recommendation
result1 = rec_agent.run_sync(
    "Recommend me some productivity apps.",
    deps=deps,
)

# Turn 2 — user refines without repeating themselves
result2 = rec_agent.run_sync(
    "Actually, only show me free ones.",
    deps=deps,
    message_history=result1.new_messages(),   # agent remembers Turn 1
)
```

`result.new_messages()` returns only the messages from that run. `result.all_messages()` returns the full history including any prior turns passed in. This is useful for building a chat interface on top of the agent.

---

## How the Pieces Fit Together — Full Skeleton

```python
# ── deps.py ───────────────────────────────────────────────────────────────────
from dataclasses import dataclass
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

@dataclass
class RecDeps:
    tfidf_matrix: csr_matrix
    tfidf_index: pd.Index
    tfidf_vectorizer: object
    als_model: AlternatingLeastSquares
    user_to_idx: dict
    idx_to_item: dict
    meta_df: pd.DataFrame
    user_item_matrix: csr_matrix
    user_review_counts: dict
    current_user_id: str             # set per-request


# ── schemas.py ────────────────────────────────────────────────────────────────
from pydantic import BaseModel, Field

class RecommendedItem(BaseModel):
    asin: str
    title: str
    category: str
    score: float = Field(ge=0.0, le=1.0)
    is_free: bool

class RecommendationResponse(BaseModel):
    method_used: str = Field(description="'content_based' or 'collaborative_filtering'")
    reason: str = Field(description="Why this method was chosen")
    recommendations: list[RecommendedItem]


# ── agent.py ──────────────────────────────────────────────────────────────────
from pydantic_ai import Agent, RunContext

rec_agent = Agent(
    "openai:gpt-4o",
    deps_type=RecDeps,
    output_type=RecommendationResponse,
    instructions=(
        "You are a software recommendation assistant. "
        "Use cf_recommend for users with >10 reviews. "
        "Use cb_recommend for new/occasional users or query-based requests. "
        "Always explain your method choice in the 'reason' field."
    ),
)

@rec_agent.system_prompt
def inject_user_profile(ctx: RunContext[RecDeps]) -> str:
    count = ctx.deps.user_review_counts.get(ctx.deps.current_user_id, 0)
    return f"Current user: {ctx.deps.current_user_id}. Review count: {count}."

@rec_agent.tool
def cb_recommend(ctx: RunContext[RecDeps], query: str, n: int = 10) -> list[dict]:
    """Content-based recommendation via TF-IDF cosine similarity. Use for new/cold-start users or item-query searches."""
    # ... TF-IDF logic here ...
    return []

@rec_agent.tool
def cf_recommend(ctx: RunContext[RecDeps], user_id: str, n: int = 10) -> list[dict]:
    """Collaborative filtering via ALS. Use for users with >10 reviews and a known user_id."""
    # ... ALS logic here ...
    return []


# ── main.py ───────────────────────────────────────────────────────────────────
import asyncio

async def main():
    deps = RecDeps(...)   # load models, matrices, dataframes
    result = await rec_agent.run(
        "Recommend me Notion-like apps, I just signed up.",
        deps=deps,
    )
    print(result.output.model_dump_json(indent=2))

asyncio.run(main())
```

---

## Quick API Reference

| Concept | API |
|---|---|
| Create agent | `Agent("openai:gpt-4o", deps_type=..., output_type=..., instructions=...)` |
| Static system prompt | `instructions="..."` on `Agent(...)` |
| Dynamic system prompt | `@agent.system_prompt` decorator |
| Tool with deps | `@agent.tool` — first param `ctx: RunContext[Deps]` |
| Tool without deps | `@agent.tool_plain` — no ctx param |
| Run async | `result = await agent.run("query", deps=deps)` |
| Run sync | `result = agent.run_sync("query", deps=deps)` |
| Get typed output | `result.output` — validated Pydantic model |
| Continue conversation | `message_history=result.new_messages()` |
| Full history | `result.all_messages()` |
| Switch model | Change first arg: `"anthropic:claude-opus-4-6"`, `"gemini-1.5-pro"` |
