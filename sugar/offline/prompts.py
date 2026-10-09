"""Fixed prompt templates for the frozen LLM (Sections 3.3.1 and 3.3.2).

Four operations map one-to-one to Section 3:
  RefinePrompt  : item text        -> structured descriptor d_v (M1)
  BuildPrompt   : descriptors      -> initial profile P_u        (M2 build)
  EditPrompt    : profile + batch  -> incrementally edited profile (M2 consolidate)
  RevisePrompt  : profile + window -> drift-reconciled profile   (M3 trigger)
"""

REFINE_PROMPT = """You are a content analyzer for a social media platform.
Given the item content below, output a JSON object with fields:
(1) topic: the dominant topic as a short phrase;
(2) subtopics: up to three secondary themes;
(3) style: the content type (news, opinion, tutorial, entertainment, promotion);
(4) audience: the intended audience;
(5) summary: a one-sentence neutral summary.

Content: {content}"""

BUILD_PROMPT = """You are a user profiling engine. Below are the topics of the
items a user has interacted with, grouped by topic with relative frequencies.

Output a JSON object with fields:
(1) dominant_interests: ranked list of objects {{"interest": str, "weight": float}},
    weights in [0,1] summing to at most 1.0;
(2) emerging_interests: list of short phrases;
(3) content_preferences: list of content styles/formats the user engages with;
(4) excluded_topics: topics the user demonstrably avoids.

Interaction topics:
{topics}"""

EDIT_PROMPT = """You are updating a user interest profile. Apply ONLY the
following new interactions to the profile: reinforce weights of recurring
topics, promote an emerging topic to dominant if its evidence accumulated,
demote interests no longer supported. Preserve all other fields verbatim.

Current profile:
{profile}

New interactions:
{delta}"""

REVISE_PROMPT = """You are reconciling a user profile with conflicting recent
behavior. The recent interaction window contradicts the stored profile.
Strengthen newly emerging interests evident in the recent window, demote
abandoned ones, and preserve any long-term interests not contradicted by the
recent evidence.

Current profile:
{profile}

Recent interactions (last {k} items):
{recent}"""

FALLBACK_INSTRUCTION = (
    "Output only the JSON object. No extra text, no markdown fences."
)


def refine_prompt(content: str) -> str:
    return REFINE_PROMPT.format(content=content) + "\n" + FALLBACK_INSTRUCTION


def build_prompt(topics_block: str) -> str:
    return BUILD_PROMPT.format(topics=topics_block) + "\n" + FALLBACK_INSTRUCTION


def edit_prompt(profile_json: str, delta_block: str) -> str:
    return EDIT_PROMPT.format(profile=profile_json, delta=delta_block) + "\n" + FALLBACK_INSTRUCTION


def revise_prompt(profile_json: str, recent_block: str, k: int) -> str:
    return REVISE_PROMPT.format(profile=profile_json, recent=recent_block, k=k) \
        + "\n" + FALLBACK_INSTRUCTION