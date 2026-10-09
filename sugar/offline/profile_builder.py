"""M2: User Profile Builder / Editor (Section 3.3.2).

Two update paths with distinct triggers (Table 3-3):
  consolidation edit : scheduled, fires when |delta| >= m_edit
  drift revision     : conflict-driven, fired ONLY by the drift detector (M3)

Consolidation contract: every interaction enters the profile exactly once.
Interactions consumed by a revision are added to the consolidated set too,
so they are never counted again when they later leave the recent window.
"""
import json


def topics_block(seq_descriptors: list, freq_cap: int = 30) -> str:
    """Serialize descriptors into a compact 'topic xN; subtopics' block for prompts."""
    counts = {}
    for d in seq_descriptors:
        t = d.get("topic", "unknown")
        counts.setdefault(t, {"n": 0, "subs": set()})
        counts[t]["n"] += 1
        for s in d.get("subtopics", [])[:2]:
            counts[t]["subs"].add(s)
    lines = []
    for t, c in sorted(counts.items(), key=lambda kv: -kv[1]["n"])[:freq_cap]:
        subs = ", ".join(sorted(c["subs"])[:3])
        lines.append(f"- {t} (count={c['n']})" + (f"; subtopics: {subs}" if subs else ""))
    return "\n".join(lines)


def recent_block(recent_descriptors: list) -> str:
    return topics_block(recent_descriptors, freq_cap=15)


class ProfileBuilder:
    def __init__(self, llm, encoder, min_profile_n0: int, edit_batch_medit: int,
                 window_size_k: int):
        self.llm = llm
        self.encoder = encoder
        self.n0 = min_profile_n0
        self.m_edit = edit_batch_medit
        self.k = window_size_k
        self.stats = {"builds": 0, "consolidations": 0, "revisions": 0,
                      "edit_failures": 0}

    # -- initial construction -------------------------------------------------
    def build(self, seq_descriptors: list) -> dict | None:
        """Initial profile P_u^(0) from all descriptors observed so far."""
        prompt = build_prompt(topics_block(seq_descriptors))
        profile = self.llm.generate_json(prompt)
        self.stats["builds"] += 1
        if profile is None:
            # deterministic fallback: profile from raw topic counts
            profile = self._fallback_profile(seq_descriptors)
        return profile

    # -- incremental consolidation --------------------------------------------
    def consolidate(self, profile: dict, delta_descriptors: list) -> dict | None:
        """Scheduled edit; on LLM failure the profile is returned unchanged."""
        if not delta_descriptors:
            return profile
        prompt = edit_prompt(json.dumps(profile, ensure_ascii=False),
                             recent_block(delta_descriptors))
        new = self.llm.generate_json(prompt)
        self.stats["consolidations"] += 1
        if new is None:
            self.stats["edit_failures"] += 1
            return profile
        return new

    # -- drift revision --------------------------------------------------------
    def revise(self, profile: dict, recent_descriptors: list) -> dict | None:
        """Conflict-driven rewrite triggered by M3 (Section 3.4)."""
        prompt = revise_prompt(json.dumps(profile, ensure_ascii=False),
                               recent_block(recent_descriptors), self.k)
        new = self.llm.generate_json(prompt)
        self.stats["revisions"] += 1
        if new is None:
            self.stats["edit_failures"] += 1
            return profile
        return new

    # -- helpers ----------------------------------------------------------------
    @staticmethod
    def _fallback_profile(seq_descriptors: list) -> dict:
        counts = {}
        total = 0
        for d in seq_descriptors:
            t = d.get("topic", "unknown")
            counts[t] = counts.get(t, 0) + 1
            total += 1
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
        return {
            "dominant_interests": [{"interest": t, "weight": round(n / total, 3)}
                                   for t, n in ranked],
            "emerging_interests": [],
            "content_preferences": [],
            "excluded_topics": [],
            "_fallback": True,
        }

    @staticmethod
    def extract_interests(profile: dict) -> tuple[list, list]:
        """(phi_i strings, weights) from dominant_interests for profile embedding."""
        interests, weights = [], []
        for entry in profile.get("dominant_interests", []):
            if isinstance(entry, dict) and entry.get("interest"):
                interests.append(str(entry["interest"]))
                weights.append(float(entry.get("weight", 0.1)))
            elif isinstance(entry, str):
                interests.append(entry)
                weights.append(0.1)
        return interests, weights

    def compute_delta(self, seq: list, consolidated: set,
                      window: list) -> list:
        """Delta = S[1..t_j] \\ (R_u union C_u)  (Section 3.3.2).

        seq: full item-index list; consolidated: set of consolidated indices;
        window: current recent-window indices R_u.
        Returns item indices not yet consolidated and outside the window.
        """
        return [v for v in seq if v not in consolidated and v not in set(window)]