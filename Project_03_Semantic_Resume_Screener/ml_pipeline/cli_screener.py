"""Standalone CLI Screener for Academic Viva Demonstration.

Screens and ranks candidates from resume files (.pdf, .txt) against a Job
Specification using local ONNX inference, PII scrubbing, and multi-factor
composite scoring.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import numpy as np
from pypdf import PdfReader

# Ensure repo root is on sys.path regardless of execution working directory
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_pipeline.export_onnx import (
    DEFAULT_OUTPUT_PATH,
    embed_texts,
    export_onnx,
    load_onnx_session,
)

logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("export_onnx").setLevel(logging.WARNING)
logger = logging.getLogger("cli_screener")

# Predefined dictionary of common tech skills and skills from synthetic JDs
COMMON_TECH_KEYWORDS = [
    "Python", "JavaScript", "TypeScript", "React", "Next.js", "Node.js",
    "HTML", "CSS", "SQL", "PostgreSQL", "MySQL", "MongoDB", "Docker",
    "Kubernetes", "AWS", "GCP", "Azure", "Git", "Linux", "C++", "Java",
    "C#", "Go", "Rust", "GraphQL", "REST", "REST APIs", "CI/CD", "Terraform",
    "Redis", "FastAPI", "Django", "Flask", "PyTorch", "TensorFlow", "Pandas",
    "Spark", "Apache Spark", "Airflow", "Kafka", "Snowflake", "dbt",
    "Machine Learning", "ML", "AI", "Artificial Intelligence", "Deep Learning",
    "Linux", "Prometheus", "Financial Modeling", "Excel", "Valuation",
    "Forecasting", "B2B Sales", "Contract Negotiation", "Lead Generation",
    "Talent Acquisition", "Sourcing", "Applicant Tracking Systems",
]

def _build_known_skills() -> dict[str, str]:
    """Build normalized map (lowercase -> canonical casing) of known skills."""
    skills_map: dict[str, str] = {}

    # 1. Base keywords
    for skill in COMMON_TECH_KEYWORDS:
        s = skill.strip()
        if s:
            skills_map[s.lower()] = s

    # 2. Extract skills from DETERMINISTIC_JDS if available
    try:
        from ml_pipeline.generate_jds import DETERMINISTIC_JDS
        for jds in DETERMINISTIC_JDS.values():
            for jd in jds:
                for req in str(jd.get("required_skills", "")).split("|"):
                    s = req.strip()
                    if s and s.lower() not in skills_map:
                        skills_map[s.lower()] = s
                for pref in str(jd.get("preferred_skills", "")).split("|"):
                    s = pref.strip()
                    if s and s.lower() not in skills_map:
                        skills_map[s.lower()] = s
    except Exception:
        pass

    return skills_map


KNOWN_SKILLS: dict[str, str] = _build_known_skills()

EDU_SCORES: dict[str, float] = {
    "PhD": 100.0,
    "Masters": 85.0,
    "Bachelors": 70.0,
    "Diploma": 50.0,
    "None": 0.0,
}


@dataclass
class JobSpec:
    """Structured job specification."""

    title: str = ""
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    min_experience_years: float = 0.0
    embedding: Optional[np.ndarray] = None


@dataclass
class CandidateProfile:
    """Structured representation of an applicant extracted from resume."""

    candidate_id: str
    technical_skills: list[str] = field(default_factory=list)
    total_years_experience: float = 0.0
    highest_degree: Literal["PhD", "Masters", "Bachelors", "Diploma", "None"] = "None"
    embedding: Optional[np.ndarray] = None
    experience_phrases: list[str] = field(default_factory=list)


def scrub_pii(text: str) -> str:
    """Deterministically redact emails, phone numbers, and URLs from unstructured text.

    Redacts:
      - URL: https?://[^\\s]+ -> [REDACTED_URL]
      - Email: [A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,} -> [REDACTED_EMAIL]
      - Phone: (\\+?\\d{1,3}[-.\\s]?)?\\(?\\d{3}\\)?[-.\\s]?\\d{3}[-.\\s]?\\d{4} -> [REDACTED_PHONE]
    """
    text = re.sub(r"https?://[^\s]+", "[REDACTED_URL]", text)
    text = re.sub(
        r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
        "[REDACTED_EMAIL]",
        text,
    )
    text = re.sub(
        r"(\+?\d{1,3}[\-.\s]?)?\(?\d{3}\)?[\-.\s]?\d{3}[\-.\s]?\d{4}",
        "[REDACTED_PHONE]",
        text,
    )
    return text


def extract_text_from_file(file_path: str | Path) -> str:
    """Extract plain text from .pdf or .txt file.

    Args:
        file_path: Path to the resume file.

    Returns:
        Extracted text string.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages_text: list[str] = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages_text.append(t)
        return "\n\n".join(pages_text).strip()
    elif suffix in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    else:
        raise ValueError(
            f"Unsupported resume format '{suffix}' for {path.name}. Supported formats: .pdf, .txt"
        )


# Canonical domain concept groups and synonym clusters for semantic skill matching
SKILL_CONCEPT_MAP: dict[str, str] = {
    "ml": "concept_ml",
    "machine learning": "concept_ml",
    "predictive modeling": "concept_ml",
    "ai": "concept_ai",
    "artificial intelligence": "concept_ai",
    "llm": "concept_llm",
    "large language model": "concept_llm",
    "llm integration": "concept_llm_integration",
    "integrated local llm inference": "concept_llm_integration",
    "local llm inference": "concept_llm_integration",
    "llm inference": "concept_llm_integration",
    "language model integration": "concept_llm_integration",
    "mistral": "concept_llm_integration",
    "ollama": "concept_llm_integration",
    "react": "concept_react",
    "react.js": "concept_react",
    "reactjs": "concept_react",
    "node": "concept_node",
    "node.js": "concept_node",
    "nodejs": "concept_node",
    "next": "concept_next",
    "next.js": "concept_next",
    "nextjs": "concept_next",
    "ts": "concept_ts",
    "typescript": "concept_ts",
    "js": "concept_js",
    "javascript": "concept_js",
    "py": "concept_py",
    "python": "concept_py",
    "docker": "concept_docker",
    "containerization": "concept_docker",
    "k8s": "concept_k8s",
    "kubernetes": "concept_k8s",
    "postgres": "concept_postgres",
    "postgresql": "concept_postgres",
    "aws": "concept_aws",
    "amazon web services": "concept_aws",
    "gcp": "concept_gcp",
    "google cloud": "concept_gcp",
    "google cloud platform": "concept_gcp",
    "azure": "concept_azure",
    "microsoft azure": "concept_azure",
    "ci/cd": "concept_cicd",
    "cicd": "concept_cicd",
    "continuous integration": "concept_cicd",
    "continuous deployment": "concept_cicd",
    "tf": "concept_tensorflow",
    "tensorflow": "concept_tensorflow",
    "pytorch": "concept_pytorch",
    "torch": "concept_pytorch",
    "nlp": "concept_nlp",
    "natural language processing": "concept_nlp",
    "cv": "concept_computervision",
    "computer vision": "concept_computervision",
    "graphql": "concept_graphql",
    "gql": "concept_graphql",
    "mongo": "concept_mongodb",
    "mongodb": "concept_mongodb",
}


def _contains_skill(text: str, skill: str) -> bool:
    """Check if skill chip / term exists in text using word boundary patterns."""
    escaped = re.escape(skill)
    if re.match(r"^[\w\s]+$", skill):
        pattern = rf"\b{escaped}\b"
    else:
        pattern = rf"(?<![a-zA-Z0-9]){escaped}(?![a-zA-Z0-9+#])"
    return bool(re.search(pattern, text, re.IGNORECASE))


PREFERRED_SKILLS_PATTERN = re.compile(
    r"(?:preferred\s*skills?|nice\s*to\s*have|bonus\s*skills?):\s*([^.\n]+)",
    re.IGNORECASE,
)


def extract_preferred_skills_from_job_spec(job_spec_text: str) -> list[str]:
    """Extract preferred or nice-to-have skills from Job Specification text string."""
    found_skills: dict[str, str] = {}
    pref_match = PREFERRED_SKILLS_PATTERN.search(job_spec_text)
    if pref_match:
        section = pref_match.group(1).strip()
        tokens = re.split(r"[,|;]|\band\b", section)
        for token in tokens:
            t = token.strip()
            if t and len(t) < 40:
                found_skills[t.lower()] = t

    return list(found_skills.values())


def extract_skills_from_job_spec(job_spec_text: str) -> list[str]:
    """Extract required skills from a Job Specification text string.

    Supports explicit headers (e.g. 'Required Skills: Python, SQL'),
    free-text mentions matching known skill catalogs, and role title defaults.
    """
    found_skills: dict[str, str] = {}
    preferred_skills = {s.lower() for s in extract_preferred_skills_from_job_spec(job_spec_text)}

    # 1. Check for explicit labeled sections
    req_match = re.search(
        r"(?:required\s*skills?|skills?|must\s*have):\s*([^.\n]+)",
        job_spec_text,
        re.IGNORECASE,
    )
    if req_match:
        section = req_match.group(1).strip()
        tokens = re.split(r"[,|;]|\band\b", section)
        for token in tokens:
            t = token.strip()
            if t and len(t) < 40 and t.lower() not in preferred_skills:
                found_skills[t.lower()] = t

    # 2. Check for skills mentioned in text from KNOWN_SKILLS catalog
    # If explicit sections exist, exclude preferred section from required search
    text_to_search = job_spec_text
    pref_section_match = PREFERRED_SKILLS_PATTERN.search(job_spec_text)
    if pref_section_match:
        text_to_search = (
            job_spec_text[: pref_section_match.start()]
            + job_spec_text[pref_section_match.end() :]
        )

    for norm_skill, canonical in KNOWN_SKILLS.items():
        if (
            norm_skill not in found_skills
            and norm_skill not in preferred_skills
            and _contains_skill(text_to_search, canonical)
        ):
            found_skills[norm_skill] = canonical

    # 3. If no skills found, check if Job Spec matches any title in synthetic JDs
    if not found_skills:
        try:
            from ml_pipeline.generate_jds import DETERMINISTIC_JDS
            lower_jd = job_spec_text.lower().strip()
            for jds in DETERMINISTIC_JDS.values():
                for jd in jds:
                    title = str(jd.get("title", "")).lower()
                    if lower_jd in title or title in lower_jd:
                        req_str = str(jd.get("required_skills", ""))
                        for s in req_str.split("|"):
                            s_clean = s.strip()
                            if s_clean and s_clean.lower() not in preferred_skills:
                                found_skills[s_clean.lower()] = s_clean
                        break
                if found_skills:
                    break
        except Exception:
            pass

    return list(found_skills.values())


# Backwards compatibility alias
extract_skills_from_jd = extract_skills_from_job_spec


def extract_skills_from_resume(
    resume_text: str,
    pool_skills: Optional[list[str]] = None,
) -> list[str]:
    """Extract verified technical skills from resume text.

    Args:
        resume_text: Sanitized candidate resume text.
        pool_skills: Optional prioritized skills (e.g. required skills from JD).

    Returns:
        List of identified skills with canonical casing.
    """
    candidate_skills: dict[str, str] = {}

    # Prioritize checking pool skills (from JD)
    if pool_skills:
        for skill in pool_skills:
            s_clean = skill.strip()
            if s_clean and _contains_skill(resume_text, s_clean):
                candidate_skills[s_clean.lower()] = s_clean

    # Also check known tech skills catalog
    for norm_skill, canonical in KNOWN_SKILLS.items():
        if norm_skill not in candidate_skills and _contains_skill(resume_text, canonical):
            candidate_skills[norm_skill] = canonical

    return list(candidate_skills.values())


def extract_experience_years(text: str) -> float:
    """Extract total years of experience using deterministic heuristics."""
    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)(?:\s+(?:of\s+)?experience|\s+(?:of\s+)?exp)?",
        r"(?:experience|exp):\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)",
    ]
    max_years = 0.0
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                yrs = float(match.group(1))
                if 0.0 < yrs < 60.0 and yrs > max_years:
                    max_years = yrs
            except ValueError:
                pass
    return max_years


def extract_degree(text: str) -> Literal["PhD", "Masters", "Bachelors", "Diploma", "None"]:
    """Extract highest education tier from resume text."""
    if re.search(r"\b(ph\.?d|doctor of philosophy|doctorate)\b", text, re.IGNORECASE):
        return "PhD"
    if re.search(r"\b(master(?:'s)?|m\.?s\.?|m\.?tech|mba|mca|m\.?sc)\b", text, re.IGNORECASE):
        return "Masters"
    if re.search(
        r"\b(bachelor(?:'s)?|b\.?s\.?|b\.?tech|b\.?e\.?|bca|b\.?sc|bba)\b",
        text,
        re.IGNORECASE,
    ):
        return "Bachelors"
    if re.search(r"\b(diploma|associate(?:'s)?)\b", text, re.IGNORECASE):
        return "Diploma"
    return "None"


def _match_target_skills(
    targets: list[str],
    clean_cand: list[str],
    consumed_cand_indices: set[int],
    exp_phrases: list[str],
) -> tuple[list[str], list[str]]:
    """Match a target skill list against candidate skills and experience phrases."""
    matched: list[str] = []
    missing: list[str] = []

    for target in targets:
        target_norm = target.lower()
        target_concept = SKILL_CONCEPT_MAP.get(target_norm)

        # 1. Exact or case-insensitive match
        found_exact = False
        for i, cand in enumerate(clean_cand):
            if i in consumed_cand_indices:
                continue
            if target_norm == cand.lower():
                matched.append(target)
                consumed_cand_indices.add(i)
                for j, other in enumerate(clean_cand):
                    other_norm = other.lower()
                    if other_norm == target_norm or (
                        target_concept and SKILL_CONCEPT_MAP.get(other_norm) == target_concept
                    ):
                        consumed_cand_indices.add(j)
                found_exact = True
                break

        if found_exact:
            continue

        # 2. Concept synonym match
        found_synonym = False
        if target_concept:
            for i, cand in enumerate(clean_cand):
                if i in consumed_cand_indices:
                    continue
                cand_concept = SKILL_CONCEPT_MAP.get(cand.lower())
                if cand_concept and cand_concept == target_concept:
                    matched.append(f"{target} (via {cand})")
                    consumed_cand_indices.add(i)
                    for j, other in enumerate(clean_cand):
                        other_norm = other.lower()
                        if other_norm == cand.lower() or (
                            target_concept and SKILL_CONCEPT_MAP.get(other_norm) == target_concept
                        ):
                            consumed_cand_indices.add(j)
                    found_synonym = True
                    break

        if found_synonym:
            continue

        # 3. Project / Experience phrase match
        found_phrase = False
        if exp_phrases:
            for phrase in exp_phrases:
                p_norm = phrase.lower()
                if _contains_skill(p_norm, target) or (
                    target_concept
                    and any(
                        _contains_skill(p_norm, k)
                        for k, c in SKILL_CONCEPT_MAP.items()
                        if c == target_concept
                    )
                ):
                    matched.append(f"{target} (via {phrase})")
                    found_phrase = True
                    for j, other in enumerate(clean_cand):
                        other_norm = other.lower()
                        if other_norm == target_norm or (
                            target_concept and SKILL_CONCEPT_MAP.get(other_norm) == target_concept
                        ):
                            consumed_cand_indices.add(j)
                    break

        if found_phrase:
            continue

        missing.append(target)

    return matched, missing


def _normalize_target_skills(
    required_skills: Optional[list[str]], preferred_skills: Optional[list[str]]
) -> tuple[list[str], list[str]]:
    """Deduplicate required and preferred skills case-insensitively, excluding required from preferred."""
    seen_req: set[str] = set()
    clean_req: list[str] = []
    for s in (required_skills or []):
        s_clean = s.strip()
        if s_clean and s_clean.lower() not in seen_req:
            clean_req.append(s_clean)
            seen_req.add(s_clean.lower())

    seen_pref: set[str] = set()
    clean_pref: list[str] = []
    for s in (preferred_skills or []):
        s_clean = s.strip()
        if s_clean and s_clean.lower() not in seen_pref and s_clean.lower() not in seen_req:
            clean_pref.append(s_clean)
            seen_pref.add(s_clean.lower())

    return clean_req, clean_pref


def canonicalize_skills(
    skills: Optional[list[str]], preferred_keywords: Optional[list[str]] = None
) -> list[str]:
    """Canonicalize skills by domain concepts, deduplicating synonyms (e.g. ML and Machine Learning)."""
    pref_norms = {k.strip().lower() for k in (preferred_keywords or []) if k.strip()}
    seen_concepts: dict[str, str] = {}
    seen_norm: set[str] = set()
    result: list[str] = []

    for s in (skills or []):
        clean = s.strip()
        if not clean:
            continue
        norm = clean.lower()
        if norm in seen_norm:
            continue

        concept = SKILL_CONCEPT_MAP.get(norm)
        if concept:
            if concept in seen_concepts:
                existing = seen_concepts[concept]
                if norm in pref_norms and existing.lower() not in pref_norms:
                    idx = result.index(existing)
                    result[idx] = clean
                    seen_concepts[concept] = clean
                    seen_norm.remove(existing.lower())
                    seen_norm.add(norm)
                continue
            seen_concepts[concept] = clean
            seen_norm.add(norm)
            result.append(clean)
        else:
            seen_norm.add(norm)
            result.append(clean)

    return result


def _evaluate_candidate_skills(
    required_skills: Optional[list[str]],
    preferred_skills: Optional[list[str]],
    candidate_skills: Optional[list[str]],
    experience_phrases: Optional[list[str]],
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str], set[int]]:
    """Normalize target skills and match required/preferred skills against candidate competencies."""
    clean_req, clean_pref = _normalize_target_skills(required_skills, preferred_skills)
    clean_cand = canonicalize_skills(candidate_skills, clean_req + clean_pref)
    exp_phrases = [p.strip() for p in (experience_phrases or []) if p.strip()]

    consumed_cand_indices: set[int] = set()
    matched_req, missing_req = _match_target_skills(
        clean_req, clean_cand, consumed_cand_indices, exp_phrases
    )
    matched_pref, missing_pref = _match_target_skills(
        clean_pref, clean_cand, consumed_cand_indices, exp_phrases
    )
    return (
        clean_req,
        clean_pref,
        clean_cand,
        matched_req,
        matched_pref,
        missing_req,
        consumed_cand_indices,
    )


def compute_skill_audit(
    required_skills: list[str],
    candidate_skills: list[str],
    experience_phrases: Optional[list[str]] = None,
    preferred_skills: Optional[list[str]] = None,
) -> dict[str, list[str]]:
    """Compute matched, missing, and bonus skill breakdowns.

    Evaluates both required and preferred skills against candidate competencies.
    """
    (
        clean_req,
        clean_pref,
        clean_cand,
        matched_req,
        matched_pref,
        missing_req,
        consumed_cand_indices,
    ) = _evaluate_candidate_skills(
        required_skills, preferred_skills, candidate_skills, experience_phrases
    )

    # Bonus skills: unique candidate skills not consumed by required or preferred skills
    req_norm_set = {s.lower() for s in clean_req}
    pref_norm_set = {s.lower() for s in clean_pref}
    req_concepts = {
        SKILL_CONCEPT_MAP.get(s.lower()) for s in clean_req if SKILL_CONCEPT_MAP.get(s.lower())
    }
    pref_concepts = {
        SKILL_CONCEPT_MAP.get(s.lower()) for s in clean_pref if SKILL_CONCEPT_MAP.get(s.lower())
    }

    bonus: list[str] = []
    seen_bonus_norm: set[str] = set()
    seen_bonus_concepts: set[str] = set()
    for i, cand in enumerate(clean_cand):
        cand_norm = cand.lower()
        cand_concept = SKILL_CONCEPT_MAP.get(cand_norm)
        is_covered = (
            cand_norm in req_norm_set
            or cand_norm in pref_norm_set
            or (cand_concept and (cand_concept in req_concepts or cand_concept in pref_concepts))
        )
        if (
            i not in consumed_cand_indices
            and not is_covered
            and cand_norm not in seen_bonus_norm
            and (not cand_concept or cand_concept not in seen_bonus_concepts)
        ):
            bonus.append(cand)
            seen_bonus_norm.add(cand_norm)
            if cand_concept:
                seen_bonus_concepts.add(cand_concept)

    return {
        "matched": matched_req + matched_pref,
        "missing": missing_req,
        "bonus": bonus,
        "matched_required": matched_req,
        "matched_preferred": matched_pref,
    }


def compute_composite_score(
    profile: CandidateProfile,
    job_spec: JobSpec,
    weights: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """Calculate multi-factor composite score.

    Formula:
      S = (W_skills * S_skills) + (W_exp * S_exp) + (W_edu * S_edu)
      Default weights: W_skills=0.5, W_exp=0.3, W_edu=0.2

    Components:
      Coverage_req  = len(matched_req) / len(required_skills) (1.0 if empty)
      Coverage_pref = len(matched_pref) / len(preferred_skills) (0.0 if empty)
      SkillMatch:
        - both: min(1.0, Coverage_req + 0.15 * Coverage_pref)
        - req only: Coverage_req
        - pref only: Coverage_pref
        - neither: 1.0
      S_skills = 0.70 * non_neg_cosine + 0.30 * SkillMatch (scaled 0..100)
      S_exp    = min(100, (actual_years / required_years) * 100)
      S_edu    = PhD: 100, Masters: 85, Bachelors: 70, Diploma: 50, None: 0

    Uses clean domain dataclasses (CandidateProfile and JobSpec).
    """
    candidate_skills = profile.technical_skills
    actual_exp_years = profile.total_years_experience
    degree = profile.highest_degree

    required_skills = job_spec.required_skills
    preferred_skills = getattr(job_spec, "preferred_skills", []) or []
    required_exp_years = job_spec.min_experience_years

    cosine_sim = 0.0
    if profile.embedding is not None and job_spec.embedding is not None:
        norm_p = np.linalg.norm(profile.embedding)
        norm_j = np.linalg.norm(job_spec.embedding)
        if norm_p > 0 and norm_j > 0:
            cosine_sim = float(np.dot(profile.embedding, job_spec.embedding) / (norm_p * norm_j))

    if weights is None:
        weights = {"skills": 0.5, "exp": 0.3, "edu": 0.2}

    w_skills = weights.get("skills", 0.5)
    w_exp = weights.get("exp", 0.3)
    w_edu = weights.get("edu", 0.2)

    # 1. Education Score (S_edu)
    s_edu = EDU_SCORES.get(degree, 0.0)

    # 2. Experience Score (S_exp)
    if required_exp_years > 0.0:
        s_exp = min(100.0, max(0.0, (actual_exp_years / required_exp_years) * 100.0))
    else:
        s_exp = 100.0

    # 3. Requirement Recall Coverage & Additive Preferred Skill Boost
    exp_phrases = [
        p.strip() for p in (getattr(profile, "experience_phrases", None) or []) if p.strip()
    ]
    (
        clean_req,
        clean_pref,
        _,
        matched_req,
        matched_pref,
        _,
        _,
    ) = _evaluate_candidate_skills(
        required_skills, preferred_skills, candidate_skills, exp_phrases
    )

    cov_req = (len(matched_req) / len(clean_req)) if clean_req else 1.0
    cov_pref = (len(matched_pref) / len(clean_pref)) if clean_pref else 0.0

    if clean_req and clean_pref:
        # Additive bonus: required skills form the baseline; preferred skills provide up to +15% additive lift, capped at 1.0
        skill_match = min(1.0, cov_req + 0.15 * cov_pref)
    elif clean_req:
        skill_match = cov_req
    elif clean_pref:
        skill_match = cov_pref
    else:
        skill_match = 1.0

    non_neg_cosine = max(0.0, min(1.0, float(cosine_sim)))
    s_skills = min(100.0, max(0.0, (0.70 * non_neg_cosine + 0.30 * skill_match) * 100.0))

    # 4. Composite Score
    raw_composite = (w_skills * s_skills) + (w_exp * s_exp) + (w_edu * s_edu)

    return {
        "s_skills": round(s_skills, 1),
        "s_exp": round(s_exp, 1),
        "s_edu": round(s_edu, 1),
        "composite_score": round(raw_composite, 1),
    }


def format_table(results: list[dict[str, Any]]) -> str:
    """Format candidate results into an aligned ASCII/Markdown ranked table.

    Columns:
      Rank | Candidate | Score | Matched Skills | Missing Skills
    """
    if not results:
        return "No resumes found to screen."

    headers = ["Rank", "Candidate", "Score", "Matched Skills", "Missing Skills"]
    col_widths = [len(h) for h in headers]

    rows: list[list[str]] = []
    for r in results:
        rank_str = str(r.get("rank", ""))
        cand_str = str(r.get("candidate", ""))
        score_str = f"{r.get('score', 0.0):.1f}"
        matched_str = ", ".join(r.get("matched_skills", [])) or "None"
        missing_str = ", ".join(r.get("missing_skills", [])) or "None"

        row = [rank_str, cand_str, score_str, matched_str, missing_str]
        rows.append(row)
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Format header, divider, and data rows
    header_line = " | ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers)))
    divider_line = " | ".join("-" * col_widths[i] for i in range(len(headers)))
    data_lines = [
        " | ".join(f"{row[i]:<{col_widths[i]}}" for i in range(len(headers)))
        for row in rows
    ]

    return "\n".join([header_line, divider_line] + data_lines)


def screen_resumes(
    job_spec_text: str = "",
    resumes_dir: str | Path = "",
    top_n: int = 5,
    model_path: str | Path = DEFAULT_OUTPUT_PATH,
    weights: Optional[dict[str, float]] = None,
    jd_text: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Screen and rank resumes in directory against Job Specification.

    Args:
        job_spec_text: Job specification free text.
        resumes_dir: Directory containing .pdf and .txt resumes.
        top_n: Number of ranked candidates to return.
        model_path: Path to ONNX model.
        weights: Optional scoring weights.
        jd_text: Backward-compatible alias for job_spec_text.

    Returns:
        List of candidate ranking dictionaries.
    """
    effective_spec_text = job_spec_text or jd_text or ""
    res_path = Path(resumes_dir)
    if not res_path.exists() or not res_path.is_dir():
        raise FileNotFoundError(f"Resume directory not found: {resumes_dir}")

    # Discover candidate files (.pdf, .txt)
    supported_exts = {".pdf", ".txt"}
    resume_files = sorted(
        [p for p in res_path.iterdir() if p.is_file() and p.suffix.lower() in supported_exts]
    )

    if not resume_files:
        logger.warning("No supported (.pdf, .txt) resume files found in %s", resumes_dir)
        return []

    # Ensure ONNX model is available
    onnx_file = Path(model_path)
    if not onnx_file.exists():
        logger.info("ONNX model '%s' not found. Exporting quantized model...", onnx_file)
        export_onnx(output_path=onnx_file)

    session = load_onnx_session(onnx_file)

    # 1. Embed Job Specification
    job_spec_embedding = embed_texts([effective_spec_text], session=session)[0]
    required_skills = extract_skills_from_job_spec(effective_spec_text)
    preferred_skills = extract_preferred_skills_from_job_spec(effective_spec_text)
    required_exp_years = extract_experience_years(effective_spec_text)
    job_spec = JobSpec(
        title=effective_spec_text.splitlines()[0][:60] if effective_spec_text else "",
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        min_experience_years=required_exp_years,
        embedding=job_spec_embedding,
    )

    # 2. Process each resume
    candidates_data: list[dict[str, Any]] = []
    texts_to_embed: list[str] = []
    profiles: list[CandidateProfile] = []

    for file_path in resume_files:
        try:
            raw_text = extract_text_from_file(file_path)
        except Exception as e:
            logger.error("Failed to read %s: %s", file_path.name, e)
            continue

        # PII Scrubbing
        scrubbed = scrub_pii(raw_text)
        cand_skills = extract_skills_from_resume(scrubbed, pool_skills=required_skills)
        exp_years = extract_experience_years(scrubbed)
        degree = extract_degree(scrubbed)
        experience_phrases = [
            s.strip()
            for s in re.split(r"[\n\.\;]", scrubbed)
            if len(s.strip()) > 15
        ]

        texts_to_embed.append(scrubbed)
        profiles.append(
            CandidateProfile(
                candidate_id=file_path.name,
                technical_skills=cand_skills,
                total_years_experience=exp_years,
                highest_degree=degree,
                experience_phrases=experience_phrases,
            )
        )

    if not texts_to_embed:
        return []

    # Batch embed candidate texts using the same ONNX session
    resume_embeddings = embed_texts(texts_to_embed, session=session)

    # 3. Calculate Scores for each candidate
    job_spec.embedding = job_spec_embedding
    for profile, cand_emb in zip(profiles, resume_embeddings):
        profile.embedding = cand_emb
        audit = compute_skill_audit(
            job_spec.required_skills,
            profile.technical_skills,
            profile.experience_phrases,
            job_spec.preferred_skills,
        )
        scoring_res = compute_composite_score(
            profile=profile,
            job_spec=job_spec,
            weights=weights,
        )

        candidates_data.append({
            "candidate": profile.candidate_id,
            "score": scoring_res["composite_score"],
            "matched_skills": audit["matched"],
            "missing_skills": audit["missing"],
            "bonus_skills": audit["bonus"],
            "s_skills": scoring_res["s_skills"],
            "s_exp": scoring_res["s_exp"],
            "s_edu": scoring_res["s_edu"],
            "exp_years": profile.total_years_experience,
            "degree": profile.highest_degree,
        })

    # Sort descending by composite score; tie-break deterministically by filename
    candidates_data.sort(
        key=lambda c: (-c["score"], c["candidate"])
    )

    # Assign 1-based ranks
    for rank, cand in enumerate(candidates_data, start=1):
        cand["rank"] = rank

    # Return top N candidates
    return candidates_data[:top_n]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Standalone CLI Screener for Academic Viva Demonstration."
    )
    parser.add_argument(
        "--jd",
        type=str,
        required=True,
        help="Free-text Job Specification string (required)",
    )
    parser.add_argument(
        "--resumes",
        type=str,
        required=True,
        help="Path to directory containing .pdf or .txt resume files (required)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of ranked candidates to display (default: 5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Path to ONNX model (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """Main CLI entrypoint."""
    args = parse_args(argv)
    results = screen_resumes(
        job_spec_text=args.jd,
        resumes_dir=args.resumes,
        top_n=args.top,
        model_path=args.model,
    )
    table_output = format_table(results)
    print(table_output)


if __name__ == "__main__":
    main()
