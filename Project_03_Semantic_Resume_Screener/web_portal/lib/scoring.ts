import { generateSkillVector, SKILL_CONCEPT_MAP } from './providers';

export type Weights = { skills: number; exp: number; edu: number };
export type EduLevel = "PhD" | "Masters" | "Bachelors" | "Diploma" | "None";

export type CandidateProfile = {
  candidate_id: string;
  technical_skills: string[];
  total_years_experience: number;
  highest_degree: EduLevel;
  embedding: number[];
  project_sentences?: string[];
  experience_summary?: Array<{
    title?: string;
    duration_years?: number;
    technologies?: string[];
    description?: string;
  }> | string[];
};

export type JobSpec = {
  required_skills: string[];
  preferred_skills?: string[];
  min_experience_years: number;
  embedding: number[];
};

/**
 * Formats a Job Specification into a canonical structured text document
 * for dense embedding projection.
 */
export function formatJobSpecText(spec: {
  title?: string;
  required_skills?: string[];
  preferred_skills?: string[];
  education_level?: string;
  min_experience_years?: number;
}): string {
  return (
    [
      spec.title || '',
      spec.required_skills?.length
        ? `Required Skills: ${spec.required_skills.join(', ')}`
        : '',
      spec.preferred_skills?.length
        ? `Preferred Skills: ${spec.preferred_skills.join(', ')}`
        : '',
      spec.education_level ? `Education: ${spec.education_level}` : '',
      spec.min_experience_years
        ? `Minimum Experience: ${spec.min_experience_years} years`
        : '',
    ]
      .filter(Boolean)
      .join('\n') || 'Job Specification'
  );
}

export type ScoredCandidate = CandidateProfile & {
  composite_score: number;
  skill_audit: {
    matched: string[];
    missing: string[];
    bonus: string[];
    matched_required?: string[];
    matched_preferred?: string[];
  };
  s_skills: number;
  s_exp: number;
  s_edu: number;
};

export interface RecommendationTier {
  label: 'Strong Match' | 'Potential Match' | 'Not Recommended';
  tier: 'strong' | 'potential' | 'unrecommended';
  badgeClass: string;
  ringColor: string;
}

const ALL_WEIGHT_KEYS: (keyof Weights)[] = ['skills', 'exp', 'edu'];

/**
 * Maps composite score to recommendation tier and visual badge colors.
 * - >= 80: "Strong Match"
 * - 60–79.9: "Potential Match"
 * - < 60: "Not Recommended"
 */
export function getRecommendationTier(score: number): RecommendationTier {
  if (score >= 80) {
    return {
      tier: 'strong',
      label: 'Strong Match',
      badgeClass: 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-500/10 dark:text-emerald-400 dark:border-emerald-500/30',
      ringColor: '#10b981',
    };
  }
  if (score >= 60) {
    return {
      tier: 'potential',
      label: 'Potential Match',
      badgeClass: 'bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-500/10 dark:text-amber-400 dark:border-amber-500/30',
      ringColor: '#f59e0b',
    };
  }
  return {
    tier: 'unrecommended',
    label: 'Not Recommended',
    badgeClass: 'bg-rose-50 text-rose-700 border-rose-200 dark:bg-rose-500/10 dark:text-rose-400 dark:border-rose-500/30',
    ringColor: '#f43f5e',
  };
}

export const getRecommendation = getRecommendationTier;

/**
 * Redistributes weights proportionally when one slider is adjusted so that
 * the total sum remains exactly 1.0 (100%).
 * Supports both decimal (0..1) and percentage (0..100) input formats.
 */
export function redistributeWeights(
  current: Weights,
  changedKey: keyof Weights,
  newValue: number
): Weights {
  // Normalize current weights if provided in percentage scale
  const normSkills = current.skills > 1 ? current.skills / 100 : current.skills;
  const normExp = current.exp > 1 ? current.exp / 100 : current.exp;
  const normEdu = current.edu > 1 ? current.edu / 100 : current.edu;

  const currentNorm: Weights = {
    skills: normSkills,
    exp: normExp,
    edu: normEdu,
  };

  // Normalize newValue to 0.0 - 1.0
  let targetVal = newValue > 1.0 ? newValue / 100 : newValue;
  targetVal = Math.max(0, Math.min(1, targetVal));
  targetVal = Math.round(targetVal * 100) / 100;

  const remaining = Math.round((1.0 - targetVal) * 100) / 100;
  const otherKeys = ALL_WEIGHT_KEYS.filter((k) => k !== changedKey);
  const key1 = otherKeys[0];
  const key2 = otherKeys[1];

  let redistributedWeight1 = 0;
  let redistributedWeight2 = 0;

  if (remaining <= 0) {
    redistributedWeight1 = 0;
    redistributedWeight2 = 0;
  } else {
    const sumOthers = currentNorm[key1] + currentNorm[key2];
    if (sumOthers <= 0.0001) {
      redistributedWeight1 = Math.round((remaining / 2) * 100) / 100;
      redistributedWeight2 = Math.round((remaining - redistributedWeight1) * 100) / 100;
    } else {
      const ratio1 = currentNorm[key1] / sumOthers;
      redistributedWeight1 = Math.round((remaining * ratio1) * 100) / 100;
      redistributedWeight2 = Math.round((remaining - redistributedWeight1) * 100) / 100;
    }
  }

  // Exact 1.0 total compensation
  const total = Math.round((targetVal + redistributedWeight1 + redistributedWeight2) * 100) / 100;
  const diff = Math.round((1.0 - total) * 100) / 100;
  if (diff !== 0) {
    redistributedWeight2 = Math.max(0, Math.round((redistributedWeight2 + diff) * 100) / 100);
  }

  return {
    ...currentNorm,
    [changedKey]: targetVal,
    [key1]: Math.max(0, redistributedWeight1),
    [key2]: Math.max(0, redistributedWeight2),
  };
}

const EDU_SCORES: Record<EduLevel, number> = {
  PhD: 100,
  Masters: 85,
  Bachelors: 70,
  Diploma: 50,
  None: 0,
};

function round1(val: number): number {
  return Math.round(val * 10) / 10;
}

/**
 * Computes cosine similarity between two numeric vectors.
 * Returns 0.0 if vectors are empty, mismatched in length, or have zero magnitude.
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (!a || !b || a.length === 0 || b.length === 0 || a.length !== b.length) {
    return 0.0;
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (normA <= 0 || normB <= 0) {
    return 0.0;
  }

  const sim = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  return Math.min(1.0, Math.max(-1.0, sim));
}

interface SemanticMatchResult {
  matched: boolean;
  phrase: string;
  isExact: boolean;
  similarity: number;
  candidateIndex?: number;
}

function findBestCandidateSkillMatch(
  req: string,
  candSkills: string[],
  consumedIndices: Set<number>,
  threshold: number = 0.75
): SemanticMatchResult {
  const reqNorm = req.toLowerCase();
  const vReq = generateSkillVector(req);

  let bestIndex = -1;
  let bestPhrase = '';
  let bestSim = -1;

  for (let i = 0; i < candSkills.length; i++) {
    if (consumedIndices.has(i)) continue;
    const cand = candSkills[i];
    const candNorm = cand.toLowerCase();

    // Exact case-insensitive match
    if (reqNorm === candNorm) {
      return {
        matched: true,
        phrase: cand,
        isExact: true,
        similarity: 1.0,
        candidateIndex: i,
      };
    }

    const vCand = generateSkillVector(cand);
    const sim = cosineSimilarity(vReq, vCand);
    if (sim >= threshold && sim > bestSim) {
      bestSim = sim;
      bestIndex = i;
      bestPhrase = cand;
    }
  }

  if (bestIndex >= 0) {
    return {
      matched: true,
      phrase: bestPhrase,
      isExact: false,
      similarity: bestSim,
      candidateIndex: bestIndex,
    };
  }

  return { matched: false, phrase: '', isExact: false, similarity: 0 };
}

/**
 * Strips any internal parenthetical matching provenance (e.g. " (via ...)") from
 * a skill string so chips display clean canonical skill names in the UI.
 */
export function getSkillDisplayName(skill: string): string {
  const viaIdx = skill.search(/\s*\(via\s+/i);
  if (viaIdx !== -1) {
    return skill.slice(0, viaIdx).trim();
  }
  return skill.trim();
}

function findBestSentenceMatch(
  req: string,
  sentences: string[],
  threshold: number = 0.75
): SemanticMatchResult {
  const vReq = generateSkillVector(req);
  let bestPhrase = '';
  let bestSim = -1;

  for (const sentence of sentences) {
    const vSentence = generateSkillVector(sentence);
    const sim = cosineSimilarity(vReq, vSentence);
    if (sim >= threshold && sim > bestSim) {
      bestSim = sim;
      bestPhrase = sentence.replace(/\s*\([^)]*\)/g, '').trim();
    }
  }

  if (bestSim >= threshold) {
    return {
      matched: true,
      phrase: bestPhrase,
      isExact: false,
      similarity: bestSim,
    };
  }

  return { matched: false, phrase: '', isExact: false, similarity: 0 };
}

/**
 * Canonicalizes a list of skills using domain concepts (SKILL_CONCEPT_MAP).
 * Collapses synonyms (e.g. "ML" and "Machine Learning") into a single skill representation
 * so synonyms are never counted as two distinct skills.
 */
export function canonicalizeSkills(skills: string[], preferredKeywords?: string[]): string[] {
  const preferredNorms = new Set((preferredKeywords || []).map((k) => k.trim().toLowerCase()));

  const seenConcepts = new Map<string, string>();
  const seenNorm = new Set<string>();
  const result: string[] = [];

  for (const s of skills || []) {
    const clean = s.trim();
    if (!clean) continue;
    const norm = clean.toLowerCase();
    if (seenNorm.has(norm)) continue;

    const concept = SKILL_CONCEPT_MAP[norm];
    if (concept) {
      if (seenConcepts.has(concept)) {
        const existing = seenConcepts.get(concept)!;
        if (preferredNorms.has(norm) && !preferredNorms.has(existing.toLowerCase())) {
          const idx = result.indexOf(existing);
          if (idx >= 0) {
            result[idx] = clean;
            seenConcepts.set(concept, clean);
            seenNorm.delete(existing.toLowerCase());
            seenNorm.add(norm);
          }
        }
        continue;
      }
      seenConcepts.set(concept, clean);
      seenNorm.add(norm);
      result.push(clean);
    } else {
      seenNorm.add(norm);
      result.push(clean);
    }
  }

  return result;
}

function matchSkillTargets(
  targets: string[],
  candSkills: string[],
  matchedIndices: Set<number>,
  experiencePhrases: string[]
): { matched: string[]; missing: string[] } {
  const matched: string[] = [];
  const missing: string[] = [];

  for (const target of targets) {
    const targetNorm = target.toLowerCase();
    const targetConcept = SKILL_CONCEPT_MAP[targetNorm];

    const skillMatch = findBestCandidateSkillMatch(target, candSkills, matchedIndices);
    if (skillMatch.matched && skillMatch.candidateIndex !== undefined) {
      matchedIndices.add(skillMatch.candidateIndex);
      const matchedNorm = candSkills[skillMatch.candidateIndex].toLowerCase();
      const matchedConcept = SKILL_CONCEPT_MAP[matchedNorm] || targetConcept;

      // Mark ALL candidate skills that represent this target or concept as consumed
      candSkills.forEach((c, idx) => {
        const cNorm = c.toLowerCase();
        if (
          cNorm === matchedNorm ||
          cNorm === targetNorm ||
          (matchedConcept && SKILL_CONCEPT_MAP[cNorm] === matchedConcept) ||
          cosineSimilarity(generateSkillVector(c), generateSkillVector(target)) >= 0.75
        ) {
          matchedIndices.add(idx);
        }
      });

      if (skillMatch.isExact) {
        matched.push(target);
      } else {
        matched.push(`${target} (via ${skillMatch.phrase})`);
      }
      continue;
    }

    const sentenceMatch = findBestSentenceMatch(target, experiencePhrases);
    if (sentenceMatch.matched) {
      candSkills.forEach((c, idx) => {
        const cNorm = c.toLowerCase();
        if (
          cNorm === targetNorm ||
          (targetConcept && SKILL_CONCEPT_MAP[cNorm] === targetConcept) ||
          cosineSimilarity(generateSkillVector(c), generateSkillVector(target)) >= 0.75
        ) {
          matchedIndices.add(idx);
        }
      });
      matched.push(`${target} (via ${sentenceMatch.phrase})`);
      continue;
    }

    missing.push(target);
  }

  return { matched, missing };
}

/**
 * Scores a Candidate Profile against a Job Specification using the composite scoring formula:
 * S = (W_skills * S_skills) + (W_exp * S_exp) + (W_edu * S_edu)
 *
 * Implements pure local semantic skill matching (cosine >= 0.75), contextual candidate phrasings,
 * requirement recall coverage, and weighted preferred skills.
 */
export function scoreCandidate(
  profile: CandidateProfile,
  spec: JobSpec,
  weights: Weights
): ScoredCandidate {
  // 1. Education Score (S_edu)
  const s_edu = EDU_SCORES[profile.highest_degree] ?? 0;

  // 2. Experience Score (S_exp)
  let s_exp = 100;
  if (spec.min_experience_years > 0) {
    s_exp = Math.min(
      100,
      Math.max(0, (profile.total_years_experience / spec.min_experience_years) * 100)
    );
  }

  // 3. Semantic Skill Audit & Requirement Recall Coverage
  const reqSkillMap = new Map<string, string>();
  for (const s of spec.required_skills || []) {
    const clean = s.trim();
    if (clean && !reqSkillMap.has(clean.toLowerCase())) {
      reqSkillMap.set(clean.toLowerCase(), clean);
    }
  }
  const reqSkills = Array.from(reqSkillMap.values());

  const prefSkillMap = new Map<string, string>();
  for (const s of spec.preferred_skills || []) {
    const clean = s.trim();
    if (clean && !prefSkillMap.has(clean.toLowerCase()) && !reqSkillMap.has(clean.toLowerCase())) {
      prefSkillMap.set(clean.toLowerCase(), clean);
    }
  }
  const prefSkills = Array.from(prefSkillMap.values());

  // Deduplicate and canonicalize candidate skills using domain concepts so synonyms (e.g. "ML" and "Machine Learning")
  // do not count as two distinct skills
  const candSkills = canonicalizeSkills(
    profile.technical_skills || [],
    [...reqSkills, ...prefSkills]
  );

  // Candidate experience sentences from project_sentences or experience_summary
  const candidateExperiencePhrases: string[] = [];
  if (Array.isArray(profile.project_sentences)) {
    candidateExperiencePhrases.push(...profile.project_sentences);
  }
  if (Array.isArray(profile.experience_summary)) {
    for (const exp of profile.experience_summary) {
      if (typeof exp === 'string') {
        candidateExperiencePhrases.push(exp);
      } else if (exp && typeof exp === 'object') {
        if (exp.title) candidateExperiencePhrases.push(exp.title);
        if (exp.description) candidateExperiencePhrases.push(exp.description);
        if (Array.isArray(exp.technologies)) candidateExperiencePhrases.push(...exp.technologies);
      }
    }
  }

  // Track matched candidate skill indices to exclude them from bonus skills
  const matchedCandSkillIndices = new Set<number>();

  // 3a. Match Required Skills
  const { matched: matchedReq, missing: missingReq } = matchSkillTargets(
    reqSkills,
    candSkills,
    matchedCandSkillIndices,
    candidateExperiencePhrases
  );

  // 3b. Match Preferred Skills
  const { matched: matchedPref } = matchSkillTargets(
    prefSkills,
    candSkills,
    matchedCandSkillIndices,
    candidateExperiencePhrases
  );

  // Bonus skills: unique candidate skills NOT consumed by required or preferred skills
  const normalizedReqSkills = new Set(reqSkills.map((s) => s.toLowerCase()));
  const normalizedPrefSkills = new Set(prefSkills.map((s) => s.toLowerCase()));
  const reqConcepts = new Set(
    reqSkills.map((s) => SKILL_CONCEPT_MAP[s.toLowerCase()]).filter(Boolean)
  );
  const prefConcepts = new Set(
    prefSkills.map((s) => SKILL_CONCEPT_MAP[s.toLowerCase()]).filter(Boolean)
  );

  const bonus: string[] = [];
  const normalizedBonusSkills = new Set<string>();
  const bonusConcepts = new Set<string>();

  for (let i = 0; i < candSkills.length; i++) {
    const candSkillNormalized = candSkills[i].toLowerCase();
    const candConcept = SKILL_CONCEPT_MAP[candSkillNormalized];
    const isCoveredByReqOrPref =
      normalizedReqSkills.has(candSkillNormalized) ||
      normalizedPrefSkills.has(candSkillNormalized) ||
      (candConcept && (reqConcepts.has(candConcept) || prefConcepts.has(candConcept)));

    if (
      !matchedCandSkillIndices.has(i) &&
      !normalizedBonusSkills.has(candSkillNormalized) &&
      !isCoveredByReqOrPref &&
      (!candConcept || !bonusConcepts.has(candConcept))
    ) {
      bonus.push(candSkills[i]);
      normalizedBonusSkills.add(candSkillNormalized);
      if (candConcept) bonusConcepts.add(candConcept);
    }
  }

  // 3c. Pure Requirement Recall Coverage & Additive Preferred Skill Boost
  const covReq = reqSkills.length > 0 ? matchedReq.length / reqSkills.length : 1.0;
  const covPref = prefSkills.length > 0 ? matchedPref.length / prefSkills.length : 0.0;

  let skillMatchScore: number;
  if (reqSkills.length > 0 && prefSkills.length > 0) {
    // Additive bonus: required skills form the baseline; preferred skills provide up to +15% additive lift, capped at 1.0
    skillMatchScore = Math.min(1.0, covReq + 0.15 * covPref);
  } else if (reqSkills.length > 0) {
    skillMatchScore = covReq;
  } else if (prefSkills.length > 0) {
    skillMatchScore = covPref;
  } else {
    skillMatchScore = 1.0;
  }

  // 4. Semantic Similarity & S_skills
  const cosSim = cosineSimilarity(spec.embedding, profile.embedding);
  const nonNegCosine = Math.max(0, cosSim);
  const s_skills = Math.min(100, Math.max(0, (0.70 * nonNegCosine + 0.30 * skillMatchScore) * 100));

  // 5. Composite Score
  const composite_score = computeCompositeScore(s_skills, s_exp, s_edu, weights);

  return {
    ...profile,
    technical_skills: candSkills,
    composite_score,
    skill_audit: {
      matched: [...matchedReq, ...matchedPref],
      missing: missingReq,
      bonus,
      matched_required: matchedReq,
      matched_preferred: matchedPref,
    },
    s_skills: round1(s_skills),
    s_exp: round1(s_exp),
    s_edu: round1(s_edu),
  };
}

/**
 * Computes composite score from dimension scores and weights rounded to 1 decimal place.
 */
export function computeCompositeScore(
  s_skills: number,
  s_exp: number,
  s_edu: number,
  weights: Weights
): number {
  const raw =
    (weights.skills * s_skills) +
    (weights.exp * s_exp) +
    (weights.edu * s_edu);
  return round1(raw);
}

/**
 * Deterministic candidate sort comparator: descending composite score, tie-break by candidate_id.
 */
export function compareScoredCandidates(a: ScoredCandidate, b: ScoredCandidate): number {
  if (b.composite_score !== a.composite_score) {
    return b.composite_score - a.composite_score;
  }
  return a.candidate_id.localeCompare(b.candidate_id);
}

/**
 * Scores and ranks candidate profiles descending by composite_score.
 */
export function rankCandidates(
  profiles: CandidateProfile[],
  spec: JobSpec,
  weights: Weights
): ScoredCandidate[] {
  if (!profiles || profiles.length === 0) {
    return [];
  }

  const scored = profiles.map((p) => scoreCandidate(p, spec, weights));
  return scored.sort(compareScoredCandidates);
}

/**
 * Fast O(N) re-ranking when only scoring weights change.
 * Re-uses already computed s_skills, s_exp, and s_edu without repeating
 * vector calculations, string hashing, or semantic skill audits.
 */
export function reRankWithWeights(
  candidates: ScoredCandidate[],
  weights: Weights
): ScoredCandidate[] {
  if (!candidates || candidates.length === 0) {
    return [];
  }

  return candidates
    .map((c) => ({
      ...c,
      composite_score: computeCompositeScore(c.s_skills, c.s_exp, c.s_edu, weights),
    }))
    .sort(compareScoredCandidates);
}



