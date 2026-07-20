import { GoogleGenerativeAI, SchemaType } from '@google/generative-ai';
import {
  CandidateProfile,
  JobSpec,
  ScoredCandidate,
  Weights,
  EduLevel,
  rankCandidates,
  formatJobSpecText,
  canonicalizeSkills,
} from '@/lib/scoring';
import {
  EmbeddingProvider,
  GeminiProvider,
  MockEmbeddingProvider,
  TFIDFProvider,
  computeEmbeddingsWithFallback,
} from '@/lib/providers';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 60;

const BATCH_SIZE = 10;

interface RequestJobSpec {
  title?: string;
  required_skills?: string[];
  preferred_skills?: string[];
  min_experience_years?: number;
  education_level?: string;
  weights?: Weights;
}

interface RequestResume {
  candidate_id: string;
  text: string;
}

export interface CachedCandidateInput {
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
}

interface EvaluateRequestBody {
  jobSpec: RequestJobSpec;
  resumes?: RequestResume[];
  cachedCandidates?: CachedCandidateInput[];
}

interface ParsedLLMCandidate {
  candidate_id: string;
  technical_skills: string[];
  total_years_experience: number;
  highest_degree: string;
  experience_summary?: Array<{
    title: string;
    duration_years: number;
    technologies: string[];
  }>;
}

const COMMON_TECH_KEYWORDS = [
  'python', 'javascript', 'typescript', 'react', 'next.js', 'node.js',
  'html', 'css', 'sql', 'postgresql', 'mysql', 'mongodb', 'docker',
  'kubernetes', 'aws', 'gcp', 'azure', 'git', 'linux', 'c++', 'java',
  'c#', 'go', 'rust', 'graphql', 'rest', 'ci/cd', 'terraform', 'redis',
  'machine learning', 'ml', 'ai', 'artificial intelligence', 'deep learning',
  'data science', 'nlp', 'computer vision', 'pytorch', 'tensorflow',
];

/**
 * Consolidated education normalization helper that canonicalizes degree strings
 * or unstructured resume text into standard EduLevel ("PhD" | "Masters" | "Bachelors" | "Diploma" | "None").
 */
function normalizeEducation(input: string | undefined | null): EduLevel {
  if (!input) return 'None';
  const text = input.trim();
  if (!text) return 'None';

  if (text === 'PhD' || text === 'Masters' || text === 'Bachelors' || text === 'Diploma' || text === 'None') {
    return text;
  }

  const lower = text.toLowerCase();
  if (/\b(ph\.?d|doctor(?:ate)?|doctor of philosophy)\b/i.test(text) || lower.includes('phd') || lower.includes('doctor')) return 'PhD';
  if (/\b(master(?:'?s)?|m\.?s\.?|m\.?tech|mba|mca|m\.?sc)\b/i.test(text) || lower.includes('master') || lower.includes('mba')) return 'Masters';
  if (/\b(bachelor(?:'?s)?|b\.?s\.?|b\.?tech|b\.?e\.?|bca|b\.?sc|bba)\b/i.test(text) || lower.includes('bachelor') || lower.includes('btech')) return 'Bachelors';
  if (/\b(diploma|associate(?:'?s)?)\b/i.test(text) || lower.includes('diploma') || lower.includes('associate')) return 'Diploma';
  return 'None';
}

function extractExperienceYears(text: string): number {
  const patterns = [
    /(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)(?:\s+(?:of\s+)?experience|\s+(?:of\s+)?exp)?/gi,
    /(?:experience|exp):\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)/gi,
  ];

  let maxYears = 0;
  for (const pattern of patterns) {
    let match: RegExpExecArray | null;
    while ((match = pattern.exec(text)) !== null) {
      const yrs = parseFloat(match[1]);
      if (!isNaN(yrs) && yrs > maxYears && yrs < 60) {
        maxYears = yrs;
      }
    }
  }

  return maxYears;
}

function extractSkills(text: string, jobSpec: RequestJobSpec): string[] {
  const candidateSkills = new Set<string>();

  const pool = [
    ...(jobSpec.required_skills || []),
    ...(jobSpec.preferred_skills || []),
    ...COMMON_TECH_KEYWORDS,
  ];

  for (const skill of pool) {
    const trimmed = skill.trim();
    if (!trimmed) continue;
    const lowerSkill = trimmed.toLowerCase();

    // Special handling for Go / Golang to prevent false matches on English verb "go" or substrings (e.g. MongoDB, algorithms)
    if (lowerSkill === 'go' || lowerSkill === 'golang') {
      if (/\b(golang|go\s+language)\b/i.test(text)) {
        candidateSkills.add('Go');
      }
      continue;
    }

    // Special handling for C language to prevent false matches on single letter 'c'
    if (lowerSkill === 'c') {
      if (/\b(c\s+language|c\s+programming)\b/i.test(text) || /(^|[^a-zA-Z0-9#+.])C([^a-zA-Z0-9#+.]|$)/.test(text)) {
        candidateSkills.add('C');
      }
      continue;
    }

    // Strict boundary matching: prevent false substring inclusion
    const escaped = lowerSkill.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const skillRegex = new RegExp(`(^|[^a-zA-Z0-9])${escaped}([^a-zA-Z0-9+#]|$)`, 'i');
    if (skillRegex.test(text)) {
      candidateSkills.add(trimmed);
    }
  }

  // Explicit case-insensitive matching for concise AI and ML acronyms
  if (/\b(ai|artificial intelligence)\b/i.test(text)) {
    candidateSkills.add('AI');
  }
  if (/\b(ml|machine learning)\b/i.test(text)) {
    candidateSkills.add('Machine Learning');
  }

  const requestedSkills = [
    ...(jobSpec.required_skills || []),
    ...(jobSpec.preferred_skills || []),
  ];

  return canonicalizeSkills(Array.from(candidateSkills), requestedSkills);
}

function fallbackParseResume(resume: RequestResume, jobSpec: RequestJobSpec): ParsedLLMCandidate {
  return {
    candidate_id: resume.candidate_id,
    technical_skills: extractSkills(resume.text, jobSpec),
    total_years_experience: extractExperienceYears(resume.text),
    highest_degree: normalizeEducation(resume.text),
    experience_summary: [],
  };
}

function rankProfilesForJobSpec(
  candidateProfiles: CandidateProfile[],
  jobSpec: RequestJobSpec,
  jobSpecEmbedding: number[]
): ScoredCandidate[] {
  const scoringJobSpec: JobSpec = {
    required_skills: jobSpec.required_skills || [],
    preferred_skills: jobSpec.preferred_skills || [],
    min_experience_years: jobSpec.min_experience_years || 0,
    embedding: jobSpecEmbedding,
  };
  const weights: Weights = jobSpec.weights || {
    skills: 0.5,
    exp: 0.3,
    edu: 0.2,
  };
  return rankCandidates(candidateProfiles, scoringJobSpec, weights);
}

export async function POST(req: Request) {
  const apiKey =
    req.headers.get('x-gemini-api-key') ||
    req.headers.get('X-Gemini-Api-Key') ||
    process.env.GEMINI_API_KEY;

  let body: EvaluateRequestBody;
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  const { jobSpec, resumes, cachedCandidates } = body;
  const hasResumes = Array.isArray(resumes);
  const hasCached =
    Array.isArray(cachedCandidates) &&
    cachedCandidates.length > 0 &&
    cachedCandidates.every((c) => Array.isArray(c.embedding) && c.embedding.length > 0);

  if (!jobSpec || (!hasResumes && !hasCached)) {
    return Response.json(
      { error: 'Missing required fields: jobSpec and resumes array' },
      { status: 400 }
    );
  }

  if (hasResumes && resumes!.length === 0 && !hasCached) {
    const emptyProvider: EmbeddingProvider = new TFIDFProvider(384);
    const [emptyJobSpecEmbedding] = await emptyProvider.embed([jobSpec.title || 'Job Specification']);
    return Response.json({
      candidates: [],
      jobSpecEmbedding: emptyJobSpecEmbedding || [],
    });
  }

  const hasApiKey = Boolean(apiKey && apiKey.trim().length > 0);

  // 1. INFERENCE CACHING PATH:
  // If valid cached candidate profiles and embeddings were provided (e.g. user updated only Job Spec),
  // we skip parsing resumes with Gemini Flash and skip re-embedding resumes entirely.
  // We only embed the single updated Job Spec and recompute rankings instantly.
  if (hasCached) {
    const jobSpecText = formatJobSpecText(jobSpec);
    const dim = cachedCandidates![0]?.embedding?.length || (hasApiKey ? 768 : 384);

    // Only embed the updated job spec; candidate embeddings are preserved from cache
    const embeddings = await computeEmbeddingsWithFallback(
      [jobSpecText],
      dim,
      hasApiKey ? apiKey!.trim() : undefined
    );
    const jobSpecEmbedding = embeddings[0] || [];

    const candidateProfiles: CandidateProfile[] = cachedCandidates!.map((c) => ({
      candidate_id: c.candidate_id,
      technical_skills: c.technical_skills || [],
      total_years_experience: c.total_years_experience ?? 0,
      highest_degree: normalizeEducation(c.highest_degree),
      embedding: c.embedding,
      project_sentences: c.project_sentences || [],
      experience_summary: c.experience_summary,
    }));

    const rankedCandidates = rankProfilesForJobSpec(
      candidateProfiles,
      jobSpec,
      jobSpecEmbedding
    );

    return Response.json({
      candidates: rankedCandidates,
      jobSpecEmbedding,
      cached: true,
    });
  }

  // 2. FULL EXTRACTION & EMBEDDING PATH
  const activeResumes = resumes!;
  const parsedCandidatesMap = new Map<string, ParsedLLMCandidate>();

  if (hasApiKey) {
    const genAI = new GoogleGenerativeAI(apiKey!.trim());

    // 1. Batched Structured Parsing via Gemini Flash (in parallel across batches)
    const batches: RequestResume[][] = [];
    for (let i = 0; i < activeResumes.length; i += BATCH_SIZE) {
      batches.push(activeResumes.slice(i, i + BATCH_SIZE));
    }

    await Promise.all(
      batches.map(async (batch) => {
        try {
          const flashModel = genAI.getGenerativeModel({
            model: 'gemini-2.0-flash',
            generationConfig: {
              responseMimeType: 'application/json',
              responseSchema: {
                type: SchemaType.ARRAY,
                items: {
                  type: SchemaType.OBJECT,
                  properties: {
                    candidate_id: { type: SchemaType.STRING },
                    technical_skills: {
                      type: SchemaType.ARRAY,
                      items: { type: SchemaType.STRING },
                    },
                    total_years_experience: { type: SchemaType.NUMBER },
                    highest_degree: {
                      type: SchemaType.STRING,
                      enum: ['PhD', 'Masters', 'Bachelors', 'Diploma', 'None'],
                    },
                    experience_summary: {
                      type: SchemaType.ARRAY,
                      items: {
                        type: SchemaType.OBJECT,
                        properties: {
                          title: { type: SchemaType.STRING },
                          duration_years: { type: SchemaType.NUMBER },
                          technologies: {
                            type: SchemaType.ARRAY,
                            items: { type: SchemaType.STRING },
                          },
                        },
                      },
                    },
                  },
                  required: [
                    'candidate_id',
                    'technical_skills',
                    'total_years_experience',
                    'highest_degree',
                  ],
                },
              },
            },
          });

          const prompt = `You are an expert technical resume parser. Analyze each candidate resume and extract their technical skills, total years of experience, highest educational degree ("PhD", "Masters", "Bachelors", "Diploma", or "None"), and experience summary according to the schema.

Candidate Resumes:
${batch
  .map(
    (r) =>
      `--- CANDIDATE ID: ${r.candidate_id} ---\n${r.text}`
  )
  .join('\n\n')}`;

          const result = await flashModel.generateContent(prompt);
          const text = result.response.text();
          const parsedArray: ParsedLLMCandidate[] = JSON.parse(text);

          for (const item of parsedArray) {
            if (item && item.candidate_id) {
              parsedCandidatesMap.set(item.candidate_id, item);
            }
          }

          // Check if any resume in batch was missed by LLM; fallback if so
          for (const r of batch) {
            if (!parsedCandidatesMap.has(r.candidate_id)) {
              parsedCandidatesMap.set(r.candidate_id, fallbackParseResume(r, jobSpec));
            }
          }
        } catch {
          // Graceful fallback to deterministic regex extraction for the batch
          for (const r of batch) {
            parsedCandidatesMap.set(r.candidate_id, fallbackParseResume(r, jobSpec));
          }
        }
      })
    );
  } else {
    // Graceful deterministic parsing when GEMINI_API_KEY is not configured
    for (const r of activeResumes) {
      parsedCandidatesMap.set(r.candidate_id, fallbackParseResume(r, jobSpec));
    }
  }

  // 2. Vector Embeddings via EmbeddingProvider interface
  const jdText = formatJobSpecText(jobSpec);
  const allTexts = [jdText, ...activeResumes.map((r) => r.text || ' ')];
  const allEmbeddings = await computeEmbeddingsWithFallback(
    allTexts,
    384,
    hasApiKey ? apiKey!.trim() : undefined
  );

  const jobSpecEmbedding = allEmbeddings[0] || [];
  const candidateEmbeddings = allEmbeddings.slice(1);

  // 3. Construct Candidate Profiles with semantic experience context
  const candidateProfiles: CandidateProfile[] = activeResumes.map((resume, idx) => {
    const parsed =
      parsedCandidatesMap.get(resume.candidate_id) ||
      fallbackParseResume(resume, jobSpec);

    const projectSentences = resume.text
      ? resume.text
          .split(/[\n•*\u2022]+/)
          .map((s) => s.trim())
          .filter((s) => s.length > 10)
      : [];

    const requestedSkills = [
      ...(jobSpec.required_skills || []),
      ...(jobSpec.preferred_skills || []),
    ];

    return {
      candidate_id: resume.candidate_id,
      technical_skills: canonicalizeSkills(parsed.technical_skills || [], requestedSkills),
      total_years_experience: parsed.total_years_experience ?? 0,
      highest_degree: normalizeEducation(parsed.highest_degree),
      embedding: candidateEmbeddings[idx] || [],
      project_sentences: projectSentences,
      experience_summary: parsed.experience_summary,
    };
  });

  // 4. Multi-Factor Scoring & Ranking
  const rankedCandidates = rankProfilesForJobSpec(
    candidateProfiles,
    jobSpec,
    jobSpecEmbedding
  );

  return Response.json({
    candidates: rankedCandidates,
    jobSpecEmbedding,
  });
}
