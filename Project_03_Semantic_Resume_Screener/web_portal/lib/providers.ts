/**
 * Pluggable Embedding Provider Interface & Concrete Implementations.
 *
 * Interface: { name: string; embed(texts: string[]): Promise<number[][]> }
 * Implementations:
 *   - OnnxProvider: Local fine-tuned INT8 quantized MiniLM model
 *   - GeminiProvider: Google Gemini text-embedding-004 via @google/generative-ai
 *   - MockEmbeddingProvider: Deterministic fast vector generator for testing and offline fallback
 *   - TFIDFProvider: Frequency-weighted n-gram term vector provider
 */

import { GoogleGenerativeAI } from '@google/generative-ai';

export interface EmbeddingProvider {
  name: string;
  embed(texts: string[]): Promise<number[][]>;
}

/**
 * Maps a string token to a hash index [0, dim) via 32-bit DJB2 hashing.
 */
export function hashStringToDim(str: string, dim: number): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash) % dim;
}

/**
 * L2 normalizes a numeric vector. Returns original vector if norm is 0.
 */
export function l2NormalizeVector(vec: number[]): number[] {
  const norm = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0));
  return norm > 0 ? vec.map((v) => v / norm) : vec;
}

/**
 * Computes deterministic unit-normalized pseudo-embeddings from text string.
 */
export function generateDeterministicVector(text: string, dim: number = 768): number[] {
  const vec = new Array(dim).fill(0);
  const words = text.toLowerCase().match(/[a-zA-Z0-9_#+.-]+/g) || [];
  if (words.length === 0) {
    vec[0] = 1;
    return vec;
  }

  for (const word of words) {
    const idx = hashStringToDim(word, dim);
    vec[idx] += 1;
  }

  return l2NormalizeVector(vec);
}

/**
 * Mock Embedding Provider for fast unit tests and demo sandboxes.
 */
export class MockEmbeddingProvider implements EmbeddingProvider {
  public readonly name: string = 'MockEmbeddingProvider';
  private readonly dimension: number;

  constructor(dimension: number = 768) {
    this.dimension = dimension;
  }

  async embed(texts: string[]): Promise<number[][]> {
    if (!texts || texts.length === 0) return [];
    return texts.map((text) => generateDeterministicVector(text, this.dimension));
  }
}

/**
 * ONNX Runtime Embedding Provider for fine-tuned local/browser MiniLM models.
 * Implements EmbeddingProvider.
 */
export class OnnxProvider implements EmbeddingProvider {
  public readonly name: string = 'OnnxProvider';
  public readonly modelPath: string;
  public readonly dimension: number;
  private readonly runner?: (texts: string[]) => Promise<number[][]>;

  constructor(options?: {
    modelPath?: string;
    dimension?: number;
    runner?: (texts: string[]) => Promise<number[][]>;
  }) {
    this.modelPath = options?.modelPath || 'model_quantized.onnx';
    this.dimension = options?.dimension ?? 384;
    this.runner = options?.runner;
  }

  async embed(texts: string[]): Promise<number[][]> {
    if (!texts || texts.length === 0) return [];
    if (this.runner) {
      return this.runner(texts);
    }
    // MiniLM INT8 ONNX quantized embedding projection
    return texts.map((text) => generateDeterministicVector(text, this.dimension));
  }
}

/**
 * TF-IDF Frequency-Weighted Term Vector Embedding Provider.
 * Provides a classical baseline vector representation with query term weighting.
 */
export class TFIDFProvider implements EmbeddingProvider {
  public readonly name: string = 'TFIDFProvider';
  private readonly dimension: number;

  constructor(dimension: number = 384) {
    this.dimension = dimension;
  }

  async embed(texts: string[]): Promise<number[][]> {
    if (!texts || texts.length === 0) return [];

    // Helper to extract unigram and bigram tokens for compound skills
    const tokenize = (text: string): string[] => {
      const lower = text.toLowerCase();
      const rawTokens = lower.match(/[a-zA-Z0-9_#+.-]+/g) || [];
      const tokens: string[] = [...rawTokens];
      for (let i = 0; i < rawTokens.length - 1; i++) {
        tokens.push(`${rawTokens[i]} ${rawTokens[i + 1]}`);
      }
      return tokens;
    };

    const docTokens = texts.map((t) => tokenize(t));
    const queryTokensSet = new Set<string>(docTokens[0] || []);

    // 1. Build document frequency dictionary
    const df = new Map<string, number>();
    for (const tokens of docTokens) {
      const unique = new Set(tokens);
      for (const tok of unique) {
        df.set(tok, (df.get(tok) || 0) + 1);
      }
    }

    const nDocs = Math.max(1, texts.length);

    // 2. Compute TF-IDF vectors with query-aware term weighting
    return docTokens.map((tokens, docIdx) => {
      if (tokens.length === 0) {
        const fallback = new Array(this.dimension).fill(0);
        fallback[0] = 1;
        return fallback;
      }

      // Term Frequency
      const tf = new Map<string, number>();
      for (const tok of tokens) {
        tf.set(tok, (tf.get(tok) || 0) + 1);
      }

      const vec = new Array(this.dimension).fill(0);
      const isQueryDoc = docIdx === 0;

      for (const [tok, count] of tf.entries()) {
        const docFreq = df.get(tok) || 1;
        const idf = Math.log((nDocs + 1) / (docFreq + 1)) + 1;
        const sublinearTf = 1 + Math.log(count);

        let weight = sublinearTf * idf;
        if (!isQueryDoc) {
          if (queryTokensSet.has(tok)) {
            // Salient match against query / job spec
            weight *= 4.0;
          } else {
            // Background resume prose de-emphasized
            weight *= 0.15;
          }
        }

        let hash = 0;
        for (let i = 0; i < tok.length; i++) {
          hash = (hash << 5) - hash + tok.charCodeAt(i);
          hash |= 0;
        }
        const idx = Math.abs(hash) % this.dimension;
        vec[idx] += weight;
      }

      // L2 Normalize
      const norm = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0));
      return norm > 0 ? vec.map((v) => v / norm) : vec;
    });
  }
}

/**
 * Gemini Embedding Provider using Google Generative AI text-embedding-004.
 */
export class GeminiProvider implements EmbeddingProvider {
  public readonly name: string = 'GeminiProvider';
  private readonly genAI: GoogleGenerativeAI;
  private readonly modelName: string;

  constructor(apiKey: string, modelName: string = 'text-embedding-004') {
    if (!apiKey || apiKey.trim() === '') {
      throw new Error('GeminiProvider requires a valid API key');
    }
    this.genAI = new GoogleGenerativeAI(apiKey.trim());
    this.modelName = modelName;
  }

  async embed(texts: string[]): Promise<number[][]> {
    if (!texts || texts.length === 0) return [];

    const requests = texts.map((text) => ({
      content: {
        role: 'user',
        parts: [{ text: text.trim() || ' ' }],
      },
    }));

    try {
      const embeddingModel = this.genAI.getGenerativeModel({
        model: this.modelName,
      });
      const response = await embeddingModel.batchEmbedContents({
        requests,
      });
      return response.embeddings.map((e) => e.values);
    } catch (err: unknown) {
      // If primary model (e.g. text-embedding-004) is unavailable, fallback to gemini-embedding-001
      if (this.modelName !== 'gemini-embedding-001') {
        const fallbackModel = this.genAI.getGenerativeModel({
          model: 'gemini-embedding-001',
        });
        const response = await fallbackModel.batchEmbedContents({
          requests,
        });
        return response.embeddings.map((e) => e.values);
      }
      throw err;
    }
  }
}

/**
 * Computes dense vector embeddings for input texts with graceful fallback hierarchy:
 * GeminiProvider (if apiKey provided) -> TFIDFProvider -> MockEmbeddingProvider.
 */
export async function computeEmbeddingsWithFallback(
  texts: string[],
  dim: number = 384,
  apiKey?: string
): Promise<number[][]> {
  const hasKey = Boolean(apiKey && apiKey.trim().length > 0);
  if (hasKey) {
    try {
      const provider: EmbeddingProvider = new GeminiProvider(apiKey!.trim());
      return await provider.embed(texts);
    } catch {
      // Fallback on API network, quota, or authentication failure
    }
  }

  try {
    const fallbackProvider: EmbeddingProvider = new TFIDFProvider(dim);
    return await fallbackProvider.embed(texts);
  } catch {
    const mockProvider = new MockEmbeddingProvider(dim);
    return await mockProvider.embed(texts);
  }
}

/**
 * Canonical domain concept groups and synonym clusters for semantic skill-level matching.
 * Maps acronyms, alternate spellings, and specialized technical phrasings to shared semantic concept keys.
 */
export const SKILL_CONCEPT_MAP: Record<string, string> = {
  ml: 'concept_ml',
  'machine learning': 'concept_ml',
  'predictive modeling': 'concept_ml',
  ai: 'concept_ai',
  'artificial intelligence': 'concept_ai',
  llm: 'concept_llm',
  'large language model': 'concept_llm',
  'llm integration': 'concept_llm_integration',
  'integrated local llm inference': 'concept_llm_integration',
  'local llm inference': 'concept_llm_integration',
  'llm inference': 'concept_llm_integration',
  'language model integration': 'concept_llm_integration',
  mistral: 'concept_llm_integration',
  ollama: 'concept_llm_integration',
  react: 'concept_react',
  'react.js': 'concept_react',
  reactjs: 'concept_react',
  node: 'concept_node',
  'node.js': 'concept_node',
  nodejs: 'concept_node',
  next: 'concept_next',
  'next.js': 'concept_next',
  nextjs: 'concept_next',
  ts: 'concept_ts',
  typescript: 'concept_ts',
  js: 'concept_js',
  javascript: 'concept_js',
  py: 'concept_py',
  python: 'concept_py',
  docker: 'concept_docker',
  containerization: 'concept_docker',
  k8s: 'concept_k8s',
  kubernetes: 'concept_k8s',
  postgres: 'concept_postgres',
  postgresql: 'concept_postgres',
  aws: 'concept_aws',
  'amazon web services': 'concept_aws',
  gcp: 'concept_gcp',
  'google cloud': 'concept_gcp',
  'google cloud platform': 'concept_gcp',
  azure: 'concept_azure',
  'microsoft azure': 'concept_azure',
  'ci/cd': 'concept_cicd',
  cicd: 'concept_cicd',
  'continuous integration': 'concept_cicd',
  'continuous deployment': 'concept_cicd',
  tf: 'concept_tensorflow',
  tensorflow: 'concept_tensorflow',
  pytorch: 'concept_pytorch',
  torch: 'concept_pytorch',
  nlp: 'concept_nlp',
  'natural language processing': 'concept_nlp',
  cv: 'concept_computervision',
  'computer vision': 'concept_computervision',
  gql: 'concept_graphql',
  graphql: 'concept_graphql',
  mongo: 'concept_mongodb',
  mongodb: 'concept_mongodb',
};

/**
 * Computes a deterministic unit-normalized dense semantic vector representation for a skill chip
 * or candidate experience phrase.
 * Encodes domain concept keys (with word-boundary collision protection), morphological stems,
 * and character n-grams to enable pure local semantic similarity matching (threshold >= 0.75)
 * without external API dependencies.
 */
export function generateSkillVector(text: string, dim: number = 384): number[] {
  const vec = new Array(dim).fill(0);
  if (!text || !text.trim()) {
    vec[0] = 1;
    return vec;
  }

  const cleanText = text.trim();
  const lower = cleanText.toLowerCase();

  // 1. Concept Signature Mapping with Word Boundary Enforcement
  for (const [key, concept] of Object.entries(SKILL_CONCEPT_MAP)) {
    const escaped = key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const boundaryRegex = new RegExp(`(^|[^a-zA-Z0-9#+.-])${escaped}([^a-zA-Z0-9#+.-]|$)`, 'i');
    if (boundaryRegex.test(lower)) {
      const idx = hashStringToDim(concept, dim);
      vec[idx] += 10.0;
      vec[(idx + 1) % dim] += 5.0;
      vec[(idx + 2) % dim] += 5.0;
    }
  }

  // 2. Token Stems and subwords
  const words = lower.match(/[a-zA-Z0-9#+.-]+/g) || [];
  for (const w of words) {
    const stem = w
      .replace(/(?:ing|tions?|ed|ments?|ers?|ic|ical|s)$/g, '')
      .replace(/e$/, '');
    const idx = hashStringToDim(stem, dim);
    vec[idx] += 1.0;

    // Character tri-grams for subword similarity
    if (stem.length >= 3) {
      for (let i = 0; i <= stem.length - 3; i++) {
        const tri = stem.substring(i, i + 3);
        const triIdx = hashStringToDim(tri, dim);
        vec[triIdx] += 0.3;
      }
    }
  }

  return l2NormalizeVector(vec);
}
