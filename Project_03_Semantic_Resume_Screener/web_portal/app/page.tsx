'use client';

import React, { useState, useMemo, useRef, useCallback } from 'react';
import JobSpecPanel, {
  JobSpecFormData,
  DEFAULT_JOB_SPEC,
} from '@/components/JobSpecPanel';
import ResumeUpload, {
  ExtractedResumeItem,
  ResumeFileItem,
} from '@/components/ResumeUpload';
import Leaderboard from '@/components/Leaderboard';
import CandidateDetailPopup from '@/components/CandidateDetailPopup';
import SettingsModal, { getCustomGeminiKey } from '@/components/SettingsModal';
import {
  rankCandidates,
  reRankWithWeights,
  ScoredCandidate,
  CandidateProfile,
  JobSpec,
  Weights,
} from '@/lib/scoring';
import { fetchWithRetry } from '@/lib/utils';
import { Loader2, AlertCircle, X, Settings, FileSearch } from 'lucide-react';

export interface PageProps {
  params?: Promise<Record<string, string | string[] | undefined>>;
  searchParams?: Promise<Record<string, string | string[] | undefined>>;
}

export interface ScreenerPageProps extends PageProps {
  initialJobSpec?: JobSpecFormData;
  initialResumes?: ExtractedResumeItem[];
  initialResumeFileItems?: ResumeFileItem[];
  initialCandidates?: ScoredCandidate[];
  initialJobSpecEmbedding?: number[];
}

export default function Page(props: ScreenerPageProps): React.JSX.Element;
export default function Page(props: PageProps): React.JSX.Element;
export default function Page(props: ScreenerPageProps | PageProps = {}): React.JSX.Element {
  const {
    initialJobSpec,
    initialResumes,
    initialResumeFileItems,
    initialCandidates = [],
    initialJobSpecEmbedding,
  } = (props as ScreenerPageProps) || {};

  // 1. Job specification form data
  const [jobSpec, setJobSpec] = useState<JobSpecFormData>(
    () => initialJobSpec ?? DEFAULT_JOB_SPEC
  );

  // 2. Extracted resumes from upload panel
  const [extractedResumes, setExtractedResumes] = useState<ExtractedResumeItem[]>(
    () => initialResumes ?? []
  );

  // 3. Analysis & Ranking data
  const [rankedCandidates, setRankedCandidates] = useState<ScoredCandidate[]>(
    () => initialCandidates
  );
  const [cachedCandidates, setCachedCandidates] = useState<CandidateProfile[]>(
    () => initialCandidates
  );
  const [jobSpecEmbedding, setJobSpecEmbedding] = useState<number[] | null>(
    () => initialJobSpecEmbedding ?? null
  );
  const [weights, setWeights] = useState<Weights>(
    () => initialJobSpec?.weights ?? DEFAULT_JOB_SPEC.weights
  );

  // 4. Selected candidate for Detail Popup overlay
  const [selectedCandidate, setSelectedCandidate] = useState<ScoredCandidate | null>(null);

  // 5. UI state (Loading spinner, Error banner, Settings modal)
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);

  // Maintain URL object refs for cleanup
  const pdfBlobUrlsRef = useRef<Map<string, string>>(new Map());

  // Derive resume file items if provided, or build mock stubs for existing initial extracted items
  const resolvedResumeFileItems: ResumeFileItem[] = useMemo(() => {
    if (initialResumeFileItems) return initialResumeFileItems;
    if (initialResumes && initialResumes.length > 0) {
      return initialResumes.map((r, i) => ({
        id: `init-${i}`,
        file: r.file || new File([''], `${r.candidate_id}.pdf`, { type: 'application/pdf' }),
        name: `${r.candidate_id}.pdf`,
        size: 1024,
        status: 'done',
        charCount: r.text.length,
        text: r.text,
      }));
    }
    return [];
  }, [initialResumeFileItems, initialResumes]);

  // Persistent uploaded resume items preserved across form and list interactions
  const [uploadedResumeItems, setUploadedResumeItems] = useState<ResumeFileItem[]>(
    () => resolvedResumeFileItems
  );

  // Sync uploaded items if initial props change
  React.useEffect(() => {
    if (resolvedResumeFileItems.length > 0 && uploadedResumeItems.length === 0) {
      setUploadedResumeItems(resolvedResumeFileItems);
    }
  }, [resolvedResumeFileItems, uploadedResumeItems.length]);

  // Whether analysis can be triggered (at least 1 resume extracted, not currently loading)
  const canAnalyze = extractedResumes.length > 0 && !isLoading;

  // Real-time re-ranking on weight changes: instant 0ms arithmetic using fast re-weighting
  const handleWeightsChange = useCallback(
    (newWeights: Weights) => {
      setWeights(newWeights);

      if (rankedCandidates.length > 0) {
        // Blistering fast O(N) recalculation using already-computed candidate sub-scores
        const reRanked = reRankWithWeights(rankedCandidates, newWeights);
        setRankedCandidates(reRanked);

        // Keep selected candidate updated if popup is open
        if (selectedCandidate) {
          const updatedSelected = reRanked.find(
            (c) => c.candidate_id === selectedCandidate.candidate_id
          );
          if (updatedSelected) {
            setSelectedCandidate(updatedSelected);
          }
        }
      } else if (cachedCandidates.length > 0 && jobSpecEmbedding) {
        const spec: JobSpec = {
          required_skills: jobSpec.required_skills,
          preferred_skills: jobSpec.preferred_skills,
          min_experience_years: jobSpec.min_experience_years,
          embedding: jobSpecEmbedding,
        };

        const reRanked = rankCandidates(cachedCandidates, spec, newWeights);
        setRankedCandidates(reRanked);

        // Keep selected candidate updated if popup is open
        if (selectedCandidate) {
          const updatedSelected = reRanked.find(
            (c) => c.candidate_id === selectedCandidate.candidate_id
          );
          if (updatedSelected) {
            setSelectedCandidate(updatedSelected);
          }
        }
      }
    },
    [rankedCandidates, cachedCandidates, jobSpecEmbedding, jobSpec, selectedCandidate]
  );

  // Analyze Resumes action: Orchestrates /api/evaluate call and smoothly scrolls to Leaderboard
  const handleAnalyze = async () => {
    if (extractedResumes.length === 0 || isLoading) {
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Check for user-provided custom Gemini API key in sessionStorage
      const customKey = getCustomGeminiKey();
      const headers: Record<string, string> = {
        'content-type': 'application/json',
      };
      if (customKey) {
        headers['x-gemini-api-key'] = customKey;
      }

      // Inference caching when Job Specification changes:
      // If we already have valid profiles & embeddings for all current extracted resumes,
      // pass cachedCandidates to the evaluate endpoint so it only embeds the new job spec
      const canUseCachedCandidates =
        cachedCandidates.length > 0 &&
        extractedResumes.length === cachedCandidates.length &&
        extractedResumes.every((r) => {
          const found = cachedCandidates.find((c) => c.candidate_id === r.candidate_id);
          return found && Array.isArray(found.embedding) && found.embedding.length > 0;
        });

      const payload = {
        jobSpec: {
          title: jobSpec.title,
          required_skills: jobSpec.required_skills,
          preferred_skills: jobSpec.preferred_skills,
          min_experience_years: jobSpec.min_experience_years,
          education_level: jobSpec.education_level,
          weights,
        },
        resumes: extractedResumes.map((r) => ({
          candidate_id: r.candidate_id,
          text: r.text,
        })),
        ...(canUseCachedCandidates ? { cachedCandidates } : {}),
      };

      const response = await fetchWithRetry(
        '/api/evaluate',
        {
          method: 'POST',
          headers,
          body: JSON.stringify(payload),
        },
        {
          maxRetries: 2,
          delayMs: 600,
        }
      );

      if (!response.ok) {
        let errMessage = `Evaluation failed (${response.status})`;
        try {
          const errData = await response.json();
          if (errData?.error) {
            errMessage = errData.error;
          }
        } catch {
          // Keep generic message
        }
        throw new Error(errMessage);
      }

      const data = await response.json();

      if (!data.candidates || !Array.isArray(data.candidates)) {
        throw new Error('Malformed response from evaluate API');
      }

      const receivedCandidates: ScoredCandidate[] = data.candidates;
      const embedding: number[] = data.jobSpecEmbedding || [];

      // Cache raw candidate profiles for instant slider math
      const profiles: CandidateProfile[] = receivedCandidates.map((c) => ({
        candidate_id: c.candidate_id,
        technical_skills: c.technical_skills,
        total_years_experience: c.total_years_experience,
        highest_degree: c.highest_degree,
        embedding: c.embedding,
        project_sentences: c.project_sentences,
        experience_summary: c.experience_summary,
      }));

      setCachedCandidates(profiles);
      setJobSpecEmbedding(embedding);
      setRankedCandidates(receivedCandidates);

      // Smoothly scroll down to Leaderboard results section
      if (typeof window !== 'undefined') {
        setTimeout(() => {
          const resultsElem = document.getElementById('results');
          if (resultsElem && typeof resultsElem.scrollIntoView === 'function') {
            resultsElem.scrollIntoView({ behavior: 'smooth' });
          }
        }, 100);
      }
    } catch (err: unknown) {
      let msg = err instanceof Error ? err.message : 'Unknown evaluation error';
      if (msg === 'Load failed' || msg === 'Failed to fetch') {
        msg = 'Connection to analysis service was interrupted or timed out. Please try again.';
      }
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  };

  // Select Candidate: Open Candidate Detail Popup
  const handleSelectCandidate = (candidate: ScoredCandidate) => {
    setSelectedCandidate(candidate);
  };

  // Close Detail Popup handler
  const handleCloseDetail = () => {
    setSelectedCandidate(null);
  };

  // Resolve PDF preview URL for the currently selected candidate
  const selectedResume = selectedCandidate
    ? extractedResumes.find((r) => r.candidate_id === selectedCandidate.candidate_id)
    : null;

  const selectedPdfUrl = useMemo(() => {
    if (!selectedCandidate || !selectedResume?.file) return null;
    const cid = selectedCandidate.candidate_id;
    if (pdfBlobUrlsRef.current.has(cid)) {
      return pdfBlobUrlsRef.current.get(cid)!;
    }
    if (typeof window !== 'undefined' && typeof URL !== 'undefined' && URL.createObjectURL) {
      try {
        const url = URL.createObjectURL(selectedResume.file);
        pdfBlobUrlsRef.current.set(cid, url);
        return url;
      } catch {
        return null;
      }
    }
    return null;
  }, [selectedCandidate, selectedResume]);

  return (
    <div className="min-h-screen bg-[#eff1f5] text-[#4c4f69] w-full max-w-full overflow-x-hidden relative">
      {/* Top Header / App Bar */}
      <header className="border-b border-[#ccd0da] bg-white/80 backdrop-blur-md sticky top-0 z-40 w-full max-w-full overflow-x-hidden">
        <div className="max-w-7xl mx-auto px-3.5 sm:px-6 lg:px-8 h-16 sm:h-20 flex items-center justify-between gap-2">
          <div className="flex items-center space-x-2.5 sm:space-x-3.5 min-w-0">
            <div className="h-9 w-9 sm:h-11 sm:w-11 shrink-0 rounded-2xl bg-[#7287fd]/15 border border-[#7287fd]/30 flex items-center justify-center text-[#7287fd] shadow-xs">
              <FileSearch className="h-5 w-5 sm:h-6 sm:w-6" />
            </div>
            <div className="min-w-0">
              <h1 className="text-lg sm:text-2xl font-bold tracking-tight text-[#4c4f69] truncate">
                Semantic Resume Screener
              </h1>
              <p className="text-[11px] sm:text-xs text-[#5c5f77] hidden sm:block">
                Domain-Tuned Semantic Candidate Evaluation & Explainable Match Ranking
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2 shrink-0">
            <button
              type="button"
              onClick={() => setIsSettingsOpen(true)}
              aria-label="Open Settings"
              title="Settings — API key & provider configuration"
              className="flex h-9 w-9 sm:h-10 sm:w-10 items-center justify-center rounded-xl border border-[#ccd0da] bg-white text-[#5c5f77] transition-colors hover:border-[#7287fd] hover:bg-[#eff1f5] hover:text-[#4c4f69] shadow-xs"
            >
              <Settings className="h-4 w-4 sm:h-5 sm:w-5" />
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area: Continuous Single-Page Dashboard */}
      <main className="max-w-7xl mx-auto px-3.5 sm:px-6 lg:px-8 py-5 sm:py-8 space-y-6 sm:space-y-10 w-full overflow-hidden">
        {/* Full-Page Loading Spinner Overlay during API request */}
        {isLoading && (
          <div
            role="status"
            aria-label="Analyzing resumes"
            className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-slate-900/40 backdrop-blur-sm"
          >
            <div className="flex flex-col items-center gap-4 p-8 rounded-2xl bg-white border border-[#ccd0da] shadow-2xl text-center max-w-sm">
              <Loader2 className="w-10 h-10 animate-spin text-[#7287fd]" />
              <div>
                <h3 className="text-base font-bold text-[#4c4f69]">
                  Analyzing Resumes
                </h3>
                <p className="mt-1 text-xs text-[#5c5f77]">
                  Parsing technical competencies and computing semantic match rankings...
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Inline Error Banner */}
        {error && (
          <div
            role="alert"
            className="flex items-start justify-between gap-3 rounded-2xl border border-[#fecaca] bg-[#fee2e2] p-4 text-[#991b1b]"
          >
            <div className="flex items-start gap-3">
              <AlertCircle className="h-5 w-5 shrink-0 text-[#991b1b] mt-0.5" />
              <div>
                <h4 className="text-sm font-bold text-[#991b1b]">
                  Analysis Error
                </h4>
                <p className="text-xs text-[#991b1b] mt-0.5">{error}</p>
              </div>
            </div>
            <button
              type="button"
              onClick={() => setError(null)}
              aria-label="Dismiss error"
              className="text-[#991b1b] hover:text-[#7f1d1d] transition-colors p-1"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        )}

        {/* Top Section: Side-by-side on desktop with matching heights */}
        <section aria-label="Job Specification and Resume Upload" className="grid grid-cols-1 lg:grid-cols-2 gap-6 sm:gap-8 items-stretch w-full min-w-0">
          <JobSpecPanel
            value={jobSpec}
            onChange={(newSpec) => {
              setJobSpec(newSpec);
              if (newSpec.weights) {
                handleWeightsChange(newSpec.weights);
              }
            }}
            className="h-full flex flex-col justify-between"
          />
          <ResumeUpload
            initialItems={uploadedResumeItems}
            onItemsChange={setUploadedResumeItems}
            canAnalyze={canAnalyze}
            onReady={(items) => setExtractedResumes(items)}
            onAnalyze={handleAnalyze}
            className="h-full"
          />
        </section>

        {/* Bottom Section: Continuous scroll results container */}
        <section id="results" aria-label="Evaluation Results" className="pt-4 scroll-mt-24 space-y-4 w-full max-w-full overflow-x-hidden min-w-0">
          <Leaderboard
            candidates={rankedCandidates}
            onSelect={handleSelectCandidate}
          />
        </section>

        {/* Candidate Detail Popup overlay */}
        <CandidateDetailPopup
          candidate={selectedCandidate}
          pdfUrl={selectedPdfUrl}
          anonymizedText={selectedResume?.text}
          onClose={handleCloseDetail}
        />

        {/* Settings Modal Dialog */}
        <SettingsModal
          open={isSettingsOpen}
          onClose={() => setIsSettingsOpen(false)}
          weights={weights}
          onWeightsChange={handleWeightsChange}
        />
      </main>
    </div>
  );
}
