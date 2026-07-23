'use client';

import React, { useState, useEffect } from 'react';
import {
  X,
  FileText,
  AlignLeft,
  Award,
  CheckCircle2,
  AlertCircle,
  Sparkles,
} from 'lucide-react';
import {
  ScoredCandidate,
  RecommendationTier,
  getRecommendationTier,
  getRecommendation,
  getSkillDisplayName,
} from '../../lib/scoring';
import { PdfJsCanvasViewer } from './PdfJsCanvasViewer';
import { TriangularRadarChart } from './TriangularRadarChart';

export { PdfJsCanvasViewer, TriangularRadarChart, getSkillDisplayName };
export type { RecommendationTier };
export { getRecommendationTier, getRecommendation };

interface SkillSubcategoryProps {
  label: string;
  skills: string[];
  theme: 'emerald' | 'teal';
}

function SkillSubcategory({ label, skills, theme }: SkillSubcategoryProps) {
  const isEmerald = theme === 'emerald';
  const containerClass = isEmerald
    ? 'border-emerald-200/80 bg-emerald-50/50'
    : 'border-teal-200/80 bg-teal-50/50';
  const headerClass = isEmerald ? 'text-emerald-800' : 'text-teal-800';
  const countClass = isEmerald ? 'text-emerald-700' : 'text-teal-700';
  const chipClass = isEmerald
    ? 'border-[#86efac] bg-[#dcfce7] text-[#166534]'
    : 'border-teal-200 bg-teal-100/70 text-teal-900';

  return (
    <div className={`rounded-lg border p-2.5 space-y-1.5 ${containerClass}`}>
      <div className={`flex items-center justify-between text-[11px] font-semibold uppercase tracking-wider ${headerClass}`}>
        <span>{label}</span>
        <span className={`font-medium text-[11px] ${countClass}`}>
          ({skills.length})
        </span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {skills.length > 0 ? (
          skills.map((skill, idx) => (
            <span
              key={idx}
              title={skill}
              className={`inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium ${chipClass}`}
            >
              {getSkillDisplayName(skill)}
            </span>
          ))
        ) : (
          <span className="text-xs text-[#5c5f77] italic">None</span>
        )}
      </div>
    </div>
  );
}

export interface CandidateDetailPopupProps {
  /** The candidate profile with computed scores, or null if popup is closed */
  candidate: ScoredCandidate | null;
  /** Optional URL or blob object URL to the original PDF file */
  pdfUrl?: string | null;
  /** Optional anonymized / PII-scrubbed resume text */
  anonymizedText?: string | null;
  /** Callback to close the popup */
  onClose: () => void;
  /** Optional initial tab override (defaults to 'pdf') */
  initialTab?: 'pdf' | 'text';
}

/**
 * Handles keyboard escape dismissal for the modal/popup.
 */
export function handleModalEscape(
  event: KeyboardEvent | { key: string },
  onClose: () => void
): void {
  if (event.key === 'Escape' || event.key === 'Esc') {
    onClose();
  }
}

/**
 * CandidateDetailPopup provides an inspection view for a screened applicant:
 * - Tabbed original PDF viewer vs anonymized scrubbed plain text.
 * - Score breakdown with circular score indicator, recommendation label,
 *   Skill Audit (matched, missing, bonus), and triangular radar chart.
 */
export function CandidateDetailPopup({
  candidate,
  pdfUrl,
  anonymizedText,
  onClose,
  initialTab = 'pdf',
}: CandidateDetailPopupProps) {
  const [activeTab, setActiveTab] = useState<'pdf' | 'text'>(initialTab);
  const [mobileSection, setMobileSection] = useState<'fit' | 'document'>('fit');

  // Sync initialTab when props change
  useEffect(() => {
    setActiveTab(initialTab);
  }, [initialTab]);

  // Handle Escape key listener
  useEffect(() => {
    if (!candidate) return;

    const onKeyDown = (e: KeyboardEvent) => {
      handleModalEscape(e, onClose);
    };

    window.addEventListener('keydown', onKeyDown);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [candidate, onClose]);

  // Lock body scroll on mobile/desktop while popup is open to prevent background scroll bleed
  useEffect(() => {
    if (!candidate || typeof document === 'undefined') return;

    const originalOverflow = document.body.style.overflow;
    const originalOverscroll = document.body.style.overscrollBehavior;

    document.body.style.overflow = 'hidden';
    document.body.style.overscrollBehavior = 'contain';

    return () => {
      document.body.style.overflow = originalOverflow;
      document.body.style.overscrollBehavior = originalOverscroll;
    };
  }, [candidate]);

  if (!candidate) {
    return null;
  }

  // Strictly typed props resolution (no loose record casts)
  const resolvedPdfUrl = pdfUrl ?? null;
  const resolvedText = anonymizedText ?? '';

  const recommendation = getRecommendationTier(candidate.composite_score);

  // Circular progress calculations
  const radius = 38;
  const circumference = 2 * Math.PI * radius;
  const clampedScore = Math.min(100, Math.max(0, candidate.composite_score));
  const strokeDashoffset = circumference - (clampedScore / 100) * circumference;

  const matchedSkills = candidate.skill_audit?.matched || [];
  const missingSkills = candidate.skill_audit?.missing || [];
  const bonusSkills = candidate.skill_audit?.bonus || [];

  const hasSubcategories =
    candidate.skill_audit?.matched_required !== undefined ||
    candidate.skill_audit?.matched_preferred !== undefined;

  const matchedRequired = hasSubcategories
    ? (candidate.skill_audit?.matched_required ?? [])
    : matchedSkills;
  const matchedPreferred = hasSubcategories
    ? (candidate.skill_audit?.matched_preferred ?? [])
    : [];

  return (
    <div
      role="presentation"
      onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-sm p-2 sm:p-4 transition-opacity animate-in fade-in duration-150 overscroll-contain"
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="candidate-detail-title"
        onClick={(e) => e.stopPropagation()}
        className="relative flex flex-col w-full max-w-5xl max-h-[96vh] sm:max-h-[92vh] rounded-2xl border border-[#ccd0da] bg-white text-[#4c4f69] shadow-2xl focus:outline-none overflow-hidden overscroll-contain"
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-[#ccd0da] px-3.5 sm:px-6 py-3 sm:py-4 bg-[#f4f5f8] gap-2">
          <div className="flex items-center space-x-2.5 sm:space-x-3 min-w-0">
            <div className="flex h-9 w-9 sm:h-10 sm:w-10 shrink-0 items-center justify-center rounded-xl bg-[#e6e9ef] border border-[#ccd0da] text-[#7287fd]">
              <Award className="h-5 w-5" />
            </div>
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
                <h2 id="candidate-detail-title" className="text-base sm:text-lg font-bold text-[#4c4f69] truncate">
                  {candidate.candidate_id}
                </h2>
                <span
                  className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] sm:text-xs font-semibold ${recommendation.badgeClass}`}
                >
                  {recommendation.label}
                </span>
              </div>
              <p className="text-xs text-[#5c5f77] mt-0.5 truncate">
                {candidate.total_years_experience} yrs exp &bull; Degree:{' '}
                <span className="text-[#4c4f69] font-semibold">{candidate.highest_degree}</span>
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            aria-label="Close modal"
            className="rounded-lg p-1.5 text-[#5c5f77] transition-colors hover:bg-[#e6e9ef] hover:text-[#4c4f69] focus:outline-none focus:ring-2 focus:ring-[#7287fd] shrink-0"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Mobile View Selector (<lg screens only) */}
        <div className="lg:hidden flex border-b border-[#ccd0da] bg-[#eff1f5] p-1.5 gap-1.5 shrink-0">
          <button
            type="button"
            onClick={() => setMobileSection('fit')}
            className={`flex-1 py-1.5 px-3 text-xs font-semibold rounded-lg text-center transition-all ${
              mobileSection === 'fit'
                ? 'bg-white text-[#7287fd] shadow-xs border border-[#ccd0da]'
                : 'text-[#5c5f77] hover:text-[#4c4f69]'
            }`}
          >
            Analysis
          </button>
          <button
            type="button"
            onClick={() => setMobileSection('document')}
            className={`flex-1 py-1.5 px-3 text-xs font-semibold rounded-lg text-center transition-all ${
              mobileSection === 'document'
                ? 'bg-white text-[#7287fd] shadow-xs border border-[#ccd0da]'
                : 'text-[#5c5f77] hover:text-[#4c4f69]'
            }`}
          >
            Resume
          </button>
        </div>

        {/* Modal Body: Split into Left (Document) and Right (Score breakdown) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 flex-1 overflow-y-auto divide-y lg:divide-y-0 lg:divide-x divide-[#ccd0da] min-w-0 overscroll-contain">
          {/* Left Column: Tabbed Document View */}
          <div
            className={`lg:col-span-7 flex flex-col p-3 sm:p-5 space-y-3 sm:space-y-4 bg-white min-w-0 max-w-full overflow-hidden ${
              mobileSection === 'document' ? 'flex' : 'hidden lg:flex'
            }`}
          >
            {/* Tabs Selector */}
            <div className="flex items-center space-x-2 border-b border-[#ccd0da] pb-2.5">
              <button
                type="button"
                onClick={() => setActiveTab('pdf')}
                className={`inline-flex items-center space-x-2 rounded-lg px-3 py-1.5 text-xs font-semibold transition-all ${
                  activeTab === 'pdf'
                    ? 'bg-white text-[#7287fd] shadow-sm border border-[#ccd0da]'
                    : 'text-[#5c5f77] hover:text-[#4c4f69] hover:bg-[#eff1f5]'
                }`}
              >
                <FileText className="h-4 w-4" />
                <span>Original PDF</span>
              </button>

              <button
                type="button"
                onClick={() => setActiveTab('text')}
                className={`inline-flex items-center space-x-2 rounded-lg px-3 py-1.5 text-xs font-semibold transition-all ${
                  activeTab === 'text'
                    ? 'bg-white text-[#7287fd] shadow-sm border border-[#ccd0da]'
                    : 'text-[#5c5f77] hover:text-[#4c4f69] hover:bg-[#eff1f5]'
                }`}
              >
                <AlignLeft className="h-4 w-4" />
                <span>Anonymized Text</span>
              </button>
            </div>

            {/* Tab Viewport */}
            <div className="flex-1 min-h-[320px] sm:min-h-[460px] flex flex-col">
              {activeTab === 'pdf' ? (
                resolvedPdfUrl ? (
                  <PdfJsCanvasViewer
                    pdfUrl={resolvedPdfUrl}
                    candidateId={candidate.candidate_id}
                  />
                ) : (
                  <div className="flex flex-1 min-h-[320px] sm:min-h-[460px] flex-col items-center justify-center rounded-xl border border-dashed border-[#ccd0da] bg-[#f4f5f8] p-8 text-center">
                    <div className="flex h-12 w-12 items-center justify-center rounded-full bg-white border border-[#ccd0da] mb-3 text-[#5c5f77] shadow-sm">
                      <FileText className="h-6 w-6" />
                    </div>
                    <p className="text-sm font-semibold text-[#4c4f69]">PDF not available</p>
                    <p className="text-xs text-[#5c5f77] mt-1 max-w-xs">
                      The original resume file object is not stored in memory. You can view the extracted content under the Anonymized Text tab.
                    </p>
                  </div>
                )
              ) : (
                <div className="flex flex-1 min-h-[320px] sm:min-h-[460px] flex-col rounded-xl border border-[#ccd0da] bg-[#eff1f5] p-3 sm:p-4">
                  <div className="flex items-center justify-between pb-2 border-b border-[#ccd0da] mb-2">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-[#5c5f77]">
                      Resume Text Content
                    </span>
                    <span className="text-[11px] text-[#5c5f77] bg-white px-2 py-0.5 rounded border border-[#ccd0da]">
                      Plain text
                    </span>
                  </div>
                  <pre className="flex-1 overflow-y-auto font-mono text-xs text-[#4c4f69] whitespace-pre-wrap leading-relaxed min-h-[260px] sm:min-h-[400px]">
                    {resolvedText || 'No text content available.'}
                  </pre>
                </div>
              )}
            </div>
          </div>

          {/* Right Column: Persistent Score Breakdown Panel */}
          <div
            className={`lg:col-span-5 flex flex-col p-3.5 sm:p-6 space-y-4 sm:space-y-6 bg-[#f4f5f8] min-w-0 max-w-full lg:overflow-y-auto overscroll-contain ${
              mobileSection === 'fit' ? 'flex' : 'hidden lg:flex'
            }`}
          >
            {/* Top Score Summary Banner */}
            <div className="rounded-xl border border-[#ccd0da] bg-white p-4 flex items-center justify-between shadow-sm">
              <div className="space-y-1">
                <span className="text-xs font-semibold uppercase tracking-wider text-[#5c5f77]">
                  Composite Score
                </span>
                <div className="flex items-center space-x-2">
                  <span
                    className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold ${recommendation.badgeClass}`}
                  >
                    {recommendation.label}
                  </span>
                </div>
                <p className="text-[11px] text-[#5c5f77] pt-1">
                  Weighted sum of Skills, Experience & Education
                </p>
              </div>

              {/* Circular Progress Ring */}
              <div className="relative flex items-center justify-center">
                <svg className="w-20 h-20 -rotate-90 transform" viewBox="0 0 100 100">
                  <circle
                    cx="50"
                    cy="50"
                    r={radius}
                    stroke="#e6e9ef"
                    strokeWidth="8"
                    fill="transparent"
                  />
                  <circle
                    cx="50"
                    cy="50"
                    r={radius}
                    stroke={recommendation.ringColor}
                    strokeWidth="8"
                    fill="transparent"
                    strokeLinecap="round"
                    strokeDasharray={circumference}
                    strokeDashoffset={strokeDashoffset}
                    className="transition-all duration-500 ease-out"
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                  <span className="text-lg font-bold text-[#4c4f69]">
                    {candidate.composite_score.toFixed(1)}
                  </span>
                </div>
              </div>
            </div>

            {/* Radar Chart Section */}
            <div className="rounded-xl border border-[#ccd0da] bg-white p-4 space-y-2 shadow-sm">
              <div className="flex items-center justify-between border-b border-[#ccd0da] pb-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-[#5c5f77]">
                  Competency Radar
                </span>
                <span className="text-[11px] text-[#5c5f77]">3-Axis Score</span>
              </div>
              <TriangularRadarChart
                skills={candidate.s_skills ?? 0}
                experience={candidate.s_exp ?? 0}
                education={candidate.s_edu ?? 0}
              />
            </div>

            {/* Skill Audit Sections */}
            <div className="rounded-xl border border-[#ccd0da] bg-white p-4 space-y-4 shadow-sm">
              <div className="flex items-center justify-between border-b border-[#ccd0da] pb-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-[#5c5f77]">
                  Skill Audit
                </span>
                <span className="text-[11px] text-[#5c5f77]">
                  {matchedSkills.length} matched / {missingSkills.length} missing
                </span>
              </div>

              {/* Matched Skills */}
              <div className="space-y-2">
                <div className="flex items-center space-x-1.5 text-xs font-medium text-emerald-700">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  <span>Matched Skills ({matchedSkills.length})</span>
                </div>

                {hasSubcategories ? (
                  <div className="space-y-2">
                    <SkillSubcategory
                      label="Required Skills"
                      skills={matchedRequired}
                      theme="emerald"
                    />
                    <SkillSubcategory
                      label="Preferred Skills"
                      skills={matchedPreferred}
                      theme="teal"
                    />
                  </div>
                ) : matchedSkills.length > 0 ? (
                  <div className="flex flex-wrap gap-1.5">
                    {matchedSkills.map((skill, idx) => (
                      <span
                        key={idx}
                        title={skill}
                        className="inline-flex items-center rounded-md border border-[#86efac] bg-[#dcfce7] px-2 py-0.5 text-xs font-medium text-[#166534]"
                      >
                        {getSkillDisplayName(skill)}
                      </span>
                    ))}
                  </div>
                ) : (
                  <div className="flex flex-wrap gap-1.5">
                    <span className="text-xs text-[#5c5f77] italic">None</span>
                  </div>
                )}
              </div>

              {/* Missing Skills */}
              <div className="space-y-1.5">
                <div className="flex items-center space-x-1.5 text-xs font-medium text-rose-700">
                  <AlertCircle className="h-3.5 w-3.5" />
                  <span>Missing Skills ({missingSkills.length})</span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {missingSkills.length > 0 ? (
                    missingSkills.map((skill, idx) => (
                      <span
                        key={idx}
                        title={skill}
                        className="inline-flex items-center rounded-md border border-[#fecaca] bg-[#fee2e2] px-2 py-0.5 text-xs font-medium text-[#991b1b]"
                      >
                        {getSkillDisplayName(skill)}
                      </span>
                    ))
                  ) : (
                    <span className="text-xs text-[#5c5f77] italic">None</span>
                  )}
                </div>
              </div>

              {/* Bonus Skills */}
              <div className="space-y-1.5">
                <div className="flex items-center space-x-1.5 text-xs font-medium text-[#1e66f5]">
                  <Sparkles className="h-3.5 w-3.5" />
                  <span>Bonus Skills ({bonusSkills.length})</span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {bonusSkills.length > 0 ? (
                    bonusSkills.map((skill, idx) => (
                      <span
                        key={idx}
                        title={skill}
                        className="inline-flex items-center rounded-md border border-[#1e66f5]/30 bg-[#1e66f5]/15 px-2 py-0.5 text-xs font-medium text-[#1e66f5]"
                      >
                        {getSkillDisplayName(skill)}
                      </span>
                    ))
                  ) : (
                    <span className="text-xs text-[#5c5f77] italic">None</span>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="flex items-center justify-between border-t border-[#ccd0da] bg-[#f4f5f8] px-3.5 sm:px-6 py-2.5 sm:py-3 text-xs text-[#5c5f77]">
          <span className="hidden sm:inline text-[11px]">
            Press <kbd className="rounded border border-[#ccd0da] bg-white px-1.5 py-0.5 font-mono text-[10px] text-[#4c4f69] shadow-xs">Esc</kbd> or click outside to dismiss
          </span>
          <span className="sm:hidden text-[11px]">Click outside or Close</span>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-[#ccd0da] bg-white px-3.5 py-1 text-xs font-medium text-[#4c4f69] transition-colors hover:bg-[#e6e9ef] focus:outline-none focus:ring-2 focus:ring-[#7287fd] shadow-xs"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

export default CandidateDetailPopup;
