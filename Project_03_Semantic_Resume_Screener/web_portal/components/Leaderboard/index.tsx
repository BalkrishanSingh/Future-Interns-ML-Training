'use client';

import React from 'react';
import { Award, CheckCircle2, XCircle, ChevronRight, Users } from 'lucide-react';
import { ScoredCandidate, getRecommendationTier } from '../../lib/scoring';

export type { ScoredCandidate };

export interface LeaderboardProps {
  candidates: ScoredCandidate[];
  onSelect: (candidate: ScoredCandidate) => void;
  onSettingsOpen?: () => void;
}

function getRankBadge(rank: number) {
  if (rank === 1) {
    return {
      label: '#1',
      medal: '🥇',
      badgeClass: 'border-amber-400 bg-amber-50 text-amber-700 font-bold',
      rowClass: 'border-l-4 border-l-amber-400 bg-amber-500/[0.04]',
    };
  }
  if (rank === 2) {
    return {
      label: '#2',
      medal: '🥈',
      badgeClass: 'border-slate-300 bg-slate-100 text-slate-700 font-bold',
      rowClass: 'border-l-4 border-l-slate-400 bg-slate-500/[0.03]',
    };
  }
  if (rank === 3) {
    return {
      label: '#3',
      medal: '🥉',
      badgeClass: 'border-amber-600 bg-amber-100/60 text-amber-800 font-bold',
      rowClass: 'border-l-4 border-l-amber-600 bg-amber-700/[0.03]',
    };
  }
  return {
    label: `#${rank}`,
    medal: null,
    badgeClass: 'border-[#ccd0da] bg-[#eff1f5] text-[#5c5f77]',
    rowClass: 'border-l-4 border-l-transparent',
  };
}

function getScoreColor(score: number): string {
  if (score >= 80) return 'text-[#166534]';
  if (score >= 60) return 'text-[#1e66f5]';
  return 'text-[#5c5f77]';
}

export function Leaderboard({
  candidates = [],
  onSelect,
  onSettingsOpen,
}: LeaderboardProps) {
  // Sort descending by composite_score; tie-break deterministically by candidate_id
  const sorted = [...candidates].sort((a, b) => {
    if (b.composite_score !== a.composite_score) {
      return b.composite_score - a.composite_score;
    }
    return a.candidate_id.localeCompare(b.candidate_id);
  });

  const count = sorted.length;

  return (
    <div className="w-full rounded-2xl border border-[#ccd0da] bg-white shadow-sm overflow-hidden text-[#4c4f69]">
      {/* Top Action Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#ccd0da] px-4 sm:px-6 py-3.5 sm:py-4 bg-[#f4f5f8]">
        <div className="flex items-center space-x-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[#7287fd]/15 text-[#7287fd] border border-[#7287fd]/30">
            <Award className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-base font-bold text-[#4c4f69] tracking-tight">
              Leaderboard
            </h2>
            <p className="text-xs text-[#5c5f77]">
              Semantic ranking by composite fit (Skills + Experience + Education)
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-3 self-end sm:self-auto">
          {/* Candidate Count Badge */}
          <span className="inline-flex items-center space-x-1.5 rounded-full border border-[#ccd0da] bg-white px-3 py-1 text-xs font-semibold text-[#5c5f77] shadow-xs">
            <Users className="h-3.5 w-3.5 text-[#7287fd]" />
            <span>{`${count} ${count === 1 ? 'candidate' : 'candidates'}`}</span>
          </span>
        </div>
      </div>

      {/* Ranked Candidate Table */}
      <div className="w-full max-w-full overflow-x-auto block">
        <table className="min-w-[640px] w-full text-left text-sm text-[#4c4f69] border-collapse">
          <thead className="border-b border-[#ccd0da] bg-[#f4f5f8] text-xs uppercase tracking-wider text-[#5c5f77]">
            <tr>
              <th scope="col" className="w-16 px-4 py-3.5 text-center font-semibold">
                Rank
              </th>
              <th scope="col" className="px-5 py-3.5 font-semibold">
                Candidate ID
              </th>
              <th scope="col" className="px-5 py-3.5 font-semibold">
                Composite Score
              </th>
              <th scope="col" className="px-4 py-3.5 text-center font-semibold">
                Matched Skills
              </th>
              <th scope="col" className="px-4 py-3.5 text-center font-semibold">
                Missing Skills
              </th>
              <th scope="col" className="w-36 px-4 py-3.5 text-right font-semibold">
                Recommendation
              </th>
              <th scope="col" className="w-10 px-2 py-3.5">
                <span className="sr-only">Details</span>
              </th>
            </tr>
          </thead>

          <tbody className="divide-y divide-[#ccd0da]/60">
            {count === 0 ? (
              <tr>
                <td colSpan={7} className="px-6 py-12 text-center text-[#5c5f77]">
                  <div className="flex flex-col items-center justify-center space-y-2">
                    <Users className="h-8 w-8 text-[#8c8fa1]" />
                    <p className="text-sm font-semibold text-[#4c4f69]">
                      No candidates evaluated yet
                    </p>
                    <p className="text-xs text-[#5c5f77]">
                      Upload resume PDFs above and evaluate against the job spec to see rankings
                    </p>
                  </div>
                </td>
              </tr>
            ) : (
              sorted.map((candidate, idx) => {
                const rank = idx + 1;
                const { label, medal, badgeClass, rowClass } = getRankBadge(rank);
                const matchedCount = candidate.skill_audit?.matched?.length ?? 0;
                const missingCount = candidate.skill_audit?.missing?.length ?? 0;
                const formattedScore = `${candidate.composite_score.toFixed(1)}`;
                const recTier = getRecommendationTier(candidate.composite_score);
                const scoreColor = getScoreColor(candidate.composite_score);

                return (
                  <tr
                    key={candidate.candidate_id}
                    onClick={() => onSelect(candidate)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        onSelect(candidate);
                      }
                    }}
                    role="button"
                    tabIndex={0}
                    className={`cursor-pointer transition-colors hover:bg-[#eff1f5] focus:bg-[#eff1f5] focus:outline-none ${rowClass}`}
                  >
                    {/* Rank */}
                    <td className="px-4 py-3.5 text-center whitespace-nowrap">
                      <span
                        className={`inline-flex items-center justify-center rounded-lg border px-2 py-0.5 font-mono text-xs ${badgeClass}`}
                      >
                        {medal ? `${medal} ${label}` : label}
                      </span>
                    </td>

                    {/* Candidate ID */}
                    <td className="px-5 py-3.5 font-mono font-bold text-[#4c4f69] whitespace-nowrap">
                      {candidate.candidate_id}
                    </td>

                    {/* Composite Score */}
                    <td className="px-5 py-3.5 whitespace-nowrap">
                      <div className="flex items-baseline space-x-2">
                        <span className={`font-mono text-base font-bold ${scoreColor}`}>
                          {formattedScore}
                        </span>
                        <div className="h-2 w-16 overflow-hidden rounded-full bg-[#e6e9ef]">
                          <div
                            className={`h-full ${
                              candidate.composite_score >= 80
                                ? 'bg-[#40a02b]'
                                : candidate.composite_score >= 60
                                ? 'bg-[#1e66f5]'
                                : 'bg-[#e64553]'
                            }`}
                            style={{
                              width: `${Math.min(100, Math.max(0, candidate.composite_score))}%`,
                            }}
                          />
                        </div>
                      </div>
                    </td>

                    {/* Matched Skills Count */}
                    <td className="px-4 py-3.5 text-center whitespace-nowrap">
                      <span className="inline-flex items-center space-x-1 rounded-md border border-[#86efac] bg-[#dcfce7] px-2 py-0.5 font-mono text-xs font-semibold text-[#166534]">
                        <CheckCircle2 className="h-3 w-3" />
                        <span>{`${matchedCount}`}</span>
                      </span>
                    </td>

                    {/* Missing Skills Count */}
                    <td className="px-4 py-3.5 text-center whitespace-nowrap">
                      <span
                        className={`inline-flex items-center space-x-1 rounded-md border px-2 py-0.5 font-mono text-xs font-semibold ${
                          missingCount > 0
                            ? 'border-[#fecaca] bg-[#fee2e2] text-[#991b1b]'
                            : 'border-[#ccd0da] bg-[#eff1f5] text-[#5c5f77]'
                        }`}
                      >
                        <XCircle className="h-3 w-3" />
                        <span>{`${missingCount}`}</span>
                      </span>
                    </td>

                    {/* Recommendation Tier */}
                    <td className="px-4 py-3.5 text-right whitespace-nowrap text-xs">
                      <span className={`font-semibold ${scoreColor}`}>
                        {recTier.label}
                      </span>
                    </td>

                    {/* Chevron Indicator */}
                    <td className="px-2 py-3.5 text-right text-[#5c5f77] whitespace-nowrap">
                      <ChevronRight className="h-4 w-4 inline-block text-[#5c5f77]" />
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default Leaderboard;
