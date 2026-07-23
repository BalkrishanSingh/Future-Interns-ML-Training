'use client';

import React from 'react';
import { Weights, redistributeWeights } from '../../lib/scoring';

export type { Weights };
export { redistributeWeights };

export interface WeightSlidersProps {
  weights: Weights;
  onChange: (newWeights: Weights) => void;
  debounceMs?: number;
}

const ALL_KEYS: (keyof Weights)[] = ['skills', 'exp', 'edu'];

const LABELS: Record<keyof Weights, { title: string; symbol: string; description: string; accentColor: string }> = {
  skills: {
    title: 'Skills Match',
    symbol: 'W_skills',
    description: 'Cosine semantic similarity + Jaccard skill overlap',
    accentColor: 'accent-[#7287fd] text-[#7287fd]',
  },
  exp: {
    title: 'Experience Tenure',
    symbol: 'W_exp',
    description: 'Candidate years vs min required years',
    accentColor: 'accent-[#1e66f5] text-[#1e66f5]',
  },
  edu: {
    title: 'Education Degree',
    symbol: 'W_edu',
    description: 'Tiered degree score (PhD: 100, MS: 85, BS: 70, Dip: 50)',
    accentColor: 'accent-[#40a02b] text-[#40a02b]',
  },
};

const callbackTimers = new WeakMap<Function, ReturnType<typeof setTimeout>>();

export function WeightSliders({ weights, onChange, debounceMs = 0 }: WeightSlidersProps) {
  const handleSliderChange = (key: keyof Weights, rawVal: number) => {
    const updated = redistributeWeights(weights, key, rawVal);

    if (debounceMs > 0) {
      const existing = callbackTimers.get(onChange);
      if (existing) {
        clearTimeout(existing);
      }
      const timer = setTimeout(() => {
        onChange(updated);
      }, debounceMs);
      callbackTimers.set(onChange, timer);
    } else {
      onChange(updated);
    }
  };

  // Convert weights to integer percentages for display (0 - 100)
  const skillsPct = Math.round((weights.skills > 1 ? weights.skills : weights.skills * 100));
  const expPct = Math.round((weights.exp > 1 ? weights.exp : weights.exp * 100));
  const eduPct = Math.round((weights.edu > 1 ? weights.edu : weights.edu * 100));

  const percentages: Record<keyof Weights, number> = {
    skills: skillsPct,
    exp: expPct,
    edu: eduPct,
  };

  return (
    <div className="rounded-2xl border border-[#ccd0da] bg-white p-6 shadow-sm text-[#4c4f69]">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-bold uppercase tracking-wider text-[#4c4f69]">
            Composite Scoring Weights
          </h3>
          <p className="text-xs text-[#5c5f77]">
            Real-time re-ranking • Sliders redistribute proportionally to maintain 100% total
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <span className="rounded-lg border border-[#ccd0da] bg-[#eff1f5] px-2.5 py-1 font-mono text-xs font-bold text-[#4c4f69]">
            {`Total: ${skillsPct + expPct + eduPct}%`}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        {ALL_KEYS.map((key) => {
          const config = LABELS[key];
          const pct = percentages[key];

          return (
            <div
              key={key}
              className="flex flex-col justify-between rounded-xl border border-[#ccd0da] bg-[#eff1f5] p-4 transition-colors hover:border-[#7287fd]"
            >
              <div className="mb-2 flex items-center justify-between">
                <div>
                  <span className="text-xs font-bold text-[#4c4f69]">
                    {config.title}
                  </span>
                  <span className="ml-1.5 font-mono text-[11px] text-[#5c5f77]">
                    {`(${config.symbol})`}
                  </span>
                </div>
                <span className={`font-mono text-sm font-bold ${config.accentColor.split(' ')[1]}`}>
                  {`${pct}%`}
                </span>
              </div>

              <input
                type="range"
                min="0"
                max="100"
                step="1"
                value={pct}
                data-testid={`slider-${key}`}
                aria-label={`${config.title} (${config.symbol})`}
                aria-valuenow={pct}
                aria-valuemin={0}
                aria-valuemax={100}
                onChange={(e) => handleSliderChange(key, parseFloat(e.target.value) / 100)}
                className={`h-2 w-full cursor-pointer appearance-none rounded-lg bg-[#ccd0da] ${config.accentColor.split(' ')[0]} transition-opacity hover:opacity-90`}
              />

              <p className="mt-2 text-[11px] text-[#5c5f77]">
                {config.description}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default WeightSliders;
