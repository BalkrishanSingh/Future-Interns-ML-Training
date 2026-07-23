'use client';

import React, { useEffect, useState, useCallback } from 'react';
import { X, Key, Cpu, Eye, EyeOff, Trash2, Check, ShieldCheck, Sliders } from 'lucide-react';
import { Weights, redistributeWeights } from '@/lib/scoring';

/**
 * Storage key used to persist user-supplied Gemini API key in browser session.
 */
export const CUSTOM_GEMINI_KEY_STORAGE = 'custom_gemini_key';

/**
 * Utility helper to safely retrieve custom Gemini API key from sessionStorage.
 * Returns null if running in SSR context or if key is not set.
 */
export function getCustomGeminiKey(): string | null {
  if (typeof window === 'undefined') return null;
  try {
    const storage = window.sessionStorage || (typeof sessionStorage !== 'undefined' ? sessionStorage : null);
    if (!storage) return null;
    const key = storage.getItem(CUSTOM_GEMINI_KEY_STORAGE);
    return key && key.trim().length > 0 ? key.trim() : null;
  } catch {
    return null;
  }
}

export interface SettingsModalProps {
  /** Whether the modal dialog is visible */
  open: boolean;
  /** Callback invoked when the modal is closed via backdrop, escape key, or close button */
  onClose: () => void;
  /** Optional current default scoring weights */
  weights?: Weights;
  /** Callback when scoring weights are adjusted */
  onWeightsChange?: (weights: Weights) => void;
}

const DEFAULT_WEIGHTS: Weights = { skills: 0.5, exp: 0.3, edu: 0.2 };

/**
 * SettingsModal provides a configuration dialog for recruiters to:
 * - View the active Embedding Provider ("Google Gemini — text-embedding-004")
 * - Configure initial default scoring weights redistribution
 * - Supply an optional custom GEMINI_API_KEY override stored only in sessionStorage
 * - View active key status ("Server key active" or "Using your key")
 */
export function SettingsModal({
  open,
  onClose,
  weights = DEFAULT_WEIGHTS,
  onWeightsChange,
}: SettingsModalProps) {
  const [apiKey, setApiKey] = useState<string>('');
  const [showKey, setShowKey] = useState<boolean>(false);
  const [hasSavedNotice, setHasSavedNotice] = useState<boolean>(false);
  const [localWeights, setLocalWeights] = useState<Weights>(weights);

  // Sync state from sessionStorage and props when opened
  useEffect(() => {
    if (open && typeof window !== 'undefined') {
      try {
        const stored = sessionStorage.getItem(CUSTOM_GEMINI_KEY_STORAGE);
        setApiKey(stored || '');
      } catch {
        setApiKey('');
      }
      setHasSavedNotice(false);
      setLocalWeights(weights);
    }
  }, [open, weights]);

  // Handle Escape key press
  useEffect(() => {
    if (!open) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [open, onClose]);

  // Lock body scroll on mobile/desktop while settings modal is open
  useEffect(() => {
    if (!open || typeof document === 'undefined') return;

    const originalOverflow = document.body.style.overflow;
    const originalOverscroll = document.body.style.overscrollBehavior;

    document.body.style.overflow = 'hidden';
    document.body.style.overscrollBehavior = 'contain';

    return () => {
      document.body.style.overflow = originalOverflow;
      document.body.style.overscrollBehavior = originalOverscroll;
    };
  }, [open]);

  // Handle weight slider change
  const handleSliderChange = (key: keyof Weights, rawVal: number) => {
    const next = redistributeWeights(localWeights, key, rawVal);
    setLocalWeights(next);
    if (onWeightsChange) {
      onWeightsChange(next);
    }
  };

  // Auto-persist to sessionStorage when key is typed or updated
  const handleKeyChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setApiKey(value);

    if (typeof window !== 'undefined') {
      try {
        const trimmed = value.trim();
        if (trimmed.length > 0) {
          sessionStorage.setItem(CUSTOM_GEMINI_KEY_STORAGE, trimmed);
        } else {
          sessionStorage.removeItem(CUSTOM_GEMINI_KEY_STORAGE);
        }
      } catch {
        // Ignore storage exceptions
      }
    }
  };

  // Explicit save action for user confirmation
  const handleSave = useCallback(() => {
    if (typeof window !== 'undefined') {
      try {
        const trimmed = apiKey.trim();
        if (trimmed.length > 0) {
          sessionStorage.setItem(CUSTOM_GEMINI_KEY_STORAGE, trimmed);
        } else {
          sessionStorage.removeItem(CUSTOM_GEMINI_KEY_STORAGE);
        }
      } catch {
        // Ignore storage exceptions
      }
    }
    setHasSavedNotice(true);
    setTimeout(() => {
      setHasSavedNotice(false);
    }, 2000);
  }, [apiKey]);

  // Clear custom key from sessionStorage and reset input
  const handleClear = () => {
    setApiKey('');
    if (typeof window !== 'undefined') {
      try {
        sessionStorage.removeItem(CUSTOM_GEMINI_KEY_STORAGE);
      } catch {
        // Ignore storage exceptions
      }
    }
    setHasSavedNotice(false);
  };

  if (!open) {
    return null;
  }

  const hasCustomKey = apiKey.trim().length > 0;

  return (
    <div
      role="presentation"
      onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-sm p-4 transition-opacity animate-in fade-in duration-150 overscroll-contain"
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-dialog-title"
        onClick={(e) => e.stopPropagation()}
        className="relative w-full max-w-lg max-h-[92vh] overflow-y-auto rounded-2xl border border-[#ccd0da] bg-white p-6 text-[#4c4f69] shadow-2xl focus:outline-none overscroll-contain"
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-[#ccd0da] pb-4">
          <div>
            <h2 id="settings-dialog-title" className="text-lg font-bold text-[#4c4f69]">
              Settings
            </h2>
            <p className="text-xs text-[#5c5f77] mt-0.5">
              Configure provider status, credentials, and baseline scoring weights
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close settings"
            className="rounded-lg p-1.5 text-[#5c5f77] transition-colors hover:bg-[#eff1f5] hover:text-[#4c4f69] focus:outline-none focus:ring-2 focus:ring-[#7287fd]"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="mt-5 space-y-6">
          {/* Active Embedding Provider Section */}
          <div className="space-y-2">
            <label className="text-xs font-semibold uppercase tracking-wider text-[#5c5f77]">
              Active Embedding Provider
            </label>
            <div className="flex items-center justify-between rounded-xl border border-[#ccd0da] bg-[#eff1f5] px-4 py-3">
              <div className="flex items-center space-x-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[#7287fd]/15 text-[#7287fd] border border-[#7287fd]/30">
                  <Cpu className="h-5 w-5" />
                </div>
                <div>
                  <p className="text-sm font-bold text-[#4c4f69]">
                    Google Gemini — text-embedding-004
                  </p>
                  <p className="text-xs text-[#5c5f77]">
                    768-dimensional dense vector embeddings
                  </p>
                </div>
              </div>
              <span className="inline-flex items-center rounded-full border border-emerald-300 bg-[#dcfce7] px-2.5 py-0.5 text-xs font-semibold text-[#166534]">
                Connected
              </span>
            </div>
          </div>

          {/* Initial Default Weights Configuration Section */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <label className="text-xs font-semibold uppercase tracking-wider text-[#5c5f77] flex items-center gap-1.5">
                  <Sliders className="h-3.5 w-3.5 text-[#7287fd]" />
                  Initial Default Scoring Weights
                </label>
                <p className="text-xs text-[#5c5f77] mt-0.5">
                  Weights dynamically redistribute to always sum to exactly 100%
                </p>
              </div>
              <span className="text-xs font-mono font-bold px-2 py-0.5 rounded-md bg-[#7287fd]/15 text-[#7287fd] border border-[#7287fd]/30">
                Total: {Math.round((localWeights.skills + localWeights.exp + localWeights.edu) * 100)}%
              </span>
            </div>

            <div className="rounded-xl border border-[#ccd0da] bg-[#eff1f5] p-4 space-y-3.5">
              {/* Skills weight slider */}
              <div>
                <div className="flex justify-between items-center text-xs mb-1 font-medium text-[#4c4f69]">
                  <label htmlFor="settings-weight-skills">Skills Fit (W_skills)</label>
                  <span className="font-mono text-[#7287fd] font-semibold">
                    {Math.round(localWeights.skills * 100)}% ({localWeights.skills.toFixed(2)})
                  </span>
                </div>
                <input
                  id="settings-weight-skills"
                  aria-label="Skills Weight"
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={localWeights.skills}
                  onChange={(e) => handleSliderChange('skills', parseFloat(e.target.value))}
                  className="w-full h-2 bg-[#ccd0da] rounded-lg appearance-none cursor-pointer accent-[#7287fd]"
                />
              </div>

              {/* Experience weight slider */}
              <div>
                <div className="flex justify-between items-center text-xs mb-1 font-medium text-[#4c4f69]">
                  <label htmlFor="settings-weight-exp">Experience Tenure (W_exp)</label>
                  <span className="font-mono text-[#1e66f5] font-semibold">
                    {Math.round(localWeights.exp * 100)}% ({localWeights.exp.toFixed(2)})
                  </span>
                </div>
                <input
                  id="settings-weight-exp"
                  aria-label="Experience Weight"
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={localWeights.exp}
                  onChange={(e) => handleSliderChange('exp', parseFloat(e.target.value))}
                  className="w-full h-2 bg-[#ccd0da] rounded-lg appearance-none cursor-pointer accent-[#1e66f5]"
                />
              </div>

              {/* Education weight slider */}
              <div>
                <div className="flex justify-between items-center text-xs mb-1 font-medium text-[#4c4f69]">
                  <label htmlFor="settings-weight-edu">Education Degree (W_edu)</label>
                  <span className="font-mono text-[#40a02b] font-semibold">
                    {Math.round(localWeights.edu * 100)}% ({localWeights.edu.toFixed(2)})
                  </span>
                </div>
                <input
                  id="settings-weight-edu"
                  aria-label="Education Weight"
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={localWeights.edu}
                  onChange={(e) => handleSliderChange('edu', parseFloat(e.target.value))}
                  className="w-full h-2 bg-[#ccd0da] rounded-lg appearance-none cursor-pointer accent-[#40a02b]"
                />
              </div>
            </div>
          </div>

          {/* Custom API Key Section */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label
                htmlFor="custom-gemini-key"
                className="text-xs font-semibold uppercase tracking-wider text-[#5c5f77]"
              >
                Custom GEMINI_API_KEY (optional)
              </label>
              {hasSavedNotice && (
                <span className="inline-flex items-center text-xs text-[#166534] font-medium animate-in fade-in">
                  <Check className="mr-1 h-3.5 w-3.5" />
                  Saved
                </span>
              )}
            </div>

            <p className="text-xs text-[#5c5f77]">
              Provide your personal Gemini API key override for this browser session. It is stored exclusively in your browser&apos;s sessionStorage.
            </p>

            <div className="relative mt-1">
              <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-[#5c5f77]">
                <Key className="h-4 w-4" />
              </div>
              <input
                id="custom-gemini-key"
                name="custom_gemini_key"
                type={showKey ? 'text' : 'password'}
                value={apiKey}
                onChange={handleKeyChange}
                placeholder="AIzaSy..."
                autoComplete="off"
                spellCheck={false}
                className="w-full rounded-xl border border-[#ccd0da] bg-[#eff1f5] py-2 pl-9 pr-10 text-sm text-[#4c4f69] placeholder-[#8c8fa1] transition-colors focus:border-[#7287fd] focus:bg-white focus:outline-none focus:ring-1 focus:ring-[#7287fd]"
              />
              <button
                type="button"
                onClick={() => setShowKey((prev) => !prev)}
                aria-label={showKey ? 'Hide API key' : 'Show API key'}
                className="absolute inset-y-0 right-0 flex items-center pr-3 text-[#5c5f77] hover:text-[#4c4f69]"
              >
                {showKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>

            {/* Actions for API Key: Clear and Save */}
            <div className="flex items-center justify-between pt-1">
              <button
                type="button"
                onClick={handleClear}
                disabled={!apiKey}
                className="inline-flex items-center space-x-1.5 rounded-lg border border-[#ccd0da] bg-white px-3 py-1.5 text-xs font-semibold text-[#5c5f77] transition-colors hover:bg-[#eff1f5] hover:text-[#4c4f69] disabled:cursor-not-allowed disabled:opacity-40"
              >
                <Trash2 className="h-3.5 w-3.5" />
                <span>Clear</span>
              </button>

              <button
                type="button"
                onClick={handleSave}
                disabled={!apiKey}
                className="inline-flex items-center space-x-1.5 rounded-lg bg-[#7287fd] px-3.5 py-1.5 text-xs font-semibold text-white transition-colors hover:bg-[#8839ef] shadow-sm disabled:cursor-not-allowed disabled:opacity-40"
              >
                <Check className="h-3.5 w-3.5" />
                <span>Save Key</span>
              </button>
            </div>
          </div>

          {/* Status Line Section */}
          <div className="rounded-xl border border-[#ccd0da] bg-[#eff1f5] px-4 py-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-[#5c5f77]">
                Key Status
              </span>
              <div className="flex items-center space-x-2 text-xs font-medium">
                <span
                  className={`inline-block h-2 w-2 rounded-full ${
                    hasCustomKey ? 'bg-[#40a02b]' : 'bg-[#1e66f5]'
                  }`}
                />
                <span className={hasCustomKey ? 'text-[#166534] font-semibold' : 'text-[#1e66f5] font-semibold'}>
                  {hasCustomKey ? 'Using your key' : 'Server key active'}
                </span>
              </div>
            </div>
          </div>

          {/* Security Notice */}
          <div className="flex items-start space-x-2 text-[11px] text-[#5c5f77]">
            <ShieldCheck className="h-4 w-4 flex-shrink-0 text-[#7287fd] mt-0.5" />
            <span>
              Server-side API keys are protected on the serverless runtime and are never transmitted to or readable by the client browser.
            </span>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="mt-6 flex justify-end border-t border-[#ccd0da] pt-4">
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-[#ccd0da] bg-white px-4 py-2 text-xs font-semibold text-[#4c4f69] transition-colors hover:bg-[#eff1f5] focus:outline-none focus:ring-2 focus:ring-[#7287fd]"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}

export default SettingsModal;
