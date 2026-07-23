'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { EduLevel, Weights, redistributeWeights } from '@/lib/scoring';
import { X, Code2, Sliders, AlertCircle, CheckCircle2 } from 'lucide-react';

export { redistributeWeights };

export interface JobSpecFormData {
  title: string;
  min_experience_years: number;
  education_level: EduLevel;
  required_skills: string[];
  preferred_skills: string[];
  weights: Weights;
}

export const JOB_SPEC_STORAGE_KEY = 'jobSpec';

export const DEFAULT_JOB_SPEC: JobSpecFormData = {
  title: '',
  min_experience_years: 3,
  education_level: 'Bachelors',
  required_skills: [],
  preferred_skills: [],
  weights: {
    skills: 0.5,
    exp: 0.3,
    edu: 0.2,
  },
};

const EDUCATION_LEVELS: EduLevel[] = ['PhD', 'Masters', 'Bachelors', 'Diploma', 'None'];

export interface JobSpecPanelProps {
  value?: JobSpecFormData;
  onChange?: (spec: JobSpecFormData) => void;
  className?: string;
}

export function JobSpecPanel({ value, onChange, className = '' }: JobSpecPanelProps) {
  // Initialize state with default or value prop
  const [formData, setFormData] = useState<JobSpecFormData>(() => {
    if (value) return value;
    if (typeof window !== 'undefined') {
      try {
        const stored = localStorage.getItem(JOB_SPEC_STORAGE_KEY);
        if (stored) {
          const parsed = JSON.parse(stored);
          return {
            ...DEFAULT_JOB_SPEC,
            ...parsed,
            weights: parsed.weights || DEFAULT_JOB_SPEC.weights,
          };
        }
      } catch {
        // Fallback to default on storage errors
      }
    }
    return DEFAULT_JOB_SPEC;
  });

  const [activeTab, setActiveTab] = useState<'form' | 'json'>('form');
  const [jsonText, setJsonText] = useState<string>('');
  const [jsonError, setJsonError] = useState<string | null>(null);

  // Skill draft inputs for enter / comma addition
  const [reqSkillDraft, setReqSkillDraft] = useState<string>('');
  const [prefSkillDraft, setPrefSkillDraft] = useState<string>('');
  const [expYearsDraft, setExpYearsDraft] = useState<string>(() =>
    formData.min_experience_years !== undefined ? String(formData.min_experience_years) : '0'
  );

  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isInternalUpdateRef = useRef<boolean>(false);
  const requiredInputRef = useRef<HTMLInputElement>(null);
  const preferredInputRef = useRef<HTMLInputElement>(null);

  // Sync state if external value prop changes
  useEffect(() => {
    if (value && !isInternalUpdateRef.current) {
      setFormData(value);
      setJsonText(JSON.stringify(value, null, 2));
      if (value.min_experience_years !== undefined) {
        setExpYearsDraft(String(value.min_experience_years));
      }
    }
    isInternalUpdateRef.current = false;
  }, [value]);

  // Keep jsonText in sync with formData whenever formData changes
  useEffect(() => {
    setJsonText(JSON.stringify(formData, null, 2));
    setJsonError(null);
  }, [formData]);

  // Propagate changes and persist to localStorage
  const updateFormData = useCallback(
    (next: JobSpecFormData) => {
      isInternalUpdateRef.current = true;
      setFormData(next);
      if (typeof window !== 'undefined') {
        try {
          localStorage.setItem(JOB_SPEC_STORAGE_KEY, JSON.stringify(next));
        } catch {
          // Ignore localStorage errors
        }
      }
      if (onChange) {
        onChange(next);
      }
    },
    [onChange]
  );

  // Add Skill Chip Handler
  const handleAddSkill = (type: 'required' | 'preferred', text: string) => {
    const rawTokens = text.split(',');
    const newSkills = rawTokens
      .map((t) => t.trim())
      .filter((t) => t.length > 0);

    if (newSkills.length === 0) return;

    if (type === 'required') {
      const existing = new Set(formData.required_skills.map((s) => s.toLowerCase()));
      const filtered = newSkills.filter((s) => !existing.has(s.toLowerCase()));
      if (filtered.length > 0) {
        updateFormData({
          ...formData,
          required_skills: [...formData.required_skills, ...filtered],
        });
      }
      setReqSkillDraft('');
    } else {
      const existing = new Set(formData.preferred_skills.map((s) => s.toLowerCase()));
      const filtered = newSkills.filter((s) => !existing.has(s.toLowerCase()));
      if (filtered.length > 0) {
        updateFormData({
          ...formData,
          preferred_skills: [...formData.preferred_skills, ...filtered],
        });
      }
      setPrefSkillDraft('');
    }
  };

  // Remove Skill Chip Handler
  const handleRemoveSkill = (type: 'required' | 'preferred', index: number) => {
    if (type === 'required') {
      const updated = formData.required_skills.filter((_, i) => i !== index);
      updateFormData({
        ...formData,
        required_skills: updated,
      });
    } else {
      const updated = formData.preferred_skills.filter((_, i) => i !== index);
      updateFormData({
        ...formData,
        preferred_skills: updated,
      });
    }
  };

  // Handle JSON Textarea Change with 300ms debounce
  const handleJsonChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const raw = e.target.value;
    setJsonText(raw);

    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    debounceTimerRef.current = setTimeout(() => {
      try {
        const parsed = JSON.parse(raw);
        if (typeof parsed !== 'object' || parsed === null) {
          throw new Error('JSON must be an object');
        }

        const validEdu = EDUCATION_LEVELS.includes(parsed.education_level)
          ? parsed.education_level
          : formData.education_level;

        const nextSpec: JobSpecFormData = {
          title: typeof parsed.title === 'string' ? parsed.title : formData.title,
          min_experience_years:
            typeof parsed.min_experience_years === 'number'
              ? parsed.min_experience_years
              : formData.min_experience_years,
          education_level: validEdu,
          required_skills: Array.isArray(parsed.required_skills)
            ? parsed.required_skills.map(String)
            : formData.required_skills,
          preferred_skills: Array.isArray(parsed.preferred_skills)
            ? parsed.preferred_skills.map(String)
            : formData.preferred_skills,
          weights:
            parsed.weights &&
            typeof parsed.weights.skills === 'number' &&
            typeof parsed.weights.exp === 'number' &&
            typeof parsed.weights.edu === 'number'
              ? parsed.weights
              : formData.weights,
        };

        setJsonError(null);
        updateFormData(nextSpec);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Invalid JSON';
        setJsonError(msg);
      }
    }, 300);
  };

  return (
    <div
      className={`rounded-2xl border border-[#ccd0da] bg-white p-4 sm:p-6 shadow-sm text-[#4c4f69] ${className}`}
    >
      {/* Top Header & Tabs Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#ccd0da] pb-3 mb-6">
        <div>
          <h2 className="text-base font-bold text-[#4c4f69] tracking-tight">
            Job Specification
          </h2>
          <p className="text-xs text-[#5c5f77]">
            Structured requirements, skills, and target criteria
          </p>
        </div>

        {/* Tab Switcher: Form vs JSON */}
        <div role="tablist" aria-label="Job Spec View Mode" className="flex items-center space-x-1.5 p-1 bg-[#eff1f5] rounded-xl border border-[#ccd0da]">
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'form'}
            onClick={() => setActiveTab('form')}
            className={`inline-flex items-center space-x-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              activeTab === 'form'
                ? 'bg-white text-[#7287fd] shadow-sm border border-[#ccd0da]'
                : 'text-[#5c5f77] hover:text-[#4c4f69]'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>Visual Form</span>
          </button>

          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'json'}
            onClick={() => setActiveTab('json')}
            className={`inline-flex items-center space-x-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              activeTab === 'json'
                ? 'bg-white text-[#7287fd] shadow-sm border border-[#ccd0da]'
                : 'text-[#5c5f77] hover:text-[#4c4f69]'
            }`}
          >
            <Code2 className="w-3.5 h-3.5" />
            <span>JSON Editor</span>
          </button>
        </div>
      </div>

      {/* Tab 1: Interactive Form View */}
      {activeTab === 'form' && (
        <div className="space-y-5">
          {/* Role Title */}
          <div>
            <label
              htmlFor="role-title"
              className="block text-xs font-semibold text-[#5c5f77] mb-1.5 uppercase tracking-wider"
            >
              Role Title
            </label>
            <input
              id="role-title"
              type="text"
              value={formData.title}
              onChange={(e) =>
                updateFormData({
                  ...formData,
                  title: e.target.value,
                })
              }
              placeholder="e.g. Senior Backend Engineer"
              className="w-full px-3.5 py-2.5 bg-[#eff1f5] border border-[#ccd0da] rounded-xl text-sm text-[#4c4f69] placeholder-[#8c8fa1] focus:outline-none focus:ring-2 focus:ring-[#7287fd]/30 focus:border-[#7287fd] focus:bg-white transition"
            />
          </div>

          {/* Experience and Education Two-Column Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Min Experience Years */}
            <div>
              <label
                htmlFor="min-experience"
                className="block text-xs font-semibold text-[#5c5f77] mb-1.5 uppercase tracking-wider"
              >
                Minimum Experience (Years)
              </label>
              <input
                id="min-experience"
                type="number"
                min="0"
                step="1"
                value={expYearsDraft}
                onChange={(e) => {
                  const val = e.target.value;
                  setExpYearsDraft(val);
                  if (val === '') {
                    updateFormData({
                      ...formData,
                      min_experience_years: 0,
                    });
                  } else {
                    const parsed = parseInt(val, 10);
                    if (!isNaN(parsed)) {
                      updateFormData({
                        ...formData,
                        min_experience_years: Math.max(0, parsed),
                      });
                    }
                  }
                }}
                onBlur={() => {
                  if (expYearsDraft === '' || isNaN(parseInt(expYearsDraft, 10))) {
                    setExpYearsDraft(String(formData.min_experience_years));
                  }
                }}
                placeholder="e.g. 3"
                className="w-full px-3.5 py-2.5 bg-[#eff1f5] border border-[#ccd0da] rounded-xl text-sm text-[#4c4f69] placeholder-[#8c8fa1] focus:outline-none focus:ring-2 focus:ring-[#7287fd]/30 focus:border-[#7287fd] focus:bg-white transition"
              />
            </div>

            {/* Education Level */}
            <div>
              <label
                htmlFor="education-level"
                className="block text-xs font-semibold text-[#5c5f77] mb-1.5 uppercase tracking-wider"
              >
                Education Level
              </label>
              <select
                id="education-level"
                value={formData.education_level}
                onChange={(e) =>
                  updateFormData({
                    ...formData,
                    education_level: e.target.value as EduLevel,
                  })
                }
                className="w-full px-3.5 py-2.5 bg-[#eff1f5] border border-[#ccd0da] rounded-xl text-sm text-[#4c4f69] focus:outline-none focus:ring-2 focus:ring-[#7287fd]/30 focus:border-[#7287fd] focus:bg-white transition"
              >
                {EDUCATION_LEVELS.map((edu) => (
                  <option key={edu} value={edu} className="bg-white text-[#4c4f69]">
                    {edu}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Required Skills Chip Input */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="block text-xs font-semibold text-[#5c5f77] uppercase tracking-wider">
                Required Skills (Skill Chips)
              </label>
              <span className="text-[11px] text-[#5c5f77] font-medium">
                {formData.required_skills.length} added
              </span>
            </div>
            <div
              onClick={() => requiredInputRef.current?.focus()}
              className="p-3 bg-[#eff1f5] border border-[#ccd0da] rounded-xl focus-within:ring-2 focus-within:ring-[#7287fd]/30 focus-within:border-[#7287fd] focus-within:bg-white transition cursor-text"
            >
              <div className="flex flex-wrap gap-1.5 mb-2">
                {formData.required_skills.map((skill, index) => (
                  <span
                    key={`${skill}-${index}`}
                    className="inline-flex items-center px-2.5 py-1 rounded-lg text-xs font-medium bg-[#7287fd]/15 text-[#7287fd] border border-[#7287fd]/30 shadow-xs"
                  >
                    <span>{skill}</span>
                    <button
                      type="button"
                      aria-label={`remove required skill ${skill}`}
                      onClick={() => handleRemoveSkill('required', index)}
                      className="ml-1.5 text-[#7287fd] hover:text-[#8839ef] rounded-full p-0.5 focus:outline-none"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </span>
                ))}
              </div>
              <input
                ref={requiredInputRef}
                type="text"
                placeholder="Add required skill (press Enter or comma)..."
                value={reqSkillDraft}
                onChange={(e) => {
                  const val = e.target.value;
                  if (val.includes(',')) {
                    handleAddSkill('required', val);
                  } else {
                    setReqSkillDraft(val);
                  }
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    handleAddSkill('required', reqSkillDraft);
                  }
                }}
                className="w-full bg-transparent text-sm text-[#4c4f69] placeholder-[#8c8fa1] focus:outline-none px-1 py-1"
              />
            </div>
          </div>

          {/* Preferred Skills Chip Input */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="block text-xs font-semibold text-[#5c5f77] uppercase tracking-wider">
                Preferred Skills (Skill Chips)
              </label>
              <span className="text-[11px] text-[#5c5f77] font-medium">
                {formData.preferred_skills.length} added
              </span>
            </div>
            <div
              onClick={() => preferredInputRef.current?.focus()}
              className="p-3 bg-[#eff1f5] border border-[#ccd0da] rounded-xl focus-within:ring-2 focus-within:ring-[#8839ef]/30 focus-within:border-[#8839ef] focus-within:bg-white transition cursor-text"
            >
              <div className="flex flex-wrap gap-1.5 mb-2">
                {formData.preferred_skills.map((skill, index) => (
                  <span
                    key={`${skill}-${index}`}
                    className="inline-flex items-center px-2.5 py-1 rounded-lg text-xs font-medium bg-[#8839ef]/15 text-[#8839ef] border border-[#8839ef]/30 shadow-xs"
                  >
                    <span>{skill}</span>
                    <button
                      type="button"
                      aria-label={`remove preferred skill ${skill}`}
                      onClick={() => handleRemoveSkill('preferred', index)}
                      className="ml-1.5 text-[#8839ef] hover:text-[#7287fd] rounded-full p-0.5 focus:outline-none"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </span>
                ))}
              </div>
              <input
                ref={preferredInputRef}
                type="text"
                placeholder="Add preferred skill (press Enter or comma)..."
                value={prefSkillDraft}
                onChange={(e) => {
                  const val = e.target.value;
                  if (val.includes(',')) {
                    handleAddSkill('preferred', val);
                  } else {
                    setPrefSkillDraft(val);
                  }
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    handleAddSkill('preferred', prefSkillDraft);
                  }
                }}
                className="w-full bg-transparent text-sm text-[#4c4f69] placeholder-[#8c8fa1] focus:outline-none px-1 py-1"
              />
            </div>
          </div>


        </div>
      )}

      {/* Tab 2: Bi-Directional JSON Editor */}
      {activeTab === 'json' && (
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="text-[#5c5f77]">
              Raw JSON representation synchronized in real time with the form fields.
            </span>
            {jsonError ? (
              <span className="inline-flex items-center text-red-600 font-medium">
                <AlertCircle className="w-3.5 h-3.5 mr-1" />
                {jsonError}
              </span>
            ) : (
              <span className="inline-flex items-center text-[#166534] font-medium">
                <CheckCircle2 className="w-3.5 h-3.5 mr-1" />
                Valid JSON
              </span>
            )}
          </div>

          <textarea
            aria-label="JSON Specification"
            value={jsonText}
            onChange={handleJsonChange}
            rows={14}
            spellCheck={false}
            className={`w-full p-3.5 font-mono text-xs rounded-xl border bg-[#eff1f5] text-[#4c4f69] focus:outline-none focus:ring-2 focus:bg-white transition ${
              jsonError
                ? 'border-red-400 focus:ring-red-400'
                : 'border-[#ccd0da] focus:ring-[#7287fd]/30 focus:border-[#7287fd]'
            }`}
          />
        </div>
      )}
    </div>
  );
}

export default JobSpecPanel;
