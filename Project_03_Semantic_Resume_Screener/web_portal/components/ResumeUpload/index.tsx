'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  UploadCloud,
  FileText,
  CheckCircle2,
  AlertCircle,
  AlertTriangle,
  Loader2,
  X,
} from 'lucide-react';
import { extractAndScrub } from '@/lib/extractor';

export interface ExtractedResumeItem {
  candidate_id: string;
  text: string;
  file?: File;
}

export type ExtractionStatus = 'extracting' | 'done' | 'error';

export interface ResumeFileItem {
  id: string;
  file: File;
  name: string;
  size: number;
  status: ExtractionStatus;
  charCount?: number;
  text?: string;
  errorMessage?: string;
}

export interface ValidationResult {
  validFiles: File[];
  rejectedFiles: File[];
  errorMessage: string | null;
}

export interface ProcessPdfOptions {
  onProgress?: (item: ResumeFileItem) => void;
  onReady?: (items: ExtractedResumeItem[]) => void;
}

export interface ProcessPdfResult {
  items: ResumeFileItem[];
  readyItems: ExtractedResumeItem[];
  errorMessage: string | null;
}

export interface ResumeUploadProps {
  /** Callback triggered when all valid files have completed extraction */
  onReady?: (items: ExtractedResumeItem[]) => void;
  /** Callback triggered whenever the list of resume items changes */
  onItemsChange?: (items: ResumeFileItem[]) => void;
  /** Callback triggered when the Analyze Resumes CTA button is clicked */
  onAnalyze?: () => void;
  /** Whether the parent job spec and conditions allow analysis */
  canAnalyze?: boolean;
  /** Initial items for testing or pre-hydrated state */
  initialItems?: ResumeFileItem[];
  /** Initial inline error message for testing */
  initialError?: string | null;
  /** Optional custom CSS classes */
  className?: string;
}

/**
 * Validates selected files for PDF compliance (.pdf extension or application/pdf mime type).
 * Rejects non-PDF files with an inline error message.
 */
export function validatePdfFiles(files: File[] | FileList): ValidationResult {
  const fileArray = Array.from(files);
  const validFiles: File[] = [];
  const rejectedFiles: File[] = [];

  for (const file of fileArray) {
    const isPdf =
      file.name.toLowerCase().endsWith('.pdf') ||
      file.type === 'application/pdf';

    if (isPdf) {
      validFiles.push(file);
    } else {
      rejectedFiles.push(file);
    }
  }

  let errorMessage: string | null = null;
  if (rejectedFiles.length > 0) {
    const names = rejectedFiles.map((f) => f.name).join(', ');
    errorMessage = `Only PDF files are supported. Rejected: ${names}`;
  }

  return { validFiles, rejectedFiles, errorMessage };
}

/**
 * Derives a deterministic candidate ID from the file name.
 */
export function generateCandidateId(filename: string, index?: number): string {
  const base = filename.replace(/\.[^/.]+$/, '').trim();
  const sanitized = base.replace(/[^a-zA-Z0-9_-]/g, '_').replace(/^_+|_+$/g, '');
  if (sanitized.length > 0) return sanitized;
  return `cand-${(index ?? 0) + 1}`;
}

/**
 * Formats byte size into human-readable string.
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

/**
 * Validates, extracts, and scrubs multiple files.
 * Provides real-time progress callbacks per file and invokes onReady with completed items.
 */
export async function processPdfFiles(
  files: File[],
  options?: ProcessPdfOptions
): Promise<ProcessPdfResult> {
  const validation = validatePdfFiles(files);
  const items: ResumeFileItem[] = [];
  const readyItems: ExtractedResumeItem[] = [];

  for (let idx = 0; idx < validation.validFiles.length; idx++) {
    const file = validation.validFiles[idx];
    const initialItem: ResumeFileItem = {
      id: `${file.name}-${Date.now()}-${idx}`,
      file,
      name: file.name,
      size: file.size,
      status: 'extracting',
    };
    items.push(initialItem);
    options?.onProgress?.(initialItem);

    try {
      const scrubbedText = await extractAndScrub(file);
      initialItem.status = 'done';
      initialItem.charCount = scrubbedText.length;
      initialItem.text = scrubbedText;
      options?.onProgress?.(initialItem);

      readyItems.push({
        candidate_id: generateCandidateId(file.name, idx),
        text: scrubbedText,
        file,
      });
    } catch (err) {
      initialItem.status = 'error';
      initialItem.errorMessage = err instanceof Error ? err.message : 'Extraction failed';
      options?.onProgress?.(initialItem);
    }
  }

  if (readyItems.length > 0) {
    options?.onReady?.(readyItems);
  }

  return {
    items,
    readyItems,
    errorMessage: validation.errorMessage,
  };
}

export function ResumeUpload({
  onReady,
  onItemsChange,
  onAnalyze,
  canAnalyze,
  initialItems,
  initialError,
  className = '',
}: ResumeUploadProps) {
  const [items, setItems] = useState<ResumeFileItem[]>(initialItems || []);
  const [errorMessage, setErrorMessage] = useState<string | null>(initialError || null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Sync initial items if props change (useful in testing and navigation)
  useEffect(() => {
    if (initialItems && initialItems !== items) {
      setItems(initialItems);
    }
  }, [initialItems]);

  // Keep parent notified of current items
  const onItemsChangeRef = useRef(onItemsChange);
  useEffect(() => {
    onItemsChangeRef.current = onItemsChange;
  }, [onItemsChange]);

  useEffect(() => {
    onItemsChangeRef.current?.(items);
  }, [items]);

  // Sync initial error if props change
  useEffect(() => {
    if (initialError !== undefined) {
      setErrorMessage(initialError);
    }
  }, [initialError]);

  const processFiles = useCallback(
    async (files: File[]) => {
      const validation = validatePdfFiles(files);
      setErrorMessage(validation.errorMessage);

      if (validation.validFiles.length === 0) {
        return;
      }

      const result = await processPdfFiles(files, {
        onProgress: (updatedItem) => {
          setItems((prev) => {
            const index = prev.findIndex((it) => it.id === updatedItem.id);
            if (index === -1) {
              return [...prev, updatedItem];
            }
            const copy = [...prev];
            copy[index] = updatedItem;
            return copy;
          });
        },
      });

      // Call onReady directly in the async completion block outside setItems
      if (result.readyItems.length > 0) {
        setItems((current) => {
          const allDone = current.filter((it) => it.status === 'done' && it.text);
          const combinedReady: ExtractedResumeItem[] = allDone.map((it, idx) => ({
            candidate_id: generateCandidateId(it.name, idx),
            text: it.text!,
            file: it.file,
          }));
          const payload = combinedReady.length > 0 ? combinedReady : result.readyItems;
          setTimeout(() => onReady?.(payload), 0);
          return current;
        });
      }
    },
    [onReady]
  );

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer && e.dataTransfer.files) {
      processFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFiles(Array.from(e.target.files));
      // Reset input value so same files can be re-selected if desired
      e.target.value = '';
    }
  };

  const handleBrowseClick = () => {
    fileInputRef.current?.click();
  };

  const handleRemoveItem = (id: string) => {
    setItems((prev) => {
      const remaining = prev.filter((it) => it.id !== id);
      const doneItems = remaining.filter((it) => it.status === 'done' && it.text);
      const readyPayload: ExtractedResumeItem[] = doneItems.map((it, idx) => ({
        candidate_id: generateCandidateId(it.name, idx),
        text: it.text!,
        file: it.file,
      }));
      setTimeout(() => onReady?.(readyPayload), 0);
      return remaining;
    });
  };

  const totalFilesCount = items.length;
  const doneItems = items.filter((it) => it.status === 'done');
  const hasExtractedFiles = doneItems.length > 0;
  const isLargeBatch = totalFilesCount > 50;

  // CTA button is enabled when at least 1 file is extracted and parent specifies canAnalyze
  // If canAnalyze is undefined, it defaults to true when files exist
  const isAnalyzeButtonEnabled = hasExtractedFiles && canAnalyze !== false;

  return (
    <div
      className={`rounded-2xl border border-[#ccd0da] bg-white p-4 sm:p-6 shadow-sm text-[#4c4f69] flex flex-col justify-between w-full ${className}`}
    >
      <div className="space-y-4">
        {/* Top Header matching JobSpecPanel */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#ccd0da] pb-3 mb-2">
          <div>
            <h2 className="text-base font-bold text-[#4c4f69] tracking-tight">
              Resume Upload
            </h2>
            <p className="text-xs text-[#5c5f77]">
              Client-side PDF extraction, text extraction & evaluation
            </p>
          </div>

          <div className="flex items-center space-x-2">
            <span className="inline-flex items-center space-x-1.5 rounded-full border border-[#ccd0da] bg-[#eff1f5] px-3 py-1 text-xs font-semibold text-[#5c5f77] shadow-xs">
              <FileText className="h-3.5 w-3.5 text-[#7287fd]" />
              <span>{`${totalFilesCount} ${totalFilesCount === 1 ? 'file' : 'files'}`}</span>
            </span>
          </div>
        </div>

        {/* Hidden File Picker Input */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          multiple
          className="hidden"
          onChange={handleFileInputChange}
        />

        {/* Drag & Drop Zone */}
        <div
          onDragOver={handleDragOver}
          onDragEnter={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={handleBrowseClick}
          role="button"
          tabIndex={0}
          aria-label="Upload PDF resumes"
          className={`flex flex-col items-center justify-center p-6 sm:p-8 border-2 border-dashed rounded-xl cursor-pointer transition-all duration-200 ${
            isDragging
              ? 'border-[#7287fd] bg-[#7287fd]/10 scale-[1.01]'
              : 'border-[#ccd0da] bg-[#eff1f5] hover:border-[#7287fd] hover:bg-[#e6e9ef]'
          }`}
        >
          <div className="flex flex-col items-center text-center space-y-2.5">
            <div className="p-2.5 bg-white rounded-xl border border-[#ccd0da] text-[#7287fd] shadow-xs">
              <UploadCloud className="w-7 h-7 text-[#7287fd]" />
            </div>
            <div>
              <p className="text-sm sm:text-base font-bold text-[#4c4f69]">
                Drag and drop PDF resumes
              </p>
              <p className="text-xs text-[#5c5f77] mt-0.5">
                Supports multiple files (client-side extraction)
              </p>
            </div>
            <button
              type="button"
              className="px-3.5 py-1.5 text-xs font-semibold text-[#4c4f69] bg-white hover:bg-[#eff1f5] rounded-xl border border-[#ccd0da] shadow-xs transition"
              onClick={(e) => {
                e.stopPropagation();
                handleBrowseClick();
              }}
            >
              Browse Files
            </button>
          </div>
        </div>

        {/* Inline Error Message for Non-PDF or Rejections */}
        {errorMessage && (
          <div
            role="alert"
            className="flex items-start gap-2.5 p-3 text-sm text-[#991b1b] bg-[#fee2e2] border border-[#fecaca] rounded-xl animate-in fade-in"
          >
            <AlertCircle className="w-4 h-4 mt-0.5 shrink-0 text-[#991b1b]" />
            <div className="flex-1 font-medium">{errorMessage}</div>
            <button
              type="button"
              onClick={() => setErrorMessage(null)}
              className="text-[#991b1b] hover:text-[#7f1d1d] p-0.5"
              aria-label="Dismiss error"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Soft Warning Banner for Large Batches (> 50) */}
        {isLargeBatch && (
          <div
            role="status"
            className="flex items-center gap-2.5 p-3 text-xs text-[#9a3412] bg-[#ffedd5] border border-[#fed7aa] rounded-xl"
          >
            <AlertTriangle className="w-4 h-4 shrink-0 text-[#9a3412]" />
            <span>Large batches (50+) may take longer to process.</span>
          </div>
        )}

        {/* File List */}
        {totalFilesCount > 0 && (
          <div className="border border-[#ccd0da] bg-white rounded-xl overflow-hidden shadow-xs">
            <div className="px-3.5 py-2.5 border-b border-[#ccd0da] bg-[#f4f5f8] flex items-center justify-between text-xs text-[#5c5f77]">
              <span className="font-bold text-[#4c4f69]">
                {`Uploaded Resumes (${totalFilesCount})`}
              </span>
              <span>
                {`${doneItems.length} of ${totalFilesCount} processed`}
              </span>
            </div>

            <ul className="divide-y divide-[#ccd0da]/60 max-h-56 overflow-y-auto">
              {items.map((item) => (
                <li
                  key={item.id}
                  className="px-3.5 py-2 flex items-center justify-between gap-3 hover:bg-[#eff1f5] transition-colors"
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    <FileText className="w-4 h-4 text-[#7287fd] shrink-0" />
                    <div className="min-w-0">
                      <p className="text-xs font-semibold text-[#4c4f69] truncate">
                        {item.name}
                      </p>
                      <p className="text-[11px] text-[#5c5f77]">
                        {formatFileSize(item.size)}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2.5 shrink-0">
                    {/* Status Indicator */}
                    {item.status === 'extracting' && (
                      <div className="flex items-center gap-1 text-xs text-[#7287fd]">
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        <span>Extracting...</span>
                      </div>
                    )}

                    {item.status === 'done' && (
                      <div className="flex items-center gap-1 text-xs text-[#166534] font-semibold bg-[#dcfce7] px-2 py-0.5 rounded-md border border-[#86efac]">
                        <span>{`${item.charCount?.toLocaleString()} chars`}</span>
                        <CheckCircle2 className="w-3.5 h-3.5 text-[#166534]" />
                      </div>
                    )}

                    {item.status === 'error' && (
                      <div className="flex items-center gap-1 text-xs text-[#991b1b] bg-[#fee2e2] px-2 py-0.5 rounded-md border border-[#fecaca]">
                        <span>Failed</span>
                        <AlertCircle className="w-3.5 h-3.5 text-[#991b1b]" />
                      </div>
                    )}

                    {/* Remove Item */}
                    <button
                      type="button"
                      onClick={() => handleRemoveItem(item.id)}
                      className="text-[#5c5f77] hover:text-[#4c4f69] p-1 rounded-lg hover:bg-[#eff1f5]"
                      aria-label={`Remove ${item.name}`}
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* CTA Button: Analyze Resumes */}
      {hasExtractedFiles && (
        <div className="pt-4 mt-auto flex flex-col items-end gap-1">
          <button
            type="button"
            disabled={!isAnalyzeButtonEnabled}
            onClick={onAnalyze}
            className={`w-full py-3 px-4 rounded-xl font-bold text-sm transition-all duration-200 flex items-center justify-center gap-2 shadow-xs ${
              isAnalyzeButtonEnabled
                ? 'bg-[#7287fd] hover:bg-[#8839ef] text-white shadow-md shadow-[#7287fd]/20 cursor-pointer'
                : 'bg-[#eff1f5] text-[#8c8fa1] cursor-not-allowed border border-[#ccd0da]'
            }`}
          >
            Analyze Resumes
          </button>
          {!canAnalyze && canAnalyze !== undefined && (
            <p className="text-xs text-[#5c5f77]">
              Complete the Job Specification to enable analysis.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export default ResumeUpload;
