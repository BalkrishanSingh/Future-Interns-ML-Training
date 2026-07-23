'use client';

import React, { useState, useEffect, useRef } from 'react';
import {
  ExternalLink,
  Download,
  ZoomIn,
  ZoomOut,
  RotateCcw,
} from 'lucide-react';
import * as pdfjsLib from 'pdfjs-dist/legacy/build/pdf.mjs';

if (typeof window !== 'undefined' && !pdfjsLib.GlobalWorkerOptions?.workerSrc) {
  pdfjsLib.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjsLib.version}/legacy/build/pdf.worker.min.mjs`;
}

export interface PdfJsCanvasViewerProps {
  pdfUrl: string;
  candidateId?: string;
  className?: string;
}

/**
 * Client-side multi-page PDF.js canvas renderer with iframe fallback.
 * Renders pages onto HTML5 canvas elements when browser canvas is available,
 * and maintains an iframe fallback for SSR, loading, and error states.
 */
export function PdfJsCanvasViewer({
  pdfUrl,
  candidateId = 'candidate',
  className = '',
}: PdfJsCanvasViewerProps) {
  const [scale, setScale] = useState<number>(1.0);
  const [canvasRendered, setCanvasRendered] = useState<boolean>(false);
  const [hasError, setHasError] = useState<boolean>(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Set initial scale responsive to viewport after mount to prevent SSR hydration mismatch (70% scale on < 1024px)
  useEffect(() => {
    if (typeof window !== 'undefined' && window.innerWidth < 1024) {
      setScale(0.7);
    }
  }, []);

  useEffect(() => {
    let isCancelled = false;

    async function renderPages() {
      if (!pdfUrl || typeof window === 'undefined') return;

      try {
        const loadingTask = pdfjsLib.getDocument({
          url: pdfUrl,
          isEvalSupported: false,
          useSystemFonts: true,
        });
        const pdf = await loadingTask.promise;
        if (isCancelled) return;

        const container = containerRef.current;
        if (!container) return;
        container.innerHTML = '';

        const dpr = window.devicePixelRatio || 1;

        for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
          const page = await pdf.getPage(pageNum);
          if (isCancelled) return;

          const viewport = page.getViewport({ scale });

          // Page wrapper for Catppuccin margins, shadow, and page badge
          const pageWrapper = document.createElement('div');
          pageWrapper.className = 'relative flex flex-col items-center mb-4 sm:mb-6 last:mb-0 min-w-0';

          // Page number badge ("Page 1 of N")
          const badge = document.createElement('div');
          badge.className =
            'self-end mb-2 px-2.5 py-0.5 text-[11px] font-semibold text-[#5c5f77] bg-white rounded-md border border-[#ccd0da] shadow-xs';
          badge.textContent = `Page ${pageNum} of ${pdf.numPages}`;
          pageWrapper.appendChild(badge);

          // Rendered page canvas with high-DPI scaling and physical pixel size matching zoom
          const canvas = document.createElement('canvas');
          canvas.className = 'rounded-xl shadow-md border border-[#ccd0da] bg-white block transition-all';
          canvas.style.width = `${Math.round(viewport.width)}px`;
          canvas.style.height = `${Math.round(viewport.height)}px`;
          canvas.width = Math.floor(viewport.width * dpr);
          canvas.height = Math.floor(viewport.height * dpr);

          const context = canvas.getContext('2d');
          if (!context) throw new Error('Canvas 2D context not available');
          context.scale(dpr, dpr);

          await page.render({ canvasContext: context, viewport }).promise;
          if (isCancelled) return;
          pageWrapper.appendChild(canvas);
          container.appendChild(pageWrapper);
        }

        setCanvasRendered(true);
        setHasError(false);
      } catch {
        if (!isCancelled) {
          setHasError(true);
          setCanvasRendered(false);
        }
      }
    }

    renderPages();

    return () => {
      isCancelled = true;
    };
  }, [pdfUrl, scale]);

  return (
    <div className={`relative flex flex-col w-full max-w-full flex-1 min-h-[350px] sm:min-h-[550px] overflow-hidden ${className}`}>
      {/* Dedicated PDF Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-1.5 sm:gap-2 px-2.5 sm:px-3 py-2 bg-[#f4f5f8] rounded-xl border border-[#ccd0da] mb-3 w-full max-w-full">
        {/* Zoom In / Zoom Out / Reset controls */}
        <div className="flex items-center space-x-1">
          <button
            type="button"
            onClick={() => setScale((s) => Math.max(0.5, Math.round((s - 0.15) * 100) / 100))}
            aria-label="Zoom Out"
            title="Zoom Out"
            className="p-1.5 rounded-lg text-[#5c5f77] hover:text-[#4c4f69] hover:bg-white transition border border-transparent hover:border-[#ccd0da] shadow-xs"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <span className="font-mono text-xs font-semibold text-[#4c4f69] px-1 sm:px-1.5 min-w-[38px] sm:min-w-[42px] text-center">
            {Math.round(scale * 100)}%
          </span>
          <button
            type="button"
            onClick={() => setScale((s) => Math.min(2.0, Math.round((s + 0.15) * 100) / 100))}
            aria-label="Zoom In"
            title="Zoom In"
            className="p-1.5 rounded-lg text-[#5c5f77] hover:text-[#4c4f69] hover:bg-white transition border border-transparent hover:border-[#ccd0da] shadow-xs"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            type="button"
            onClick={() => setScale(typeof window !== 'undefined' && window.innerWidth < 640 ? 0.7 : 1.0)}
            aria-label="Reset Zoom"
            title="Reset Zoom"
            className="p-1.5 rounded-lg text-[#5c5f77] hover:text-[#4c4f69] hover:bg-white transition border border-transparent hover:border-[#ccd0da] shadow-xs"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>

        {/* Action buttons: Open in New Tab & Download PDF */}
        <div className="flex items-center space-x-1.5 sm:space-x-2">
          <a
            href={pdfUrl}
            target="_blank"
            rel="noopener noreferrer"
            aria-label="Open in New Tab"
            title="Open PDF in new tab"
            className="inline-flex items-center space-x-1 sm:space-x-1.5 px-2 sm:px-2.5 py-1 text-xs font-medium rounded-lg bg-white border border-[#ccd0da] text-[#5c5f77] hover:text-[#4c4f69] hover:bg-[#eff1f5] transition shadow-xs"
          >
            <ExternalLink className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Open in New Tab</span>
            <span className="sm:hidden">Open</span>
          </a>
          <a
            href={pdfUrl}
            download={`${candidateId}.pdf`}
            aria-label="Download PDF"
            title="Download PDF"
            className="inline-flex items-center space-x-1 sm:space-x-1.5 px-2 sm:px-2.5 py-1 text-xs font-medium rounded-lg bg-[#7287fd]/15 border border-[#7287fd]/30 text-[#7287fd] hover:bg-[#7287fd]/25 transition shadow-xs"
          >
            <Download className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Download PDF</span>
            <span className="sm:hidden">PDF</span>
          </a>
        </div>
      </div>

      {/* Client-side PDF.js Canvas Container */}
      <div
        ref={containerRef}
        data-testid="pdfjs-canvas-viewer"
        className={`w-full max-w-full flex-1 overflow-x-auto overflow-y-auto min-h-[350px] sm:min-h-[550px] p-2 sm:p-4 rounded-xl border border-[#ccd0da] bg-[#eff1f5] ${
          canvasRendered && !hasError ? 'block' : 'hidden'
        }`}
      />

      {/* Iframe Fallback: Active during SSR, loading, and canvas error states */}
      {(!canvasRendered || hasError) && (
        <iframe
          src={pdfUrl}
          title={`PDF viewer for ${candidateId}`}
          className="w-full max-w-full flex-1 min-h-[350px] sm:min-h-[550px] rounded-xl border border-[#ccd0da] bg-[#eff1f5]"
        />
      )}
    </div>
  );
}

export default PdfJsCanvasViewer;
