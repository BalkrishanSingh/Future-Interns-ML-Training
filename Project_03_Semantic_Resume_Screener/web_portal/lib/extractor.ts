/**
 * Client-side PDF text extraction and PII scrubbing module.
 *
 * Provides deterministic text extraction using pdfjs-dist and regex-based PII
 * scrubbing.
 */

import * as pdfjsLib from 'pdfjs-dist/legacy/build/pdf.mjs';

// Configure worker in browser environment if available
if (typeof window !== 'undefined' && !pdfjsLib.GlobalWorkerOptions.workerSrc) {
  pdfjsLib.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjsLib.version}/legacy/build/pdf.worker.min.mjs`;
}

const EMAIL_REGEX = /[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}/g;
const PHONE_REGEX = /(\+?\d{1,3}[\-.\s]?)?\(?\d{3}\)?[\-.\s]?\d{3}[\-.\s]?\d{4}/g;
const URL_REGEX = /https?:\/\/[^\s]+/g;

// Regex patterns to identify educational institutions and schools for blind screening
const INSTITUTION_LINE_REGEX = /\b(college|university|institute|school|academy|polytechnic|campus|vidyalaya|gurukul)\b/i;
const INSTITUTION_NAME_REGEX = /\b(?:[A-Z][a-zA-Z.'-]*\s+)*(?:College|University|Institute|School|Academy|Polytechnic|Campus|Vidyalaya|Gurukul)\b(?:\s+[A-Z][a-zA-Z.'-]*)*(?:\s+(?:of|for|in|at)\s+[A-Z][a-zA-Z.'-]+(?:\s+[A-Z][a-zA-Z.'-]+)*)?|\b(?:high\s+school)\b/g;

const NON_NAME_HEADERS = new Set([
  'resume',
  'curriculum vitae',
  'cv',
  'profile',
  'summary',
  'contact',
  'experience',
  'education',
  'skills',
  'projects',
]);

const EXCLUDE_JOB_TITLE_WORDS = new Set([
  'engineer',
  'developer',
  'manager',
  'director',
  'lead',
  'architect',
  'scientist',
  'consultant',
  'analyst',
  'designer',
  'intern',
  'specialist',
  'administrator',
  'officer',
  'student',
  'senior',
  'junior',
  'staff',
  'principal',
  'experienced',
]);

export type ExtractableFile = File | Blob | ArrayBuffer | Uint8Array;

/**
 * Converts various file/buffer representations to a Uint8Array suitable for pdfjs-dist.
 */
async function toUint8Array(file: ExtractableFile): Promise<Uint8Array> {
  if (file instanceof Uint8Array) {
    return file;
  }
  if (file instanceof ArrayBuffer) {
    return new Uint8Array(file);
  }
  if (file && typeof (file as Blob).arrayBuffer === 'function') {
    const buffer = await (file as Blob).arrayBuffer();
    return new Uint8Array(buffer);
  }
  throw new TypeError('Invalid input: expected File, Blob, ArrayBuffer, or Uint8Array');
}

/**
 * Scrubs personally identifiable information (emails, phone numbers, URLs, and candidate names)
 * from unstructured resume text.
 */
export function scrubPII(text: string): string {
  if (!text) return '';

  // 1. Scrub contact details (URL, Email, Phone)
  let scrubbed = text
    .replace(URL_REGEX, '[REDACTED_URL]')
    .replace(EMAIL_REGEX, '[REDACTED_EMAIL]')
    .replace(PHONE_REGEX, '[REDACTED_PHONE]');

  // 2. Explicit labeled name patterns (e.g. "Name: Balkrishan Singh" or "Candidate Name: Jane Doe")
  scrubbed = scrubbed.replace(
    /(?:^|\b)(?:Name|Candidate Name|Applicant Name)\s*:\s*([A-Za-z\s.'-]+)(?=\n|$)/gi,
    (match, capturedName) => {
      const trimmed = capturedName.trim();
      return trimmed ? match.replace(trimmed, '[REDACTED_NAME]') : match;
    }
  );

  // 3. First prominent non-header title line / line before contact info heuristic
  const lines = scrubbed.split('\n');
  for (let i = 0; i < Math.min(lines.length, 5); i++) {
    const rawLine = lines[i].trim();
    if (!rawLine || rawLine.startsWith('[REDACTED_')) continue;

    const lower = rawLine.toLowerCase();
    if (NON_NAME_HEADERS.has(lower)) continue;

    // A candidate name typically comprises 2 to 4 capitalized words, length 3 to 45 chars
    const words = rawLine.split(/\s+/);
    if (words.length >= 2 && words.length <= 4 && rawLine.length <= 45) {
      const hasJobWord = words.some((w) =>
        EXCLUDE_JOB_TITLE_WORDS.has(w.toLowerCase().replace(/[^a-z]/g, ''))
      );
      const isAllCapitalized = words.every((w) =>
        /^[A-Z][a-zA-Z.'-]*$/.test(w)
      );

      if (isAllCapitalized && !hasJobWord) {
        const escapedName = rawLine.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const nameRegex = new RegExp(`\\b${escapedName}\\b`, 'g');
        scrubbed = scrubbed.replace(nameRegex, '[REDACTED_NAME]');
        break;
      }
    }
  }

  // 4. Scrub educational institutions for blind recruitment (e.g. "Guru Nanak Dev Engineering College", "BCM School")
  const processedLines = scrubbed.split('\n').map((line) => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('[REDACTED_')) return line;

    // Check if line mentions an educational institution / school / college
    if (INSTITUTION_LINE_REGEX.test(trimmed)) {
      const lower = trimmed.toLowerCase();
      const hasDegree = /\b(bachelor|master|b\.?tech|m\.?tech|ph\.?d|doctorate|diploma|b\.?sc|m\.?sc|b\.?e|bca|mca)\b/i.test(lower);
      const isProjectOrExpLine = /\b(built|develop|led|engineer|implement|creat|design|contribut|work|mentor|integrat|project|app|system)\b/i.test(lower);

      // If line is purely an institution declaration without degree or project context, redact clean line
      if (!hasDegree && !isProjectOrExpLine && trimmed.length <= 60) {
        return '[REDACTED_INSTITUTION]';
      }

      // Otherwise redact only the institution entity, preserving degrees and candidate project work
      return line.replace(INSTITUTION_NAME_REGEX, '[REDACTED_INSTITUTION]');
    }
    return line;
  });
  scrubbed = processedLines.join('\n');

  return scrubbed;
}

export const scrubPii = scrubPII;

/**
 * Extracts plain text from a PDF file across all pages.
 */
export async function extractText(file: ExtractableFile): Promise<string> {
  const data = await toUint8Array(file);
  const loadingTask = pdfjsLib.getDocument({
    data,
    isEvalSupported: false,
    useSystemFonts: true,
  });

  const pdfDoc = await loadingTask.promise;
  const pageTexts: string[] = [];

  for (let pageNum = 1; pageNum <= pdfDoc.numPages; pageNum++) {
    const page = await pdfDoc.getPage(pageNum);
    const content = await page.getTextContent();
    let pageText = '';

    for (const item of content.items) {
      if ('str' in item) {
        pageText += item.str + (item.hasEOL ? '\n' : ' ');
      }
    }

    const cleaned = pageText.replace(/[ \t]+/g, ' ').replace(/\n\s+/g, '\n').trim();
    if (cleaned) {
      pageTexts.push(cleaned);
    }
  }

  return pageTexts.join('\n\n');
}

/**
 * Extracts text from a PDF file and immediately scrubs all PII.
 */
export async function extractAndScrub(file: ExtractableFile): Promise<string> {
  const rawText = await extractText(file);
  return scrubPII(rawText);
}
