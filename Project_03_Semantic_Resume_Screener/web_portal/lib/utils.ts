import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export interface FetchRetryOptions {
  maxRetries?: number;
  delayMs?: number;
  fetchFn?: typeof fetch;
}

/**
 * Executes a fetch request with automatic exponential backoff retries
 * for transient network errors (e.g. "Load failed", "Failed to fetch")
 * or 502/503/504 server gateway/timeout errors.
 */
export async function fetchWithRetry(
  input: RequestInfo | URL,
  init?: RequestInit,
  options?: FetchRetryOptions
): Promise<Response> {
  const maxRetries = options?.maxRetries ?? 2;
  const initialDelay = options?.delayMs ?? 600;
  const doFetch = options?.fetchFn ?? (typeof window !== 'undefined' ? window.fetch.bind(window) : fetch);

  let lastError: unknown;
  let delay = initialDelay;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await doFetch(input, init);
      // Retry on server gateway/cold-start errors (502, 503, 504)
      if ((response.status === 502 || response.status === 503 || response.status === 504) && attempt < maxRetries) {
        await new Promise((resolve) => setTimeout(resolve, delay));
        delay *= 2;
        continue;
      }
      return response;
    } catch (err) {
      lastError = err;
      if (attempt < maxRetries) {
        await new Promise((resolve) => setTimeout(resolve, delay));
        delay *= 2;
        continue;
      }
      throw err;
    }
  }

  throw lastError;
}

