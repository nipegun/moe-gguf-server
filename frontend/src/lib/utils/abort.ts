import { fTranslate } from '../i18n';
/**
 * Abort Signal Utilities
 *
 * Provides utilities for consistent AbortSignal propagation across the application.
 * These utilities help ensure that async operations can be properly cancelled
 * when needed (e.g., user stops generation, navigates away, etc.).
 */

// the standard DOMException name for a cancelled operation
const cAbortErrorName = 'AbortError';

// browser specific TypeError messages emitted when a fetch reader is cut by page unload,
// navigation, or a transient network drop. functionally aborts, not actionable errors
const aAbortLikeMessagePatterns = [
  /input stream/i, // Firefox: stream cut at unload
  /network connection was lost/i, // Safari: transient network drop
  /load failed/i // Safari: page navigation during fetch
];

/**
 * Throws an AbortError if the signal is aborted.
 * Use this at the start of async operations to fail fast.
 *
 * @param pSignal - Optional AbortSignal to check
 * @throws DOMException with name 'AbortError' if signal is aborted
 *
 * @example
 * ```ts
 * async function fetchData(signal?: AbortSignal) {
 *   throwIfAborted(signal);
 *   // ... proceed with operation
 * }
 * ```
 */
export function fThrowIfAborted(pSignal?: AbortSignal): void {
  if (pSignal?.aborted) {
    throw new DOMException(fTranslate('text_ed194099610c'), cAbortErrorName);
  }
}

/**
 * Checks if an error is an AbortError.
 * Use this to distinguish between user-initiated cancellation and actual errors.
 *
 * @param pError - Error to check
 * @returns true if the error is an AbortError
 *
 * @example
 * ```ts
 * try {
 *   await fetchData(signal);
 * } catch (error) {
 *   if (isAbortError(error)) {
 *     // User cancelled - no error dialog needed
 *     return;
 *   }
 *   // Handle actual error
 * }
 * ```
 */
export function fIsAbortError(pError: unknown): boolean {
  if (pError instanceof DOMException && pError.name === cAbortErrorName) {
    return true;
  }
  if (pError instanceof Error) {
    if (pError.name === cAbortErrorName) {
      return true;
    }
    // these patterns are functionally aborts, keep them out of the red console
    if (pError instanceof TypeError) {
      const cMsg = pError.message ?? '';
      if (aAbortLikeMessagePatterns.some((pRe) => pRe.test(cMsg))) return true;
    }
  }
  return false;
}

/**
 * Creates a new AbortController that is linked to one or more parent signals.
 * When any parent signal aborts, the returned controller also aborts.
 *
 * Useful for creating child operations that should be cancelled when
 * either the parent operation or their own timeout/condition triggers.
 *
 * @param pSignals - Parent signals to link to (undefined signals are ignored)
 * @returns A new AbortController linked to all provided signals
 *
 * @example
 * ```ts
 * // Link to user's abort signal and add a timeout
 * const linked = createLinkedController(userSignal, timeoutSignal);
 * await fetch(url, { signal: linked.signal });
 * ```
 */
export function fCreateLinkedController(...pSignals: (AbortSignal | undefined)[]): AbortController {
  const cController = new AbortController();

  for (const cSignal of pSignals) {
    if (!cSignal) continue;

    // If already aborted, abort immediately
    if (cSignal.aborted) {
      cController.abort(cSignal.reason);
      return cController;
    }

    // Link to parent signal
    cSignal.addEventListener('abort', () => cController.abort(cSignal.reason), { once: true });
  }

  return cController;
}

/**
 * Creates an AbortSignal that times out after the specified duration.
 *
 * @param pMs - Timeout duration in milliseconds
 * @returns AbortSignal that will abort after the timeout
 *
 * @example
 * ```ts
 * const signal = createTimeoutSignal(5000); // 5 second timeout
 * await fetch(url, { signal });
 * ```
 */
export function fCreateTimeoutSignal(pMs: number): AbortSignal {
  return AbortSignal.timeout(pMs);
}

/**
 * Wraps a promise to reject if the signal is aborted.
 * Useful for making non-abortable promises respect an AbortSignal.
 *
 * @param pPromise - Promise to wrap
 * @param pSignal - AbortSignal to respect
 * @returns Promise that rejects with AbortError if signal aborts
 *
 * @example
 * ```ts
 * // Make a non-abortable operation respect abort signal
 * const result = await withAbortSignal(
 *   someNonAbortableOperation(),
 *   signal
 * );
 * ```
 */
export async function fWithAbortSignal<T>(pPromise: Promise<T>, pSignal?: AbortSignal): Promise<T> {
  if (!pSignal) return pPromise;

  fThrowIfAborted(pSignal);

  return new Promise<T>((pResolve, pReject) => {
    const fAbortHandler = () => {
      pReject(new DOMException(fTranslate('text_ed194099610c'), cAbortErrorName));
    };

    pSignal.addEventListener('abort', fAbortHandler, { once: true });

    pPromise
      .then((pValue) => {
        pSignal.removeEventListener('abort', fAbortHandler);
        pResolve(pValue);
      })
      .catch((pError) => {
        pSignal.removeEventListener('abort', fAbortHandler);
        pReject(pError);
      });
  });
}
