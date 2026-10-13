/**
 * @param pFn - The function to debounce
 * @param pDelay - The delay in milliseconds
 * @returns A debounced version of the function
 */
export function fDebounce<T extends (...pArgs: Parameters<T>) => void>(
  pFn: T,
  pDelay: number
): (...pArgs: Parameters<T>) => void {
  let vTimeoutId: ReturnType<typeof setTimeout> | null = null;

  return (...pArgs: Parameters<T>) => {
    if (vTimeoutId) {
      clearTimeout(vTimeoutId);
    }

    vTimeoutId = setTimeout(() => {
      pFn(...pArgs);
      vTimeoutId = null;
    }, pDelay);
  };
}
