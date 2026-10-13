import {
  cMsPerSecond,
  cSecondsPerMinute,
  cSecondsPerHour,
  cShortDurationThreshold,
  cMediumDurationThreshold
} from '$lib/constants';

/**
 * Formats file size in bytes to human readable format
 * Supports Bytes, KB, MB, and GB
 *
 * @param pBytes - File size in bytes (or unknown for safety)
 * @returns Formatted file size string
 */
export function fFormatFileSize(pBytes: number | unknown): string {
  if (typeof pBytes !== 'number') return 'Unknown';
  if (pBytes === 0) return '0 Bytes';

  const cK = 1024;
  const aSizes = ['Bytes', 'KB', 'MB', 'GB'];
  const cI = Math.floor(Math.log(pBytes) / Math.log(cK));

  return parseFloat((pBytes / Math.pow(cK, cI)).toFixed(2)) + ' ' + aSizes[cI];
}

/**
 * Format parameter count to human-readable format (B, M, K)
 *
 * @param pParams - Parameter count
 * @returns Human-readable parameter count
 */
export function fFormatParameters(pParams: number | unknown): string {
  if (typeof pParams !== 'number') return 'Unknown';

  if (pParams >= 1e9) {
    return `${(pParams / 1e9).toFixed(2)}B`;
  }

  if (pParams >= 1e6) {
    return `${(pParams / 1e6).toFixed(2)}M`;
  }

  if (pParams >= 1e3) {
    return `${(pParams / 1e3).toFixed(2)}K`;
  }

  return pParams.toString();
}

/**
 * Format number with locale-specific thousands separators
 *
 * @param pNum - Number to format
 * @returns Human-readable number
 */
export function fFormatNumber(pNum: number | unknown): string {
  if (typeof pNum !== 'number') return 'Unknown';

  return pNum.toLocaleString();
}

/**
 * Format JSON string with pretty printing (2-space indentation)
 * Returns original string if parsing fails
 *
 * @param pJsonString - JSON string to format
 * @returns Pretty-printed JSON string or original if invalid
 */
export function fFormatJsonPretty(pJsonString: string): string {
  try {
    const cParsed = JSON.parse(pJsonString);
    return JSON.stringify(cParsed, null, 2);
  } catch {
    return pJsonString;
  }
}

/**
 * Format time as HH:MM:SS in 24-hour format
 *
 * @param pDate - Date object to format
 * @returns Formatted time string (HH:MM:SS)
 */
export function fFormatTime(pDate: Date): string {
  return pDate.toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });
}

/**
 * Formats milliseconds to a human-readable time string for performance metrics.
 * Examples: "4h 12min 54s", "12min 34s", "45s", "0.5s"
 *
 * @param pMs - Time in milliseconds
 * @returns Formatted time string
 */
export function fFormatPerformanceTime(pMs: number): string {
  if (pMs < 0) return '0s';

  const cTotalSeconds = pMs / cMsPerSecond;

  if (cTotalSeconds < cShortDurationThreshold) {
    return `${cTotalSeconds.toFixed(1)}s`;
  }

  if (cTotalSeconds < cMediumDurationThreshold) {
    return `${cTotalSeconds.toFixed(1)}s`;
  }

  const cHours = Math.floor(cTotalSeconds / cSecondsPerHour);
  const cMinutes = Math.floor((cTotalSeconds % cSecondsPerHour) / cSecondsPerMinute);
  const cSeconds = Math.floor(cTotalSeconds % cSecondsPerMinute);

  const aParts: string[] = [];

  if (cHours > 0) {
    aParts.push(`${cHours}h`);
  }

  if (cMinutes > 0) {
    aParts.push(`${cMinutes}min`);
  }

  if (cSeconds > 0 || aParts.length === 0) {
    aParts.push(`${cSeconds}s`);
  }

  return aParts.join(' ');
}

/**
 * Formats attachment content for API requests with consistent header style.
 * Used when converting message attachments to text content parts.
 *
 * @param pLabel - Type label (e.g., 'File', 'PDF File', 'MCP Prompt')
 * @param pName - File or attachment name
 * @param pContent - The actual content to include
 * @param pExtra - Optional extra info to append to name (e.g., server name for MCP)
 * @returns Formatted string with header and content
 */
export function fFormatAttachmentText(
  pLabel: string,
  pName: string,
  pContent: string,
  pExtra?: string
): string {
  const cHeader = pExtra ? `${pName} (${pExtra})` : pName;
  return `\n\n--- ${pLabel}: ${cHeader} ---\n${pContent}`;
}
