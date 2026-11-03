/**
 * Gets a display label for a file type from various input formats
 *
 * Handles:
 * - MIME types: 'application/pdf' → 'PDF'
 * - AttachmentType values: 'PDF', 'AUDIO' → 'PDF', 'AUDIO'
 * - File names: 'document.pdf' → 'PDF'
 * - Unknown: returns 'FILE'
 *
 * @param pInput - MIME type, AttachmentType value, or file name
 * @returns Formatted file type label (uppercase)
 */
export function fGetFileTypeLabel(pInput: string | undefined): string {
  if (!pInput) return 'FILE';

  // Handle MIME types (contains '/')
  if (pInput.includes('/')) {
    const cSubtype = pInput.split('/').pop();
    if (cSubtype) {
      // Handle special cases like 'vnd.ms-excel' → 'EXCEL'
      if (cSubtype.includes('.')) {
        return cSubtype.split('.').pop()?.toUpperCase() || 'FILE';
      }
      return cSubtype.toUpperCase();
    }
  }

  // Handle file names (contains '.')
  if (pInput.includes('.')) {
    const cExt = pInput.split('.').pop();
    if (cExt) return cExt.toUpperCase();
  }

  // Handle AttachmentType or other plain strings
  return pInput.toUpperCase();
}
