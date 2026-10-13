/**
 * Automatically resizes a textarea element to fit its content
 * @param pTextareaElement - The textarea element to resize
 */
export default function fAutoResizeTextarea(pTextareaElement: HTMLTextAreaElement | null): void {
  if (pTextareaElement) {
    pTextareaElement.style.height = '1rem';
    pTextareaElement.style.height = pTextareaElement.scrollHeight + 'px';
  }
}
