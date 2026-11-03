// Shared constants for diagram blocks (mermaid and svg) that toggle between a
// rendered view and a source view. The wrapper carries the active mode, css
// drives the visibility, the click handler only flips the attribute.

export const cDiagramViewModeAttr = 'data-view-mode';
export const cDiagramViewRendered = 'rendered';
export const cDiagramViewSource = 'source';
export const cDiagramSourceClass = 'diagram-source';
export const cToggleSourceBtnClass = 'toggle-source-btn';
