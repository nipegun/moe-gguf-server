export const cBoxBorder =
  'border border-border/30 focus-within:border-border  dark:border-border/20 dark:focus-within:border-border';

export const cInputClasses = `
    bg-muted/60 dark:bg-muted/75
    ${cBoxBorder}
    shadow-sm
    outline-none
    text-foreground
`;

export const cPanelClasses = `
    bg-background
    border border-border/30 dark:border-border/20
    shadow-sm backdrop-blur-lg!
    rounded-t-lg!
`;

export const cChatFormPopoverMaxHeight = 'max-h-80';
export const cDialogSubmenuContent = 'w-60';

/** Default Tailwind size class for inline icon components (lucide, etc.). */
export const cIconClassDefault = 'h-4 w-4';

/** Icon size + spinning animation; used for live-streaming tool indicators. */
export const cIconClassSpin = 'h-4 w-4 animate-spin';
