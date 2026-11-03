<script lang="ts">
  import type { ColorLevel } from './context-gauge';
  import { fColorLevelTextClass } from './context-gauge';

  interface Props {
    percent: number | null;
    level: ColorLevel;
    size?: 'sm' | 'md';
  }

  let { percent: vPercent, level: vLevel, size: vSize = 'sm' }: Props = $props();

  const cRadius = 11;
  const cCircumference = 2 * Math.PI * cRadius;

  const cStrokeLevelClass = $derived(fColorLevelTextClass(vLevel));
  const cDimensions = $derived(vSize === 'md' ? 'h-6 w-6' : 'h-5 w-5');
  const cStrokeWidth = $derived(vSize === 'md' ? 4 : 3);
</script>

<svg viewBox="0 0 32 32" fill="none" class={cDimensions}>
  <circle
    cx="16"
    cy="16"
    r={cRadius}
    stroke="currentColor"
    stroke-opacity="0.1"
    stroke-width={cStrokeWidth}
  />

  <circle
    cx="16"
    cy="16"
    r={cRadius}
    class="transition-colors duration-300 {cStrokeLevelClass}"
    stroke="currentColor"
    stroke-width={cStrokeWidth}
    stroke-linecap="round"
    stroke-dasharray={cCircumference}
    stroke-dashoffset={vPercent !== null ? cCircumference * (1 - vPercent / 100) : cCircumference}
    transform="rotate(-90 16 16)"
  />
</svg>
