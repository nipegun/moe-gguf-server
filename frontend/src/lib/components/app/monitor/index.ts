/**
 *
 * MONITOR
 *
 * CPU, GPU and memory monitoring. Data comes from monitorStore, which polls
 * `/monitor` (Monitor page) and `/monitor/usage` (status bar).
 *
 */

/**
 * **MonitorScreen** - Monitor page: GPU and CPU usage (gpu-N and cpu-1 cards) and
 * memory of the loaded models by purpose on every GPU and in RAM.
 */
export { default as MonitorScreen } from './MonitorScreen.svelte';

/**
 * **MonitorMemoryBar** - Stacked memory bar (one GPU or the RAM) with a legend
 * that lists every segment with its size and share.
 */
export { default as MonitorMemoryBar } from './MonitorMemoryBar.svelte';

/**
 * **MonitorMeter** - Labelled horizontal usage meter (0-100 %): a green bar that
 * the usage fills in red.
 */
export { default as MonitorMeter } from './MonitorMeter.svelte';

/**
 * **StatusBar** - Fixed bottom bar with CPU and GPU usage; opens the Monitor.
 */
export { default as StatusBar } from './StatusBar.svelte';
