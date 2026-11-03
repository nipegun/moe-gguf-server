import { fTranslate } from '../i18n';
import type { MonitorSegmentKey } from '$lib/types';

/** Polling period of the Monitor page (`/monitor`). */
export const cMonitorReportIntervalMs = 2000;

/** Polling period of the status bar (`/monitor/usage`). */
export const cMonitorUsageIntervalMs = 2000;

/**
 * One colour of the memory bars. Server segments that serve the same purpose share a colour so
 * the bars never need more than the eight validated categorical hues plus a neutral "other".
 */
export interface MonitorSegmentGroup {
  id: string;
  keys: MonitorSegmentKey[];
  label: string;
  /** CSS custom property (app.css) holding the fill colour in both themes. */
  colorVar: string;
}

/** Fixed drawing order: a colour always follows its entity, whatever else is present. */
export const aMonitorSegmentGroups: MonitorSegmentGroup[] = [
  {
    id: 'mtp',
    keys: ['mtp_weights', 'draft_weights'],
    label: fTranslate('text_4ed35913646b'),
    colorVar: '--monitor-mtp'
  },
  {
    id: 'experts',
    keys: ['expert_weights'],
    label: fTranslate('text_30d702eb53df'),
    colorVar: '--monitor-experts'
  },
  {
    id: 'dense',
    keys: ['dense_weights'],
    label: fTranslate('text_bfa7eaaadb20'),
    colorVar: '--monitor-dense'
  },
  {
    id: 'kv',
    keys: ['kv_cache'],
    label: fTranslate('text_1cf4c1a7d51e'),
    colorVar: '--monitor-kv'
  },
  {
    id: 'draft-kv',
    keys: ['draft_kv_cache'],
    label: fTranslate('text_cb7e52a98a52'),
    colorVar: '--monitor-draft-kv'
  },
  {
    id: 'compute',
    keys: ['compute'],
    label: fTranslate('text_981fc58b1c31'),
    colorVar: '--monitor-compute'
  },
  {
    id: 'expert-cache',
    keys: ['expert_cache', 'prefetch'],
    label: fTranslate('text_ad5e17ca4c04'),
    colorVar: '--monitor-expert-cache'
  },
  {
    id: 'prompt-cache',
    keys: ['prompt_cache'],
    label: fTranslate('text_01dc4471f137'),
    colorVar: '--monitor-prompt-cache'
  },
  {
    id: 'other',
    keys: ['other'],
    label: fTranslate('text_c657ebb778a5'),
    colorVar: '--monitor-other'
  }
];
