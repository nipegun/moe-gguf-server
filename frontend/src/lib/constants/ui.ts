import { fTranslate } from '../i18n';
import { Activity, Box, Settings, SquarePen } from '@lucide/svelte';
import type { Component } from 'svelte';
import { cRoutes } from './routes';

export const cForkTreeDepthPadding = 8;
export const cSystemMessagePlaceholder = fTranslate('text_e066e37463a8');

/** Chats of the "Recent chats" section of the sidebar: the ones the user worked with last */
export const cSidebarRecentConversations = 5;

export const cIconStripTransitionDuration = 150;
export const cIconStripTransitionDelayMultiplier = 50;

/** Max height for tool-result code blocks (json / source / diff / streaming code). */
export const cMaxHeightCodeBlock = '22rem';

export interface DesktopIconStripItem {
  icon: Component;
  tooltip: string;
  route: string;
  activeRouteId?: string;
  activeRoutePrefix?: string;
  activeUrlIncludes?: string;
  keys?: string[];
}

export const aSidebarActionsItems: DesktopIconStripItem[] = [
  {
    icon: SquarePen,
    tooltip: fTranslate('text_db18382a249e'),
    route: cRoutes.NEW_CHAT,
    keys: ['shift', 'cmd', 'o']
  },
  {
    icon: Activity,
    tooltip: fTranslate('text_4c2e1df4570b'),
    route: cRoutes.MONITOR,
    activeRouteId: '/monitor'
  },
  {
    icon: Box,
    tooltip: fTranslate('text_5e2c614c23f0'),
    route: cRoutes.MODEL,
    activeRouteId: '/model'
  },
  {
    icon: Settings,
    tooltip: fTranslate('text_74a883a037bc'),
    route: `${cRoutes.SETTINGS}/general`,
    activeUrlIncludes: '#/settings'
  }
];
