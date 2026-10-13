import { fTranslate } from '../i18n';
import { Activity, Search, Settings, SquarePen } from '@lucide/svelte';
import McpLogo from '$lib/components/app/mcp/McpLogo.svelte';
import type { Component } from 'svelte';
import { cRoutes } from './routes';

export const cForkTreeDepthPadding = 8;
export const cSystemMessagePlaceholder = fTranslate('text_e066e37463a8');

export const cIconStripTransitionDuration = 150;
export const cIconStripTransitionDelayMultiplier = 50;

/** Max height for tool-result code blocks (json / source / diff / streaming code). */
export const cMaxHeightCodeBlock = '22rem';

export interface DesktopIconStripItem {
  icon: Component;
  tooltip: string;
  route?: string;
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
  { icon: Search, tooltip: fTranslate('text_49c266baaaa7'), keys: ['cmd', 'k'] },
  {
    icon: McpLogo,
    tooltip: fTranslate('text_aab669a72882'),
    route: cRoutes.MCP_SERVERS,
    activeRouteId: '/mcp-servers'
  },
  {
    icon: Settings,
    tooltip: fTranslate('text_74a883a037bc'),
    route: `${cRoutes.SETTINGS}/general`,
    activeUrlIncludes: '#/settings'
  },
  {
    icon: Activity,
    tooltip: fTranslate('text_4c2e1df4570b'),
    route: cRoutes.MONITOR,
    activeRouteId: '/monitor'
  }
];
