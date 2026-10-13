import { fTranslate } from '../i18n';
// Registry of built-in and frontend (browser) tools whose renderer
// shows a recognizable icon and friendly label inline in the chat UI.
//
// To add a new built-in tool, add an entry to BUILTIN_TOOL_UI. To give a
// tool a custom title or body renderer, add a dedicated component under
// ChatMessageToolCall/ and route it in ChatMessageToolCallBlock.svelte
// (see ChatMessageToolCallBlockGetDatetime and
// ChatMessageToolCallBlockSearchResults for prior art).

import type { Component } from 'svelte';
import {
  Braces,
  Clock,
  FilePen,
  FilePlus,
  FileSearch,
  FileText,
  SearchCode,
  Terminal
} from '@lucide/svelte';
import { BuiltInTool, ToolSource } from '$lib/enums';

export interface BuiltinToolUiEntry {
  icon: Component;
  label: string;
  source: ToolSource.BUILTIN | ToolSource.FRONTEND;
}

export const dBuiltinToolUi: Readonly<Record<BuiltInTool, BuiltinToolUiEntry>> = {
  [BuiltInTool.READ_FILE]: {
    icon: FileText,
    label: fTranslate('text_47659c3dccc8'),
    source: ToolSource.BUILTIN
  },
  [BuiltInTool.EDIT_FILE]: {
    icon: FilePen,
    label: fTranslate('text_9608dd142d9b'),
    source: ToolSource.BUILTIN
  },
  [BuiltInTool.WRITE_FILE]: {
    icon: FilePlus,
    label: fTranslate('text_997357233881'),
    source: ToolSource.BUILTIN
  },
  [BuiltInTool.FILE_GLOB_SEARCH]: {
    icon: FileSearch,
    label: fTranslate('text_179fed85ec50'),
    source: ToolSource.BUILTIN
  },
  [BuiltInTool.GREP_SEARCH]: {
    icon: SearchCode,
    label: fTranslate('text_5fcd436a8042'),
    source: ToolSource.BUILTIN
  },
  [BuiltInTool.GET_DATETIME]: {
    icon: Clock,
    label: fTranslate('text_06dafda7802b'),
    source: ToolSource.BUILTIN
  },
  [BuiltInTool.EXEC_SHELL_COMMAND]: {
    icon: Terminal,
    label: fTranslate('text_87e30f34875f'),
    source: ToolSource.BUILTIN
  },
  [BuiltInTool.RUN_JAVASCRIPT]: {
    icon: Braces,
    label: fTranslate('text_9c90cb6e8c27'),
    source: ToolSource.FRONTEND
  }
} as const;

export function fGetBuiltinToolUi(pToolName: string | undefined): BuiltinToolUiEntry | null {
  if (!pToolName) return null;
  return (dBuiltinToolUi as Record<string, BuiltinToolUiEntry>)[pToolName] ?? null;
}
