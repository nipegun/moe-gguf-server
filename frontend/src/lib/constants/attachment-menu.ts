import { fTranslate } from '../i18n';
import type { Component } from 'svelte';
import { MessageSquare, Zap, FolderOpen } from '@lucide/svelte';
import { cFileTypeIcons } from '$lib/constants/icons';
import {
  AttachmentAction,
  AttachmentItemEnabledWhen,
  AttachmentItemVisibleWhen,
  AttachmentMenuItemId
} from '$lib/enums';

export interface AttachmentMenuItem {
  /** Unique identifier for the item */
  id: AttachmentMenuItemId;
  /** Display label */
  label: string;
  /** Lucide icon component */
  icon: Component;
  /** Extra CSS class applied to the item (e.g. for test selectors) */
  class?: string;
  /** Whether the item requires a specific modality to be enabled */
  enabledWhen?: AttachmentItemEnabledWhen;
  /** Tooltip shown when the item is disabled */
  disabledTooltip?: string;
  /** Callback key on the Props interface to invoke when clicked */
  action: AttachmentAction;
  /** Whether the item is only shown when a specific capability is present */
  visibleWhen?: AttachmentItemVisibleWhen;
  /** Whether this item has a tooltip even when enabled (uses dynamic text) */
  hasEnabledTooltip?: boolean;
}

/**
 * File attachment menu items shown in both the desktop dropdown and mobile sheet.
 * The "Tools" submenu is handled separately by each component.
 */
export const aAttachmentFileItems: AttachmentMenuItem[] = [
  {
    id: AttachmentMenuItemId.IMAGES,
    label: fTranslate('text_be7e2f201293'),
    icon: cFileTypeIcons.image,
    class: 'images-button',
    enabledWhen: AttachmentItemEnabledWhen.HAS_VISION_MODALITY,
    disabledTooltip: fTranslate('text_da3986960d94'),
    action: AttachmentAction.FILE_UPLOAD
  },
  {
    id: AttachmentMenuItemId.AUDIO,
    label: fTranslate('text_85b791f3241c'),
    icon: cFileTypeIcons.audio,
    class: 'audio-button',
    enabledWhen: AttachmentItemEnabledWhen.HAS_AUDIO_MODALITY,
    disabledTooltip: fTranslate('text_546305b42e63'),
    action: AttachmentAction.FILE_UPLOAD
  },
  {
    id: AttachmentMenuItemId.VIDEO,
    label: fTranslate('text_f1719b8182ad'),
    icon: cFileTypeIcons.video,
    class: 'video-button',
    enabledWhen: AttachmentItemEnabledWhen.HAS_VIDEO_MODALITY,
    disabledTooltip: fTranslate('text_01f526933484'),
    action: AttachmentAction.FILE_UPLOAD
  },
  {
    id: AttachmentMenuItemId.TEXT,
    label: fTranslate('text_58c701e56182'),
    icon: cFileTypeIcons.text,
    enabledWhen: AttachmentItemEnabledWhen.ALWAYS,
    action: AttachmentAction.FILE_UPLOAD
  },
  {
    id: AttachmentMenuItemId.PDF,
    label: fTranslate('text_fad030751fb1'),
    icon: cFileTypeIcons.pdf,
    enabledWhen: AttachmentItemEnabledWhen.ALWAYS,
    disabledTooltip: 'PDFs will be converted to text. Image-based PDFs may not work properly.',
    hasEnabledTooltip: true,
    action: AttachmentAction.FILE_UPLOAD
  }
];

export const aAttachmentExtraItems: AttachmentMenuItem[] = [];

export const aAttachmentPromptItems: AttachmentMenuItem[] = [
  {
    id: AttachmentMenuItemId.SYSTEM_MESSAGE,
    label: fTranslate('text_8e5a3143184d'),
    icon: MessageSquare,
    enabledWhen: AttachmentItemEnabledWhen.ALWAYS,
    hasEnabledTooltip: true,
    action: AttachmentAction.SYSTEM_PROMPT_CLICK
  },
  {
    id: AttachmentMenuItemId.MCP_PROMPT,
    label: fTranslate('text_6829fa6ca0ec'),
    icon: Zap,
    enabledWhen: AttachmentItemEnabledWhen.ALWAYS,
    action: AttachmentAction.MCP_PROMPT_CLICK,
    visibleWhen: AttachmentItemVisibleWhen.HAS_MCP_PROMPTS_SUPPORT
  }
];

export const aAttachmentMcpItems: AttachmentMenuItem[] = [
  {
    id: AttachmentMenuItemId.MCP_RESOURCES,
    label: fTranslate('text_77a814db2847'),
    icon: FolderOpen,
    enabledWhen: AttachmentItemEnabledWhen.ALWAYS,
    action: AttachmentAction.MCP_RESOURCES_CLICK,
    visibleWhen: AttachmentItemVisibleWhen.HAS_MCP_RESOURCES_SUPPORT
  }
];

export const cAttachmentTooltipText = fTranslate('text_413359146641');
