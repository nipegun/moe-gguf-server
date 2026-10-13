import { fTranslate } from '../i18n';
import { ColorMode } from '$lib/enums/ui.enums';
import { SettingsFieldType } from '$lib/enums/settings.enums';
import { SyncableParameterType } from '$lib/enums';
import {
  Funnel,
  AlertTriangle,
  Code,
  Monitor,
  ListRestart,
  Sliders,
  PencilRuler,
  Database,
  Monitor as MonitorIcon,
  Sun,
  Moon
} from '@lucide/svelte';
import type { Component } from 'svelte';
import type {
  SettingsConfigValue,
  SyncableParameter,
  SettingsEntry,
  SettingsSectionTitle,
  SettingsSectionEntry,
  SettingsSection
} from '$lib/types';
import { cCliFlags, cDefaultMcpConfig } from '$lib/constants';
import { cSettingsKeys } from './settings-keys';
import { cRoutes, cSettingsSectionSlugs } from './routes';
import { cTitleGeneration } from './title-generation';

export const cSettingsSectionTitles = {
  GENERAL: fTranslate('text_c910d474dcd7'),
  DISPLAY: fTranslate('text_34e108c0896d'),
  SAMPLING: fTranslate('text_e41f08d35dd1'),
  PENALTIES: fTranslate('text_444206375e2a'),
  AGENTIC: fTranslate('text_7b1f65c25801'),
  TOOLS: fTranslate('text_ea93d6a262ec'),
  IMPORT_EXPORT: fTranslate('text_42191ef46da3'),
  DEVELOPER: fTranslate('text_3fb7b39416f1')
} as const;

const aStandaloneSections: { title: SettingsSectionTitle; slug: string; icon: Component }[] = [
  { title: cSettingsSectionTitles.TOOLS, slug: cSettingsSectionSlugs.TOOLS, icon: PencilRuler },
  {
    title: cSettingsSectionTitles.IMPORT_EXPORT,
    slug: cSettingsSectionSlugs.IMPORT_EXPORT,
    icon: Database
  }
];

const aColorModeOptions: Array<{ value: string; label: string; icon: Component }> = [
  { value: ColorMode.SYSTEM, label: fTranslate('text_6725e7bbcd28'), icon: MonitorIcon },
  { value: ColorMode.LIGHT, label: fTranslate('text_dbcd5e7bb7a0'), icon: Sun },
  { value: ColorMode.DARK, label: fTranslate('text_60acc53f13a5'), icon: Moon }
];

// Shared options for the title-generation radio group. Both paired registry entries
// (USE_FIRST_LINE, USE_LLM) reference this list so labels stay in lockstep.
const aTitleGenerationRadioOptions: Array<{
  value: string;
  label: string;
  key: string;
  isExperimental?: boolean;
}> = [
  {
    value: 'firstLine',
    label: fTranslate('text_d260b3167eca'),
    key: cSettingsKeys.TITLE_GENERATION_USE_FIRST_LINE
  },
  {
    value: 'llm',
    label: fTranslate('text_979477c8d433'),
    key: cSettingsKeys.TITLE_GENERATION_USE_LLM,
    isExperimental: true
  }
];

// Common shape for the conversation title radio entry.
const cTitleGenerationBase = {
  type: SettingsFieldType.RADIO,
  section: cSettingsSectionSlugs.GENERAL,
  radioOptions: aTitleGenerationRadioOptions
} as const;

const dSettingsRegistry: Record<string, SettingsSectionEntry> = {
  [cSettingsSectionSlugs.GENERAL]: {
    title: cSettingsSectionTitles.GENERAL,
    slug: cSettingsSectionSlugs.GENERAL,
    icon: Sliders,
    settings: [
      {
        key: cSettingsKeys.RESPOND_IN_INTERFACE_LANGUAGE,
        label: fTranslate('responseLanguage.label'),
        help: fTranslate('responseLanguage.help'),
        defaultValue: false,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.GENERAL
      },
      {
        key: cSettingsKeys.THEME,
        label: fTranslate('text_efb52e7172b7'),
        help: fTranslate('text_d8c2701b0dce'),
        defaultValue: ColorMode.SYSTEM,
        type: SettingsFieldType.SELECT,
        section: cSettingsSectionSlugs.GENERAL,
        options: aColorModeOptions
      },
      {
        key: cSettingsKeys.API_KEY,
        label: fTranslate('text_23189d55f697'),
        help: fTranslate('text_590ca5f81094', { p0: cCliFlags.API_KEY }),
        defaultValue: '',
        type: SettingsFieldType.INPUT,
        section: cSettingsSectionSlugs.GENERAL
      },
      {
        key: cSettingsKeys.SYSTEM_MESSAGE,
        label: fTranslate('text_8e5a3143184d'),
        help: fTranslate('text_fbd01c02e202'),
        defaultValue: '',
        type: SettingsFieldType.TEXTAREA,
        section: cSettingsSectionSlugs.GENERAL
      },
      {
        key: cSettingsKeys.PASTE_LONG_TEXT_TO_FILE_LEN,
        label: fTranslate('text_401b499e8e8f'),
        help: fTranslate('text_d50c4251977a'),
        defaultValue: 2500,
        type: SettingsFieldType.INPUT,
        section: cSettingsSectionSlugs.GENERAL
      },
      {
        key: cSettingsKeys.SEND_ON_ENTER,
        label: fTranslate('text_7240e16f8c6b'),
        help: fTranslate('text_819c91d143be'),
        defaultValue: true,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.GENERAL
      },
      {
        key: cSettingsKeys.AUTO_MIC_ON_EMPTY,
        label: fTranslate('text_8666460b20dc'),
        help: fTranslate('text_569a6b44e53d'),
        defaultValue: false,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.GENERAL,
        isExperimental: true
      },
      {
        key: cSettingsKeys.ENABLE_CONTINUE_GENERATION,
        label: fTranslate('text_086d634416fb'),
        help: fTranslate('text_c25279367499'),
        defaultValue: false,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.GENERAL,
        isExperimental: true
      },
      {
        ...cTitleGenerationBase,
        key: cSettingsKeys.TITLE_GENERATION_USE_FIRST_LINE,
        label: fTranslate('text_5b5cd3e853ae'),
        help: fTranslate('text_2798ae6854ed'),
        defaultValue: true
      },
      {
        key: cSettingsKeys.TITLE_GENERATION_PROMPT,
        label: fTranslate('text_20e4c1fbfbc0'),
        help: fTranslate('text_d737cb824eb4'),
        defaultValue: cTitleGeneration.DEFAULT_PROMPT,
        type: SettingsFieldType.TEXTAREA,
        section: cSettingsSectionSlugs.GENERAL,
        dependsOn: cSettingsKeys.TITLE_GENERATION_USE_LLM
      },
      {
        key: cSettingsKeys.COPY_TEXT_ATTACHMENTS_AS_PLAIN_TEXT,
        label: fTranslate('text_ba488dcfbd3e'),
        help: fTranslate('text_bac76e8ab9fb'),
        defaultValue: false,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.GENERAL
      },
      {
        key: cSettingsKeys.PDF_AS_IMAGE,
        label: fTranslate('text_9ec72f537886'),
        help: fTranslate('text_96fe65afdb37'),
        defaultValue: false,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.GENERAL
      },
      {
        key: cSettingsKeys.MAX_IMAGE_RESOLUTION,
        label: fTranslate('text_2954877fdbea'),
        help: fTranslate('text_a9e27241ec14'),
        defaultValue: 0,
        type: SettingsFieldType.INPUT,
        section: cSettingsSectionSlugs.GENERAL
      }
    ]
  },
  [cSettingsSectionSlugs.DISPLAY]: {
    title: cSettingsSectionTitles.DISPLAY,
    slug: cSettingsSectionSlugs.DISPLAY,
    icon: Monitor,
    settings: [
      {
        key: cSettingsKeys.SHOW_MESSAGE_STATS,
        label: fTranslate('text_58767901d07b'),
        help: fTranslate('text_1e886427b492'),
        defaultValue: false,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.DISPLAY
      },
      {
        key: cSettingsKeys.SHOW_AGENTIC_TURN_STATS,
        label: fTranslate('text_795b48b44734'),
        help: fTranslate('text_d531a39769ee'),
        defaultValue: false,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.DISPLAY,
        dependsOn: cSettingsKeys.SHOW_MESSAGE_STATS
      },
      {
        key: cSettingsKeys.SHOW_THOUGHT_IN_PROGRESS,
        label: fTranslate('text_c699c01d5c4c'),
        help: fTranslate('text_a51f7f782ae8'),
        defaultValue: true,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.DISPLAY
      },
      {
        key: cSettingsKeys.ALWAYS_SHOW_TOOL_CALL_CONTENT,
        label: fTranslate('text_9ae2d6e1f7ed'),
        help: fTranslate('text_0a711551837d'),
        defaultValue: false,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.DISPLAY
      },
      {
        key: cSettingsKeys.EXPAND_LOCAL_SHELL_COMMANDS,
        label: fTranslate('text_ba1c2d8d0e4f'),
        help: fTranslate('text_0fd2633d8398'),
        defaultValue: false,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.DISPLAY
      },
      {
        key: cSettingsKeys.RENDER_USER_CONTENT_AS_MARKDOWN,
        label: fTranslate('text_11910b83c707'),
        help: fTranslate('text_c3af243c77aa'),
        defaultValue: false,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.DISPLAY
      },
      {
        key: cSettingsKeys.RENDER_THINKING_AS_MARKDOWN,
        label: fTranslate('text_047a9804886c'),
        help: fTranslate('text_3b2811cf8cbc'),
        defaultValue: true,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.DISPLAY
      },
      {
        key: cSettingsKeys.FULL_HEIGHT_CODE_BLOCKS,
        label: fTranslate('text_ee7633d878e8'),
        help: fTranslate('text_7f545155cd7c'),
        defaultValue: false,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.DISPLAY
      },
      {
        key: cSettingsKeys.DISABLE_AUTO_SCROLL,
        label: fTranslate('text_1d17af5ec5f4'),
        help: fTranslate('text_3253a89f0cd9'),
        defaultValue: false,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.DISPLAY
      },
      {
        key: cSettingsKeys.ALWAYS_SHOW_SIDEBAR_ON_DESKTOP,
        label: fTranslate('text_87776621d9c2'),
        help: fTranslate('text_2d79bf0c56fd'),
        defaultValue: false,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.DISPLAY
      },
      {
        key: cSettingsKeys.SHOW_RAW_MODEL_NAMES,
        label: fTranslate('text_8dfac9650c4a'),
        help: fTranslate('text_980c9fab0818'),
        defaultValue: false,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.DISPLAY
      },
      {
        key: cSettingsKeys.SHOW_MODEL_QUANTIZATION,
        label: fTranslate('text_b1c372445053'),
        help: fTranslate('text_49aedab8ac92'),
        defaultValue: true,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.DISPLAY
      },
      {
        key: cSettingsKeys.SHOW_MODEL_TAGS,
        label: fTranslate('text_68d073cd1d69'),
        help: fTranslate('text_0ed5075d0dba'),
        defaultValue: true,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.DISPLAY
      },
      {
        key: cSettingsKeys.SHOW_BUILD_VERSION,
        label: fTranslate('text_4d8d312c1cf1'),
        help: fTranslate('text_216237a41c51'),
        defaultValue: false,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.DISPLAY
      },
      {
        key: cSettingsKeys.SHOW_STATUS_BAR,
        label: fTranslate('text_e9736daa7c32'),
        help: fTranslate('text_a1e7c1ca856c'),
        defaultValue: true,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.DISPLAY
      }
    ]
  },
  [cSettingsSectionSlugs.SAMPLING]: {
    title: cSettingsSectionTitles.SAMPLING,
    slug: cSettingsSectionSlugs.SAMPLING,
    icon: Funnel,
    settings: [
      {
        key: cSettingsKeys.TEMPERATURE,
        label: fTranslate('text_b958ce8b871a'),
        help: fTranslate('text_4a6ef954759e'),
        defaultValue: undefined,
        type: SettingsFieldType.INPUT,
        section: cSettingsSectionSlugs.SAMPLING,
        sync: {
          serverKey: cSettingsKeys.TEMPERATURE,
          paramType: SyncableParameterType.NUMBER
        }
      },
      {
        key: cSettingsKeys.DYNATEMP_RANGE,
        label: fTranslate('text_262220def130'),
        help: fTranslate('text_ff2049bb8185'),
        defaultValue: undefined,
        type: SettingsFieldType.INPUT,
        section: cSettingsSectionSlugs.SAMPLING,
        sync: {
          serverKey: cSettingsKeys.DYNATEMP_RANGE,
          paramType: SyncableParameterType.NUMBER
        }
      },
      {
        key: cSettingsKeys.DYNATEMP_EXPONENT,
        label: fTranslate('text_e7499a75dbee'),
        help: fTranslate('text_ed4443b1e4aa'),
        defaultValue: undefined,
        type: SettingsFieldType.INPUT,
        section: cSettingsSectionSlugs.SAMPLING,
        sync: {
          serverKey: cSettingsKeys.DYNATEMP_EXPONENT,
          paramType: SyncableParameterType.NUMBER
        }
      },
      {
        key: cSettingsKeys.TOP_K,
        label: fTranslate('text_74fc1150e902'),
        help: fTranslate('text_8dbff0f8b0d9'),
        defaultValue: undefined,
        type: SettingsFieldType.INPUT,
        section: cSettingsSectionSlugs.SAMPLING,
        sync: { serverKey: cSettingsKeys.TOP_K, paramType: SyncableParameterType.NUMBER }
      },
      {
        key: cSettingsKeys.TOP_P,
        label: fTranslate('text_c7a8872a31da'),
        help: fTranslate('text_c42b88da2f69'),
        defaultValue: undefined,
        type: SettingsFieldType.INPUT,
        section: cSettingsSectionSlugs.SAMPLING,
        sync: { serverKey: cSettingsKeys.TOP_P, paramType: SyncableParameterType.NUMBER }
      },
      {
        key: cSettingsKeys.MIN_P,
        label: fTranslate('text_b0815a3244ed'),
        help: fTranslate('text_bbd244420e7b'),
        defaultValue: undefined,
        type: SettingsFieldType.INPUT,
        section: cSettingsSectionSlugs.SAMPLING,
        sync: { serverKey: cSettingsKeys.MIN_P, paramType: SyncableParameterType.NUMBER }
      },
      {
        key: cSettingsKeys.XTC_PROBABILITY,
        label: fTranslate('text_f137399227a7'),
        help: fTranslate('text_c718b5a0c65c'),
        defaultValue: undefined,
        type: SettingsFieldType.INPUT,
        section: cSettingsSectionSlugs.SAMPLING,
        sync: {
          serverKey: cSettingsKeys.XTC_PROBABILITY,
          paramType: SyncableParameterType.NUMBER
        }
      },
      {
        key: cSettingsKeys.XTC_THRESHOLD,
        label: fTranslate('text_d4ba9412017a'),
        help: fTranslate('text_48793e125258'),
        defaultValue: undefined,
        type: SettingsFieldType.INPUT,
        section: cSettingsSectionSlugs.SAMPLING,
        sync: {
          serverKey: cSettingsKeys.XTC_THRESHOLD,
          paramType: SyncableParameterType.NUMBER
        }
      },
      {
        key: cSettingsKeys.TYP_P,
        label: fTranslate('text_866fba8d4873'),
        help: fTranslate('text_2a4f56aa43d8'),
        defaultValue: undefined,
        type: SettingsFieldType.INPUT,
        section: cSettingsSectionSlugs.SAMPLING,
        sync: { serverKey: cSettingsKeys.TYP_P, paramType: SyncableParameterType.NUMBER }
      },
      {
        key: cSettingsKeys.MAX_TOKENS,
        label: fTranslate('text_409e75c09fcc'),
        help: fTranslate('text_0afe6457ffaf'),
        defaultValue: undefined,
        type: SettingsFieldType.INPUT,
        section: cSettingsSectionSlugs.SAMPLING,
        sync: {
          serverKey: cSettingsKeys.MAX_TOKENS,
          paramType: SyncableParameterType.NUMBER
        }
      },
      {
        key: cSettingsKeys.SAMPLERS,
        label: fTranslate('text_1c9adac18d8e'),
        help: fTranslate('text_0527a017301b'),
        defaultValue: '',
        type: SettingsFieldType.INPUT,
        section: cSettingsSectionSlugs.SAMPLING,
        sync: { serverKey: cSettingsKeys.SAMPLERS, paramType: SyncableParameterType.STRING }
      },
      {
        key: cSettingsKeys.BACKEND_SAMPLING,
        label: fTranslate('text_20d1867091ca'),
        help: fTranslate('text_d18beca92501'),
        defaultValue: false,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.SAMPLING
      }
    ]
  },
  [cSettingsSectionSlugs.PENALTIES]: {
    title: cSettingsSectionTitles.PENALTIES,
    slug: cSettingsSectionSlugs.PENALTIES,
    icon: AlertTriangle,
    settings: [
      {
        key: cSettingsKeys.REPEAT_LAST_N,
        label: fTranslate('text_c917bb45649a'),
        help: fTranslate('text_ea4a39630c59'),
        defaultValue: undefined,
        type: SettingsFieldType.INPUT,
        section: cSettingsSectionSlugs.PENALTIES,
        sync: {
          serverKey: cSettingsKeys.REPEAT_LAST_N,
          paramType: SyncableParameterType.NUMBER
        }
      },
      {
        key: cSettingsKeys.REPEAT_PENALTY,
        label: fTranslate('text_bbbc898044f9'),
        help: fTranslate('text_df8fffc1e7e8'),
        defaultValue: undefined,
        type: SettingsFieldType.INPUT,
        section: cSettingsSectionSlugs.PENALTIES,
        sync: {
          serverKey: cSettingsKeys.REPEAT_PENALTY,
          paramType: SyncableParameterType.NUMBER
        }
      },
      {
        key: cSettingsKeys.PRESENCE_PENALTY,
        label: fTranslate('text_5891319b9abe'),
        help: fTranslate('text_1f5fec90f468'),
        defaultValue: undefined,
        type: SettingsFieldType.INPUT,
        section: cSettingsSectionSlugs.PENALTIES,
        sync: {
          serverKey: cSettingsKeys.PRESENCE_PENALTY,
          paramType: SyncableParameterType.NUMBER
        }
      },
      {
        key: cSettingsKeys.FREQUENCY_PENALTY,
        label: fTranslate('text_7f9413709c95'),
        help: fTranslate('text_cc7e2f1ca212'),
        defaultValue: undefined,
        type: SettingsFieldType.INPUT,
        section: cSettingsSectionSlugs.PENALTIES,
        sync: {
          serverKey: cSettingsKeys.FREQUENCY_PENALTY,
          paramType: SyncableParameterType.NUMBER
        }
      },
      {
        key: cSettingsKeys.DRY_MULTIPLIER,
        label: fTranslate('text_a473fa39828e'),
        help: fTranslate('text_e52fc11a5708'),
        defaultValue: undefined,
        type: SettingsFieldType.INPUT,
        section: cSettingsSectionSlugs.PENALTIES,
        sync: {
          serverKey: cSettingsKeys.DRY_MULTIPLIER,
          paramType: SyncableParameterType.NUMBER
        }
      },
      {
        key: cSettingsKeys.DRY_BASE,
        label: fTranslate('text_f982fb90c90b'),
        help: fTranslate('text_29ed70517b86'),
        defaultValue: undefined,
        type: SettingsFieldType.INPUT,
        section: cSettingsSectionSlugs.PENALTIES,
        sync: { serverKey: cSettingsKeys.DRY_BASE, paramType: SyncableParameterType.NUMBER }
      },
      {
        key: cSettingsKeys.DRY_ALLOWED_LENGTH,
        label: fTranslate('text_88e074dbde59'),
        help: fTranslate('text_06ba2e536c75'),
        defaultValue: undefined,
        type: SettingsFieldType.INPUT,
        section: cSettingsSectionSlugs.PENALTIES,
        sync: {
          serverKey: cSettingsKeys.DRY_ALLOWED_LENGTH,
          paramType: SyncableParameterType.NUMBER
        }
      },
      {
        key: cSettingsKeys.DRY_PENALTY_LAST_N,
        label: fTranslate('text_c8ab732e8d70'),
        help: fTranslate('text_e804042a5e31'),
        defaultValue: undefined,
        type: SettingsFieldType.INPUT,
        section: cSettingsSectionSlugs.PENALTIES,
        sync: {
          serverKey: cSettingsKeys.DRY_PENALTY_LAST_N,
          paramType: SyncableParameterType.NUMBER
        }
      }
    ]
  },
  [cSettingsSectionSlugs.AGENTIC]: {
    title: cSettingsSectionTitles.AGENTIC,
    slug: cSettingsSectionSlugs.AGENTIC,
    icon: ListRestart,
    settings: [
      {
        key: cSettingsKeys.AGENTIC_MAX_TURNS,
        label: fTranslate('text_7e334c6eec91'),
        help: fTranslate('text_f8db79fce786'),
        defaultValue: 50,
        type: SettingsFieldType.INPUT,
        section: cSettingsSectionSlugs.AGENTIC,
        isPositiveInteger: true
      },
      {
        key: cSettingsKeys.MCP_REQUEST_TIMEOUT_SECONDS,
        label: fTranslate('text_ce1c49ac8a96'),
        help: fTranslate('text_eb6b3380faf2'),
        defaultValue: cDefaultMcpConfig.requestTimeoutSeconds,
        type: SettingsFieldType.INPUT,
        section: cSettingsSectionSlugs.AGENTIC,
        isPositiveInteger: true
      }
    ]
  },
  [cSettingsSectionSlugs.DEVELOPER]: {
    title: cSettingsSectionTitles.DEVELOPER,
    slug: cSettingsSectionSlugs.DEVELOPER,
    icon: Code,
    settings: [
      {
        key: cSettingsKeys.PRE_ENCODE_CONVERSATION,
        label: fTranslate('text_642329b42435'),
        help: fTranslate('text_b59a8e16b7c0'),
        defaultValue: true,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.DEVELOPER
      },
      {
        key: cSettingsKeys.DISABLE_REASONING_PARSING,
        label: fTranslate('text_70edf5da9d33'),
        help: fTranslate('text_da5c0c85580d'),
        defaultValue: false,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.DEVELOPER
      },
      {
        key: cSettingsKeys.EXCLUDE_REASONING_FROM_CONTEXT,
        label: fTranslate('text_1f3579b2968e'),
        help: fTranslate('text_6c6741612135'),
        defaultValue: false,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.DEVELOPER
      },
      {
        key: cSettingsKeys.SHOW_RAW_OUTPUT_SWITCH,
        label: fTranslate('text_922618c6a92e'),
        help: fTranslate('text_2da37fca08fe'),
        defaultValue: false,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.DEVELOPER
      },
      {
        key: cSettingsKeys.JS_SANDBOX_ENABLED,
        label: fTranslate('text_085046e2a92a'),
        help: fTranslate('text_328ff7861fd0'),
        defaultValue: false,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.DEVELOPER
      },
      {
        key: cSettingsKeys.SYMBOLIC_MATH_ENABLED,
        label: fTranslate('text_072ccd309854'),
        help: fTranslate('text_03344003f473'),
        defaultValue: false,
        type: SettingsFieldType.CHECKBOX,
        section: cSettingsSectionSlugs.DEVELOPER,
        dependsOn: cSettingsKeys.JS_SANDBOX_ENABLED
      },
      {
        key: cSettingsKeys.CUSTOM_JSON,
        label: fTranslate('text_4b225723787e'),
        help: fTranslate('text_c0eecd6c7ea5'),
        defaultValue: '',
        type: SettingsFieldType.TEXTAREA,
        section: cSettingsSectionSlugs.DEVELOPER
      },
      {
        key: cSettingsKeys.CUSTOM_CSS,
        label: fTranslate('text_9d3b9a0e78ea'),
        help: fTranslate('text_eddb4b393605'),
        defaultValue: '',
        type: SettingsFieldType.TEXTAREA,
        section: cSettingsSectionSlugs.DEVELOPER
      }
    ]
  }
} as const;

const aNonUiSettings: SettingsEntry[] = [
  {
    key: cSettingsKeys.SHOW_SYSTEM_MESSAGE,
    label: fTranslate('text_b710ee80506f'),
    help: fTranslate('text_8cae084b4d35'),
    defaultValue: true,
    type: SettingsFieldType.CHECKBOX
  },
  {
    key: cSettingsKeys.MCP_SERVERS,
    label: fTranslate('text_22a7559f09bf'),
    help: fTranslate('text_7b544c202be9'),
    defaultValue: '[]',
    type: SettingsFieldType.INPUT
  },
  {
    key: cSettingsKeys.TITLE_GENERATION_USE_LLM,
    label: fTranslate('text_979477c8d433'),
    help: fTranslate('text_0e24ac7791c3'),
    defaultValue: false,
    type: SettingsFieldType.CHECKBOX
  }
  // {
  //   key: SETTINGS_KEYS.PY_INTERPRETER_ENABLED,
  //   label: 'Python interpreter enabled',
  //   help: 'Enable Python interpreter using Pyodide. Allows running Python code in markdown code blocks.',
  //   defaultValue: false,
  //   type: SettingsFieldType.CHECKBOX,
  //   isExperimental: true,
  //
  // }
];

function fGetAllSettings(): SettingsEntry[] {
  const aResult: SettingsEntry[] = [];
  for (const cSection of Object.values(dSettingsRegistry)) {
    aResult.push(...cSection.settings);
  }
  aResult.push(...aNonUiSettings);
  return aResult;
}

/** Flat config object (kept by the server in ~/.mgs/config/ui-settings.json, without the API key). */
export const dSettingConfigDefault: Record<string, SettingsConfigValue> = Object.fromEntries(
  fGetAllSettings().map((pS) => [pS.key, pS.defaultValue])
) as Record<string, SettingsConfigValue>;

/** Help text for every setting (including non-UI). */
export const dSettingConfigInfo: Record<string, string> = Object.fromEntries(
  fGetAllSettings().map((pS) => [pS.key, pS.help])
) as Record<string, string>;

/** Theme select options. */
export const cSettingsColorModesConfig = aColorModeOptions;

/** Sidebar sections + field configs (as consumed by UI). */
export const aSettingsChatSections: SettingsSection[] = [
  ...Object.values(dSettingsRegistry).map((pSection) => ({
    title: pSection.title,
    slug: pSection.slug,
    icon: pSection.icon,
    fields: pSection.settings.map((pS) => ({
      key: pS.key,
      label: pS.label,
      type: pS.type,
      isExperimental: pS.isExperimental,
      isPositiveInteger: pS.isPositiveInteger,
      dependsOn: pS.dependsOn,
      help: pS.help,
      options: pS.options,
      radioOptions: pS.radioOptions
    }))
  })),
  ...aStandaloneSections
];

/** INPUT-type settings whose value is a number. */
export const cNumericFields = fGetAllSettings()
  .filter((pS) => pS.type === SettingsFieldType.INPUT && typeof pS.defaultValue !== 'string')
  .map((pS) => pS.key) as readonly string[];

/** Numeric fields clamped to ≥ 1 and rounded. */
export const cPositiveIntegerFields = fGetAllSettings()
  .filter((pS) => pS.isPositiveInteger)
  .map((pS) => pS.key) as readonly string[];

/** Derived for the parameter sync service. */
export const aSyncableParameters: SyncableParameter[] = fGetAllSettings()
  .filter((pS) => pS.sync !== undefined)
  .map((pS) => ({
    key: pS.key,
    serverKey: pS.sync!.serverKey,
    type: pS.sync!.paramType,
    canSync: true
  }));

export const cSettingsFallbackExitRoute = cRoutes.START;
