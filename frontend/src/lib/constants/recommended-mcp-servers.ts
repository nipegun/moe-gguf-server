import { fTranslate } from '../i18n';
import type { RecommendedMCPServer } from '$lib/types';

// Suggested MCP servers shown as opt-in cards in the "Add New Server" dialog.
// Rendering these cards never reaches the upstream domain - favicons come
// from local bundles in static/recommended-mcp/ and the URL is only used
// after the user clicks Add.
export const aRecommendedMcpServers: RecommendedMCPServer[] = [
  {
    id: 'exa',
    name: 'Exa',
    description: fTranslate('text_2fc097e2378b'),
    url: 'https://mcp.exa.ai/mcp',
    iconUrl: '/recommended-mcp/exa.ico'
  },
  {
    id: 'huggingface',
    name: 'Hugging Face',
    description: fTranslate('text_464508c5f0f4'),
    url: 'https://huggingface.co/mcp',
    iconUrl: '/recommended-mcp/huggingface.ico'
  },
  {
    id: 'github',
    name: 'GitHub',
    description: fTranslate('text_5a80bd7f18ef'),
    url: 'https://api.githubcopilot.com/mcp',
    iconUrlLight: '/recommended-mcp/github-light.png',
    iconUrlDark: '/recommended-mcp/github-dark.png',
    needsAuthorization: true
  },
  {
    id: 'context7',
    name: 'Context7',
    description: fTranslate('text_df8d5fe20367'),
    url: 'https://mcp.context7.com/mcp',
    iconUrl: '/recommended-mcp/context7.png'
  }
];
