import { fTranslate } from '../i18n';
import { MimeTypeImage } from '$lib/enums';

// File extension patterns for resource type detection
export const cImageFileExtensionRegex = /\.(png|jpg|jpeg|gif|svg|webp)$/i;
export const cCodeFileExtensionRegex =
  /\.(js|ts|json|yaml|yml|xml|html|css|py|rs|go|java|cpp|c|h|rb|sh|toml)$/i;
export const cTextFileExtensionRegex = /\.(txt|md|log)$/i;

// URI protocol prefix pattern
export const cProtocolPrefixRegex = /^[a-z]+:\/\//;

// File extension regex for display name extraction
export const cFileExtensionRegex = /\.[^.]+$/;

// Separator regex for splitting display names (kebab-case/snake_case)
export const cDisplayNameSeparatorRegex = /[-_]/;

// Regex for matching base64-encoded data URIs
export const cDataUriBase64Regex = /^data:([^;]+);base64,([A-Za-z0-9+/]+=*)$/;

// Prefix for MCP attachment filenames
export const cMcpAttachmentNamePrefix = 'mcp-attachment';

// Prefix for MCP resource attachment IDs
export const cMcpResourceAttachmentIdPrefix = 'res';

// Default file extension for unknown image types
export const cDefaultImageExtension = 'img';

// Default filename for resource content downloads
export const cDefaultResourceFilename = 'resource.txt';

// Path separator for resource URI parsing
export const cPathSeparator = '/';

// Separator for joining text content from multiple resource parts
export const cResourceTextContentSeparator = '\n\n';

// Fallback text for unknown content types
export const cResourceUnknownType = 'unknown type';

// Label prefix for binary blob content
export const cBinaryContentLabel = fTranslate('text_5231884069c6');

/**
 * Mapping from image MIME types to file extensions.
 * Used for generating attachment filenames from MIME types.
 */
export const dImageMimeToExtension: Record<string, string> = {
  [MimeTypeImage.JPEG]: 'jpg',
  [MimeTypeImage.JPG]: 'jpg',
  [MimeTypeImage.PNG]: 'png',
  [MimeTypeImage.GIF]: 'gif',
  [MimeTypeImage.WEBP]: 'webp'
} as const;
