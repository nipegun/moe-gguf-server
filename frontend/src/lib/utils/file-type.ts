import {
  cAudioFileTypes,
  cVideoFileTypes,
  cImageFileTypes,
  cPdfFileTypes,
  cTextFileTypes
} from '$lib/constants';
import {
  FileExtensionAudio,
  FileExtensionImage,
  FileExtensionPdf,
  FileExtensionText,
  FileTypeCategory,
  MimeTypeApplication,
  MimeTypeAudio,
  MimeTypeVideo,
  MimeTypeImage,
  MimeTypeText
} from '$lib/enums';

function fNormalizeMimeType(pMimeType: string): string {
  return pMimeType.trim().toLowerCase();
}

export function fGetFileTypeCategory(pMimeType: string): FileTypeCategory | null {
  switch (fNormalizeMimeType(pMimeType)) {
    // Images
    case MimeTypeImage.JPEG:
    case MimeTypeImage.PNG:
    case MimeTypeImage.GIF:
    case MimeTypeImage.WEBP:
    case MimeTypeImage.SVG:
    case MimeTypeImage.HEIC:
    case MimeTypeImage.HEIF:
      return FileTypeCategory.IMAGE;

    // Audio
    case MimeTypeAudio.MP3_MPEG:
    case MimeTypeAudio.MP3:
    case MimeTypeAudio.MP4:
    case MimeTypeAudio.WAV:
    case MimeTypeAudio.WAVE:
    case MimeTypeAudio.X_WAV:
    case MimeTypeAudio.X_WAVE:
    case MimeTypeAudio.VND_WAVE:
    case MimeTypeAudio.X_PN_WAV:
    case MimeTypeAudio.WEBM:
    case MimeTypeAudio.WEBM_OPUS:
      return FileTypeCategory.AUDIO;

    // Video
    case MimeTypeVideo.MP4:
    case MimeTypeVideo.OGG:
      return FileTypeCategory.VIDEO;

    // PDF
    case MimeTypeApplication.PDF:
      return FileTypeCategory.PDF;

    // Text
    case MimeTypeText.PLAIN:
    case MimeTypeText.MARKDOWN:
    case MimeTypeText.ASCIIDOC:
    case MimeTypeText.JAVASCRIPT:
    case MimeTypeText.JAVASCRIPT_APP:
    case MimeTypeText.TYPESCRIPT:
    case MimeTypeText.JSX:
    case MimeTypeText.TSX:
    case MimeTypeText.CSS:
    case MimeTypeText.HTML:
    case MimeTypeText.JSON:
    case MimeTypeText.XML_TEXT:
    case MimeTypeText.XML_APP:
    case MimeTypeText.YAML_TEXT:
    case MimeTypeText.YAML_APP:
    case MimeTypeText.CSV:
    case MimeTypeText.PYTHON:
    case MimeTypeText.JAVA:
    case MimeTypeText.CPP_SRC:
    case MimeTypeText.C_SRC:
    case MimeTypeText.C_HDR:
    case MimeTypeText.PHP:
    case MimeTypeText.RUBY:
    case MimeTypeText.GO:
    case MimeTypeText.RUST:
    case MimeTypeText.SHELL:
    case MimeTypeText.BAT:
    case MimeTypeText.SQL:
    case MimeTypeText.R:
    case MimeTypeText.SCALA:
    case MimeTypeText.KOTLIN:
    case MimeTypeText.SWIFT:
    case MimeTypeText.DART:
    case MimeTypeText.VUE:
    case MimeTypeText.SVELTE:
    case MimeTypeText.LATEX:
    case MimeTypeText.BIBTEX:
    case MimeTypeText.CUDA:
    case MimeTypeText.CPP_HDR:
    case MimeTypeText.CSHARP:
    case MimeTypeText.HASKELL:
    case MimeTypeText.PROPERTIES:
    case MimeTypeText.TEX:
    case MimeTypeText.TEX_APP:
      return FileTypeCategory.TEXT;

    default:
      return null;
  }
}

export function fGetFileTypeCategoryByExtension(pFilename: string): FileTypeCategory | null {
  const cExtension = pFilename.toLowerCase().substring(pFilename.lastIndexOf('.'));

  switch (cExtension) {
    // Images
    case FileExtensionImage.JPG:
    case FileExtensionImage.JPEG:
    case FileExtensionImage.PNG:
    case FileExtensionImage.GIF:
    case FileExtensionImage.WEBP:
    case FileExtensionImage.SVG:
    case FileExtensionImage.HEIC:
    case FileExtensionImage.HEIF:
      return FileTypeCategory.IMAGE;

    // Audio
    case FileExtensionAudio.MP3:
    case FileExtensionAudio.WAV:
      return FileTypeCategory.AUDIO;

    // PDF
    case FileExtensionPdf.PDF:
      return FileTypeCategory.PDF;

    // Text
    case FileExtensionText.TXT:
    case FileExtensionText.MD:
    case FileExtensionText.ADOC:
    case FileExtensionText.JS:
    case FileExtensionText.TS:
    case FileExtensionText.JSX:
    case FileExtensionText.TSX:
    case FileExtensionText.CSS:
    case FileExtensionText.HTML:
    case FileExtensionText.HTM:
    case FileExtensionText.JSON:
    case FileExtensionText.XML:
    case FileExtensionText.YAML:
    case FileExtensionText.YML:
    case FileExtensionText.CSV:
    case FileExtensionText.LOG:
    case FileExtensionText.PY:
    case FileExtensionText.JAVA:
    case FileExtensionText.CPP:
    case FileExtensionText.C:
    case FileExtensionText.H:
    case FileExtensionText.PHP:
    case FileExtensionText.RB:
    case FileExtensionText.GO:
    case FileExtensionText.RS:
    case FileExtensionText.SH:
    case FileExtensionText.BAT:
    case FileExtensionText.SQL:
    case FileExtensionText.R:
    case FileExtensionText.SCALA:
    case FileExtensionText.KT:
    case FileExtensionText.SWIFT:
    case FileExtensionText.DART:
    case FileExtensionText.VUE:
    case FileExtensionText.SVELTE:
    case FileExtensionText.TEX:
    case FileExtensionText.BIB:
    case FileExtensionText.COMP:
    case FileExtensionText.CU:
    case FileExtensionText.CUH:
    case FileExtensionText.HPP:
    case FileExtensionText.HS:
    case FileExtensionText.PROPERTIES:
      return FileTypeCategory.TEXT;

    default:
      return null;
  }
}

export function fGetFileTypeByExtension(pFilename: string): string | null {
  const cExtension = pFilename.toLowerCase().substring(pFilename.lastIndexOf('.'));

  for (const [cKey, cType] of Object.entries(cImageFileTypes)) {
    if ((cType.extensions as readonly string[]).includes(cExtension)) {
      return `${FileTypeCategory.IMAGE}:${cKey}`;
    }
  }

  for (const [cKey, cType] of Object.entries(cAudioFileTypes)) {
    if ((cType.extensions as readonly string[]).includes(cExtension)) {
      return `${FileTypeCategory.AUDIO}:${cKey}`;
    }
  }

  for (const [cKey, cType] of Object.entries(cVideoFileTypes)) {
    if ((cType.extensions as readonly string[]).includes(cExtension)) {
      return `${FileTypeCategory.VIDEO}:${cKey}`;
    }
  }

  for (const [cKey, cType] of Object.entries(cPdfFileTypes)) {
    if ((cType.extensions as readonly string[]).includes(cExtension)) {
      return `${FileTypeCategory.PDF}:${cKey}`;
    }
  }

  for (const [cKey, cType] of Object.entries(cTextFileTypes)) {
    if ((cType.extensions as readonly string[]).includes(cExtension)) {
      return `${FileTypeCategory.TEXT}:${cKey}`;
    }
  }

  return null;
}

export function fIsFileTypeSupported(pFilename: string, pMimeType?: string): boolean {
  // Images are detected and handled separately for vision models
  if (pMimeType) {
    const cCategory = fGetFileTypeCategory(pMimeType);
    if (
      cCategory === FileTypeCategory.IMAGE ||
      cCategory === FileTypeCategory.AUDIO ||
      cCategory === FileTypeCategory.PDF
    ) {
      return true;
    }
  }

  // Check extension for known types (especially images without MIME)
  const cExtCategory = fGetFileTypeCategoryByExtension(pFilename);
  if (
    cExtCategory === FileTypeCategory.IMAGE ||
    cExtCategory === FileTypeCategory.AUDIO ||
    cExtCategory === FileTypeCategory.PDF
  ) {
    return true;
  }

  // Fallback: treat everything else as text (inclusive by default)
  return true;
}
