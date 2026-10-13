import { fTranslate } from '../i18n';
/**
 * PDF processing utilities using PDF.js
 * Handles PDF text extraction and image conversion in the browser
 */

import { browser } from '$app/environment';
import { MimeTypeApplication, MimeTypeImage } from '$lib/enums';
import * as pdfjs from 'pdfjs-dist';

type TextContent = {
  items: Array<{ str: string }>;
};

if (browser) {
  // Import worker as text and create blob URL for inline bundling
  import('pdfjs-dist/build/pdf.worker.min.mjs?raw')
    .then((pWorkerModule) => {
      const cWorkerBlob = new Blob([pWorkerModule.default], { type: 'application/javascript' });
      pdfjs.GlobalWorkerOptions.workerSrc = URL.createObjectURL(cWorkerBlob);
    })
    .catch(() => {
      console.warn('Failed to load PDF.js worker, PDF processing may not work');
    });
}

/**
 * Convert a File object to ArrayBuffer for PDF.js processing
 * @param pFile - The PDF file to convert
 * @returns Promise resolving to the file's ArrayBuffer
 */
async function fGetFileAsBuffer(pFile: File): Promise<ArrayBuffer> {
  return new Promise((pResolve, pReject) => {
    const cReader = new FileReader();
    cReader.onload = (pEvent) => {
      if (pEvent.target?.result) {
        pResolve(pEvent.target.result as ArrayBuffer);
      } else {
        pReject(new Error(fTranslate('text_cb49d2515b19')));
      }
    };
    cReader.onerror = () => {
      pReject(new Error(fTranslate('text_cb49d2515b19')));
    };
    cReader.readAsArrayBuffer(pFile);
  });
}

/**
 * Extract text content from a PDF file
 * @param pFile - The PDF file to process
 * @returns Promise resolving to the extracted text content
 */
export async function fConvertPDFToText(pFile: File): Promise<string> {
  if (!browser) {
    throw new Error(fTranslate('text_b870294aa9b0'));
  }

  try {
    const cBuffer = await fGetFileAsBuffer(pFile);
    const cPdf = await pdfjs.getDocument({ data: cBuffer }).promise;
    const cNumPages = cPdf.numPages;

    const aTextContentPromises: Promise<TextContent>[] = [];

    for (let vI = 1; vI <= cNumPages; vI++) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      aTextContentPromises.push(cPdf.getPage(vI).then((pPage: any) => pPage.getTextContent()));
    }

    const cTextContents = await Promise.all(aTextContentPromises);
    const cTextItems = cTextContents.flatMap((pTextContent: TextContent) =>
      pTextContent.items.map((pItem) => pItem.str ?? '')
    );

    return cTextItems.join('\n');
  } catch (vError) {
    console.error('Error converting PDF to text:', vError);
    throw new Error(
      fTranslate('text_85baba7e1b0f', {
        p0: vError instanceof Error ? vError.message : 'Unknown error'
      })
    );
  }
}

/**
 * Convert PDF pages to PNG images as data URLs
 * @param pFile - The PDF file to convert
 * @param pScale - Rendering scale factor (default: 1.5)
 * @returns Promise resolving to array of PNG data URLs
 */
export async function fConvertPDFToImage(pFile: File, pScale: number = 1.5): Promise<string[]> {
  if (!browser) {
    throw new Error(fTranslate('text_b870294aa9b0'));
  }

  try {
    const cBuffer = await fGetFileAsBuffer(pFile);
    const cDoc = await pdfjs.getDocument({ data: cBuffer }).promise;
    const aPages: Promise<string>[] = [];

    for (let vI = 1; vI <= cDoc.numPages; vI++) {
      const cPage = await cDoc.getPage(vI);
      const cViewport = cPage.getViewport({ scale: pScale });
      const cCanvas = document.createElement('canvas');
      const cCtx = cCanvas.getContext('2d');

      cCanvas.width = cViewport.width;
      cCanvas.height = cViewport.height;

      if (!cCtx) {
        throw new Error(fTranslate('text_dc7810d73a65'));
      }

      const cTask = cPage.render({
        canvasContext: cCtx,
        viewport: cViewport,
        canvas: cCanvas
      });
      aPages.push(
        cTask.promise.then(() => {
          return cCanvas.toDataURL(MimeTypeImage.PNG);
        })
      );
    }

    return await Promise.all(aPages);
  } catch (vError) {
    console.error('Error converting PDF to images:', vError);
    throw new Error(
      fTranslate('text_3bda54bf78a3', {
        p0: vError instanceof Error ? vError.message : 'Unknown error'
      })
    );
  }
}

/**
 * Check if a file is a PDF based on its MIME type
 * @param pFile - The file to check
 * @returns True if the file is a PDF
 */
export function fIsPdfFile(pFile: File): boolean {
  return pFile.type === MimeTypeApplication.PDF;
}

/**
 * Check if a MIME type represents a PDF
 * @param pMimeType - The MIME type to check
 * @returns True if the MIME type is application/pdf
 */
export function fIsApplicationMimeType(pMimeType: string): boolean {
  return pMimeType === MimeTypeApplication.PDF;
}
