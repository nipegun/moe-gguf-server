<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import type { ChatAttachmentDisplayItem } from '$lib/types';
  import { FileText, Eye, Info } from '@lucide/svelte';
  import { Button } from '$lib/components/ui/button';
  import * as Alert from '$lib/components/ui/alert';
  import { SyntaxHighlightedCode } from '$lib/components/app';
  import { getLanguageFromFilename } from '$lib/utils';
  import { convertPDFToImage } from '$lib/utils/browser-only';
  import { PdfViewMode } from '$lib/enums';

  interface Props {
    currentItem: ChatAttachmentDisplayItem | null;
    displayName: string;
    displayTextContent: string | undefined;
    hasVisionModality: boolean;
    activeModelId?: string;
  }

  let {
    currentItem: vCurrentItem,
    displayName: vDisplayName,
    displayTextContent: vDisplayTextContent,
    hasVisionModality: vHasVisionModality,
    activeModelId: vActiveModelId
  }: Props = $props();

  let vPdfViewMode = $state<PdfViewMode>(PdfViewMode.PAGES);
  let vPdfImages = $state<string[]>([]);
  let vPdfImagesLoading = $state(false);
  let vPdfImagesError = $state<string | null>(null);

  let vLanguage = $derived(getLanguageFromFilename(vDisplayName));

  async function fLoadPdfImages() {
    if (vPdfImages.length > 0 || vPdfImagesLoading || !vCurrentItem) return;

    vPdfImagesLoading = true;
    vPdfImagesError = null;

    try {
      let vFile: File | null = null;

      if (vCurrentItem.uploadedFile?.file) {
        vFile = vCurrentItem.uploadedFile.file;
      } else if (vCurrentItem.attachment) {
        // Check if we have pre-processed images
        if (
          'images' in vCurrentItem.attachment &&
          vCurrentItem.attachment.images &&
          Array.isArray(vCurrentItem.attachment.images) &&
          vCurrentItem.attachment.images.length > 0
        ) {
          vPdfImages = vCurrentItem.attachment.images;
          return;
        }

        // Convert base64 back to File for processing
        if ('base64Data' in vCurrentItem.attachment && vCurrentItem.attachment.base64Data) {
          const cBase64Data = vCurrentItem.attachment.base64Data;
          const cByteCharacters = atob(cBase64Data);
          const cByteNumbers = new Array(cByteCharacters.length);
          for (let vI = 0; vI < cByteCharacters.length; vI++) {
            cByteNumbers[vI] = cByteCharacters.charCodeAt(vI);
          }
          const cByteArray = new Uint8Array(cByteNumbers);
          vFile = new File([cByteArray], vDisplayName, { type: 'application/pdf' });
        }
      }

      if (vFile) {
        vPdfImages = await convertPDFToImage(vFile);
      } else {
        throw new Error(fTranslate('text_4291e9544b36'));
      }
    } catch (vError) {
      vPdfImagesError = vError instanceof Error ? vError.message : fTranslate('text_e6728ce51717');
    } finally {
      vPdfImagesLoading = false;
    }
  }

  $effect(() => {
    if (vPdfViewMode === PdfViewMode.PAGES) {
      fLoadPdfImages();
    }
  });
</script>

<div class="mb-4 flex items-center justify-end gap-2">
  <Button
    variant={vPdfViewMode === PdfViewMode.TEXT ? 'default' : 'outline'}
    size="sm"
    onclick={() => (vPdfViewMode = PdfViewMode.TEXT)}
    disabled={vPdfImagesLoading}
  >
    <FileText class="mr-1 {cIconClassDefault}" />
    {fTranslate('text_71988c4d8e08')}
  </Button>

  <Button
    variant={vPdfViewMode === PdfViewMode.PAGES ? 'default' : 'outline'}
    size="sm"
    onclick={() => (vPdfViewMode = PdfViewMode.PAGES)}
    disabled={vPdfImagesLoading}
  >
    {#if vPdfImagesLoading}
      <div
        class="mr-1 {cIconClassDefault} animate-spin rounded-full border-2 border-current border-t-transparent"
      ></div>
    {:else}
      <Eye class="mr-1 {cIconClassDefault}" />
    {/if}
    {fTranslate('text_9046da16aea9')}
  </Button>
</div>

{#if !vHasVisionModality && vActiveModelId && vCurrentItem}
  <Alert.Root class="mb-4 max-w-4xl">
    <Info class={cIconClassDefault} />
    <Alert.Title>{fTranslate('text_3673f9c181ba')}</Alert.Title>
    <Alert.Description>
      <span class="inline-flex">
        {fTranslate('text_7d2b2a1b79ad')}
        <!-- svelte-ignore a11y_click_events_have_key_events -->
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <span
          class="mx-1 cursor-pointer underline"
          onclick={() => (vPdfViewMode = PdfViewMode.TEXT)}
        >
          {fTranslate('text_982d9e3eb996')}
        </span>
        {fTranslate('text_030cfd60beee')}
      </span>
    </Alert.Description>
  </Alert.Root>
{/if}

{#if vPdfImagesLoading}
  <div class="flex flex-1 items-center justify-center p-8">
    <div class="text-center">
      <div
        class="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-4 border-white border-t-transparent"
      ></div>
      <p class="text-white/70">{fTranslate('text_cef551a7cb0f')}</p>
    </div>
  </div>
{:else if vPdfImagesError}
  <div class="flex flex-1 items-center justify-center p-8">
    <div class="text-center">
      <FileText class="mx-auto mb-4 h-16 w-16 text-white/50" />
      <p class="mb-4 text-white/70">{fTranslate('text_e6728ce51717')}</p>
      <p class="text-sm text-white/50">{vPdfImagesError}</p>
    </div>
  </div>
{:else if vPdfImages.length > 0}
  {#each vPdfImages as image, index (image)}
    <p class="mb-2 text-sm text-white/50">{fTranslate('text_0a30a815d67d')} {index + 1}</p>
    <img
      src={image}
      alt={fTranslate('text_6ff607e169d0', { p0: index + 1 })}
      class="mx-auto max-w-[85vw] rounded-lg shadow-lg"
    />
    <div class="h-4"></div>
  {/each}
{:else}
  <div class="flex flex-1 items-center justify-center p-8">
    <div class="text-center">
      <FileText class="mx-auto mb-4 h-16 w-16 text-white/50" />
      <p class="text-white/70">{fTranslate('text_35ccf4acfc40')}</p>
    </div>
  </div>
{/if}

{#if vPdfViewMode === PdfViewMode.TEXT && vDisplayTextContent}
  <div class="px-4 pb-4">
    <SyntaxHighlightedCode
      class="max-w-4xl"
      code={vDisplayTextContent}
      language={vLanguage}
      maxHeight="none"
    />
  </div>
{/if}
