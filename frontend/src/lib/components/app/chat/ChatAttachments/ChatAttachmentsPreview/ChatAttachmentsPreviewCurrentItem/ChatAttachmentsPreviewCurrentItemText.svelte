<script lang="ts">
  import { SyntaxHighlightedCode } from '$lib/components/app';

  interface Props {
    displayTextContent: string | undefined;
    language: string;
  }

  let { displayTextContent: vDisplayTextContent, language: vLanguage }: Props = $props();
</script>

{#if vDisplayTextContent}
  <div class="px-4 pb-4">
    <SyntaxHighlightedCode
      class="max-w-4xl"
      code={vDisplayTextContent}
      language={vLanguage}
      maxHeight="none"
    />
  </div>
{/if}
