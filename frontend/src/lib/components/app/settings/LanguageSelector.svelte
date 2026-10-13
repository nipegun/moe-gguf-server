<script lang="ts">
  import { aLanguages, fGetLanguage, fSetLanguage, fTranslate } from '$lib/i18n';
</script>

<div class="language-setting">
  <label for="language-selector">{fTranslate('language')}</label>
  <select
    id="language-selector"
    value={fGetLanguage()}
    onchange={(pEvent) => fSetLanguage(pEvent.currentTarget.value)}
  >
    {#each aLanguages as vLanguage}
      <option value={vLanguage} dir="ltr">{vLanguage} · {fTranslate(`language.${vLanguage}`)}</option>
    {/each}
  </select>
  <p>{fTranslate('languageReload')}</p>
</div>

<style>
  .language-setting {
    display: grid;
    gap: 0.5rem;
    padding: 1rem;
    max-width: 100%;
  }
  select {
    min-width: 0;
    width: 100%;
    max-width: 24rem;
    color: var(--foreground);
    background: var(--background);
    border: 1px solid var(--border);
    border-radius: 0.5rem;
  }
  p {
    font-size: 0.875rem;
    overflow-wrap: anywhere;
  }
</style>
