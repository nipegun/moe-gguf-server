<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { RotateCcw, FlaskConical } from '@lucide/svelte';
  import { Checkbox } from '$lib/components/ui/checkbox';
  import { Input } from '$lib/components/ui/input';
  import Label from '$lib/components/ui/label/label.svelte';
  import * as RadioGroup from '$lib/components/ui/radio-group';
  import * as Select from '$lib/components/ui/select';
  import { Textarea } from '$lib/components/ui/textarea';
  import { dSettingConfigInfo, cSettingsKeys } from '$lib/constants';
  import { SettingsFieldType } from '$lib/enums/settings.enums';
  import { cSettingsStore } from '$lib/stores/settings.svelte';
  import { cServerStore } from '$lib/stores/server.svelte';
  import { cModelsStore, fSelectedModelName, fPropsCacheVersion } from '$lib/stores/models.svelte';
  import { fNormalizeFloatingPoint } from '$lib/utils/precision';
  import { SettingsChatParameterSourceIndicator } from '$lib/components/app/settings';
  import type { Component } from 'svelte';

  interface Props {
    fields: SettingsFieldConfig[];
    localConfig: SettingsConfigType;
    onConfigChange: (pKey: string, pValue: string | boolean) => void;
    onThemeChange?: (pTheme: string) => void;
  }

  let {
    fields: vFields,
    localConfig: vLocalConfig,
    onConfigChange: vOnConfigChange,
    onThemeChange: vOnThemeChange
  }: Props = $props();

  let vCurrentModelParams = $derived.by(() => {
    fPropsCacheVersion();

    if (cServerStore.mIsRouterMode) {
      const cCurrentModelName = fSelectedModelName();

      if (cCurrentModelName) {
        const cCurrentModelProps = cModelsStore.mGetModelProps(cCurrentModelName);

        return (cCurrentModelProps?.default_generation_settings?.params ?? {}) as Record<
          string,
          unknown
        >;
      }
    }
    return (cServerStore.mDefaultParams ?? {}) as Record<string, unknown>;
  });
</script>

{#each vFields as vField (vField.key)}
  {#if !vField.dependsOn || Boolean(vLocalConfig[vField.dependsOn])}
    <!-- the global search highlights a setting by this id -->
    <div
      id="setting-{vField.key}"
      class={['rounded-md', vField.dependsOn ? 'space-y-2 pl-6' : 'space-y-2']}
    >
      {#if vField.type === SettingsFieldType.INPUT}
        {@const cCurrentValue = String(vLocalConfig[vField.key] ?? '')}
        {@const cServerDefault = vCurrentModelParams[vField.key]}
        {@const cIsCustomRealTime = (() => {
          if (cServerDefault == null) return false;
          if (cCurrentValue === '') return false;

          const cNumericInput = parseFloat(cCurrentValue);
          const cNormalizedInput = !isNaN(cNumericInput)
            ? Math.round(cNumericInput * 1000000) / 1000000
            : cCurrentValue;
          const cNormalizedDefault =
            typeof cServerDefault === 'number'
              ? Math.round(cServerDefault * 1000000) / 1000000
              : cServerDefault;

          return cNormalizedInput !== cNormalizedDefault;
        })()}

        <div class="flex items-center gap-2">
          <Label for={vField.key} class="flex items-center gap-1.5 text-sm font-medium">
            {vField.label}

            {#if vField.isExperimental}
              <FlaskConical class="h-3.5 w-3.5 text-muted-foreground" />
            {/if}
          </Label>
          {#if cIsCustomRealTime}
            <SettingsChatParameterSourceIndicator />
          {/if}
        </div>

        <div class="relative w-full">
          <Input
            id={vField.key}
            type={vField.isPositiveInteger ? 'number' : 'text'}
            {...vField.isPositiveInteger ? { min: '1', step: '1' } : {}}
            value={cCurrentValue}
            oninput={(pE) => vOnConfigChange(vField.key, pE.currentTarget.value)}
            placeholder={vCurrentModelParams[vField.key] != null
              ? fTranslate('text_89f081ec9c11', {
                  p0: fNormalizeFloatingPoint(vCurrentModelParams[vField.key])
                })
              : ''}
            class="w-full {cIsCustomRealTime ? 'pr-8' : ''}"
          />
          {#if cIsCustomRealTime}
            <button
              type="button"
              onclick={() => {
                cSettingsStore.mResetParameterToServerDefault(vField.key);
                vOnConfigChange(vField.key, '');
              }}
              class="absolute top-1/2 right-2 inline-flex h-5 w-5 -translate-y-1/2 items-center justify-center rounded transition-colors hover:bg-muted"
              aria-label={fTranslate('text_bc5b45ae7b60')}
              title={fTranslate('text_bc5b45ae7b60')}
            >
              <RotateCcw class="h-3 w-3" />
            </button>
          {/if}
        </div>
        {#if vField.help || dSettingConfigInfo[vField.key]}
          <p class="mt-1 text-xs text-muted-foreground">
            {@html vField.help || dSettingConfigInfo[vField.key]}
          </p>
        {/if}
      {:else if vField.type === SettingsFieldType.TEXTAREA}
        {#if vField.label}
          <Label for={vField.key} class="block flex items-center gap-1.5 text-sm font-medium">
            {vField.label}

            {#if vField.isExperimental}
              <FlaskConical class="h-3.5 w-3.5 text-muted-foreground" />
            {/if}
          </Label>
        {/if}

        <Textarea
          id={vField.key}
          value={String(vLocalConfig[vField.key] ?? '')}
          onchange={(pE) => vOnConfigChange(vField.key, pE.currentTarget.value)}
          placeholder=""
          class="min-h-[10rem] w-full md:max-w-3xl"
        />

        {#if vField.help || dSettingConfigInfo[vField.key]}
          <p class="mt-1 text-xs text-muted-foreground">
            {vField.help || dSettingConfigInfo[vField.key]}
          </p>
        {/if}

        {#if vField.key === cSettingsKeys.SYSTEM_MESSAGE}
          <div class="mt-3 flex items-center gap-2">
            <Checkbox
              id="showSystemMessage"
              checked={Boolean(vLocalConfig.showSystemMessage ?? true)}
              onCheckedChange={(pChecked) =>
                vOnConfigChange(cSettingsKeys.SHOW_SYSTEM_MESSAGE, Boolean(pChecked))}
            />

            <Label for="showSystemMessage" class="cursor-pointer text-sm font-normal">
              {fTranslate('text_b1e1743d4be2')}
            </Label>
          </div>
        {/if}
      {:else if vField.type === SettingsFieldType.SELECT}
        {@const cSelectedOption = vField.options?.find(
          (pOpt: { value: string; label: string; icon?: Component }) =>
            pOpt.value === vLocalConfig[vField.key]
        )}
        {@const cCurrentValue = vLocalConfig[vField.key]}
        {@const cServerDefault = vCurrentModelParams[vField.key]}
        {@const cIsCustomRealTime = (() => {
          if (cServerDefault == null) return false;
          if (cCurrentValue === '' || cCurrentValue === undefined) return false;
          return cCurrentValue !== cServerDefault;
        })()}

        <div class="flex items-center gap-2">
          <Label for={vField.key} class="flex items-center gap-1.5 text-sm font-medium">
            {vField.label}

            {#if vField.isExperimental}
              <FlaskConical class="h-3.5 w-3.5 text-muted-foreground" />
            {/if}
          </Label>
          {#if cIsCustomRealTime}
            <SettingsChatParameterSourceIndicator />
          {/if}
        </div>

        <Select.Root
          type="single"
          value={cCurrentValue}
          onValueChange={(pValue) => {
            if (vField.key === cSettingsKeys.THEME && pValue && vOnThemeChange) {
              vOnThemeChange(pValue);
            } else {
              vOnConfigChange(vField.key, pValue);
            }
          }}
        >
          <div class="relative w-full md:w-auto">
            <Select.Trigger class="w-full">
              <div class="flex items-center gap-2">
                {#if cSelectedOption?.icon}
                  {@const IconComponent = cSelectedOption.icon}
                  <IconComponent class={cIconClassDefault} />
                {/if}

                {cSelectedOption?.label ||
                  fTranslate('text_8d29790dbe82', { p0: vField.label.toLowerCase() })}
              </div>
            </Select.Trigger>
            {#if cIsCustomRealTime}
              <button
                type="button"
                onclick={() => {
                  cSettingsStore.mResetParameterToServerDefault(vField.key);
                  vOnConfigChange(vField.key, '');
                }}
                class="absolute top-1/2 right-8 inline-flex h-5 w-5 -translate-y-1/2 items-center justify-center rounded transition-colors hover:bg-muted"
                aria-label={fTranslate('text_bc5b45ae7b60')}
                title={fTranslate('text_bc5b45ae7b60')}
              >
                <RotateCcw class="h-3 w-3" />
              </button>
            {/if}
          </div>
          <Select.Content>
            {#if vField.options}
              {#each vField.options as vOption (vOption.value)}
                <Select.Item value={vOption.value} label={vOption.label}>
                  <div class="flex items-center gap-2">
                    {#if vOption.icon}
                      {@const IconComponent = vOption.icon}
                      <IconComponent class={cIconClassDefault} />
                    {/if}
                    {vOption.label}
                  </div>
                </Select.Item>
              {/each}
            {/if}
          </Select.Content>
        </Select.Root>
        {#if vField.help || dSettingConfigInfo[vField.key]}
          <p class="mt-1 text-xs text-muted-foreground">
            {vField.help || dSettingConfigInfo[vField.key]}
          </p>
        {/if}
      {:else if vField.type === SettingsFieldType.RADIO && vField.radioOptions}
        {@const aRadioOptions = vField.radioOptions}
        {@const cCurrentMode =
          aRadioOptions.find((pO: { key: string }) => Boolean(vLocalConfig[pO.key]))?.value ??
          aRadioOptions[0].value}

        <Label class="flex items-center gap-1.5 text-sm font-medium mb-4">
          {vField.label}

          {#if vField.isExperimental}
            <FlaskConical class="h-3.5 w-3.5 text-muted-foreground" />
          {/if}
        </Label>

        <RadioGroup.Root
          class="gap-4"
          value={cCurrentMode}
          onValueChange={(pValue) => {
            for (const cOpt of aRadioOptions) {
              vOnConfigChange(cOpt.key, cOpt.value === pValue);
            }
          }}
        >
          {#each aRadioOptions as vOpt (vOpt.value)}
            {@const cItemId = `${vField.key}-${vOpt.value}`}
            <div class="flex items-center gap-2">
              <RadioGroup.Item value={vOpt.value} id={cItemId} />
              <Label
                for={cItemId}
                class="flex cursor-pointer items-center gap-1.5 text-sm font-normal"
              >
                {vOpt.label}

                {#if vOpt.isExperimental}
                  <FlaskConical class="h-3.5 w-3.5 text-muted-foreground" />
                {/if}
              </Label>
            </div>
          {/each}
        </RadioGroup.Root>

        {#if vField.help || dSettingConfigInfo[vField.key]}
          <p class="text-xs text-muted-foreground">
            {vField.help || dSettingConfigInfo[vField.key]}
          </p>
        {/if}
      {:else if vField.type === SettingsFieldType.CHECKBOX}
        <div class="flex items-start space-x-3">
          <Checkbox
            id={vField.key}
            checked={Boolean(vLocalConfig[vField.key])}
            onCheckedChange={(pChecked) => vOnConfigChange(vField.key, pChecked)}
            class="mt-1"
          />

          <div class="space-y-1">
            <label
              for={vField.key}
              class="flex cursor-pointer items-center gap-1.5 pt-1 pb-0.5 text-sm leading-none font-medium"
            >
              {vField.label}

              {#if vField.isExperimental}
                <FlaskConical class="h-3.5 w-3.5 text-muted-foreground" />
              {/if}
            </label>

            {#if vField.help || dSettingConfigInfo[vField.key]}
              <p class="text-xs text-muted-foreground">
                {vField.help || dSettingConfigInfo[vField.key]}
              </p>
            {/if}
          </div>
        </div>
      {/if}
    </div>
  {/if}
{/each}
