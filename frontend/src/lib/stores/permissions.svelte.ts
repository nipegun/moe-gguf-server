import { cUiSettingsStore } from '$lib/stores/ui-settings.svelte';

import { SvelteSet } from 'svelte/reactivity';

class PermissionsStore {
  private vTools = $state(new SvelteSet<string>());

  constructor() {
    // always-allowed tools are kept in the server's settings document (section "alwaysAllowedTools")
    cUiSettingsStore.mRegister('alwaysAllowedTools', (pSaved) => {
      this.vTools.clear();
      if (Array.isArray(pSaved)) {
        for (const cName of pSaved) {
          if (typeof cName === 'string') this.vTools.add(cName);
        }
      }
    });
  }

  get mTools(): ReadonlySet<string> {
    return this.vTools;
  }

  mHasTool(pKey: string): boolean {
    return this.vTools.has(pKey);
  }

  mAllowTool(pKey: string): void {
    this.vTools.add(pKey);
    this.mPersist();
  }

  mAllowTools(pKeys: string[]): void {
    for (const cKey of pKeys) this.vTools.add(cKey);
    this.mPersist();
  }

  mRevokeTool(pKey: string): void {
    this.vTools.delete(pKey);
    this.mPersist();
  }

  private mPersist(): void {
    cUiSettingsStore.mSet('alwaysAllowedTools', [...this.vTools]);
  }
}

export const cPermissionsStore = new PermissionsStore();
