<script lang="ts">
  import { fTranslate } from '../../../i18n';

  // Centered dialog for the sudo password prompts of the commands run by the local shell MCP
  // server (--local-shell-mcp-server). It is mounted once in the root layout, so it opens on any
  // page: while a shell call runs it asks the server every SHELL_MCP_ASKPASS_POLL_MS whether sudo
  // is waiting for a password, and shows the first prompt with the tool, the command, a password
  // field and the Accept and Cancel buttons. The password only goes to the server, which hands it
  // to sudo; it is never kept here once sent. The password field takes the focus as soon as a
  // prompt shows, Enter accepts like the Accept button, Escape cancels like the Cancel button, and
  // a click outside does nothing, so sudo always gets an answer.

  import { untrack } from 'svelte';
  import { KeyRound } from '@lucide/svelte';
  import SyntaxHighlightedCode from '$lib/components/app/content/SyntaxHighlightedCode.svelte';
  import { Button } from '$lib/components/ui/button';
  import * as Dialog from '$lib/components/ui/dialog';
  import { Input } from '$lib/components/ui/input';
  import { cMaxHeightCodeBlock, cShellMcpAskpassPollMs } from '$lib/constants';
  import { FileTypeText, KeyboardKey } from '$lib/enums';
  import { LocalShellAskpassService, type LocalShellAskpassPrompt } from '$lib/services';
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import type { LocalShellActiveCall } from '$lib/types';
  import { isIMEComposing } from '$lib/utils';

  interface LocalShellPendingPrompt extends LocalShellAskpassPrompt {
    call: LocalShellActiveCall;
  }

  const cLCalls = $derived(cMcpStore.mActiveShellCalls());

  let vLPrompts = $state<LocalShellPendingPrompt[]>([]);
  // prompts already answered, hidden before the next check confirms it
  let vLAnsweredIds = $state<string[]>([]);
  // calls that already received a password: a new prompt means sudo is asking again
  let vLPasswordCallIds = $state<string[]>([]);
  let vPassword = $state('');
  let vSending = $state(false);
  let vError = $state('');
  let vShownId = '';

  const dVisible = $derived(
    vLPrompts.find(
      (pPrompt) =>
        !vLAnsweredIds.includes(pPrompt.id) && cLCalls.some((pCall) => pCall.id === pPrompt.call.id)
    ) ?? null
  );
  const vAskingAgain = $derived(dVisible !== null && vLPasswordCallIds.includes(dVisible.call.id));
  // only the id: the poll replaces the prompt objects on every tick
  const vVisibleId = $derived(dVisible?.id ?? '');

  // While shell calls run, ask whether sudo is waiting for a password in any of them.
  $effect(() => {
    const cLRunning = cLCalls;
    if (cLRunning.length === 0) {
      vLPrompts = [];
      vLAnsweredIds = [];
      vLPasswordCallIds = [];
      return;
    }
    let vStopped = false;
    let vTimer: ReturnType<typeof setTimeout> | undefined;
    const fPoll = async () => {
      const cLChecked = await Promise.all(
        cLRunning.map(async (pCall): Promise<LocalShellPendingPrompt | null> => {
          try {
            const dPending = await LocalShellAskpassService.mPending(pCall.id);
            return dPending ? { ...dPending, call: pCall } : null;
          } catch {
            // a failed check keeps what the last one found and is repeated on the next tick
            return vLPrompts.find((pPrompt) => pPrompt.call.id === pCall.id) ?? null;
          }
        })
      );
      if (vStopped) return;
      vLPrompts = cLChecked.filter(
        (pPrompt): pPrompt is LocalShellPendingPrompt => pPrompt !== null
      );
      vTimer = setTimeout(fPoll, cShellMcpAskpassPollMs);
    };
    untrack(() => fPoll());
    return () => {
      vStopped = true;
      clearTimeout(vTimer);
    };
  });

  // Every new prompt starts with an empty field and no error.
  $effect(() => {
    const vId = dVisible?.id ?? '';
    if (vId !== vShownId) {
      vShownId = vId;
      vPassword = '';
      vError = '';
    }
  });

  async function fAnswer(pCancel: boolean) {
    const dPrompt = dVisible;
    if (!dPrompt || vSending || (!pCancel && vPassword.length === 0)) return;
    vSending = true;
    vError = '';
    try {
      if (pCancel) {
        await LocalShellAskpassService.mCancel(dPrompt.call.id, dPrompt.id);
      } else {
        await LocalShellAskpassService.mAnswer(dPrompt.call.id, dPrompt.id, vPassword);
        if (!vLPasswordCallIds.includes(dPrompt.call.id)) {
          vLPasswordCallIds = [...vLPasswordCallIds, dPrompt.call.id];
        }
      }
      vLAnsweredIds = [...vLAnsweredIds, dPrompt.id];
    } catch (pError) {
      vError = fTranslate('text_962d26ea3965', {
        p0: pError instanceof Error ? pError.message : String(pError)
      });
    } finally {
      vPassword = '';
      vSending = false;
    }
  }

  function fHandleSubmit(pEvent: Event) {
    pEvent.preventDefault();
    fAnswer(false);
  }

  function fHandleEscape(pEvent: KeyboardEvent) {
    pEvent.preventDefault();
    fAnswer(true);
  }

  // Enter accepts like the Accept button, except while an IME is still composing a word. It does
  // not rely on the implicit submission of the form, which depends on the browser.
  function fHandlePasswordKeydown(pEvent: KeyboardEvent) {
    if (pEvent.key !== KeyboardKey.ENTER || isIMEComposing(pEvent)) return;
    pEvent.preventDefault();
    fAnswer(false);
  }

  // The dialog would focus its first focusable element; the password field takes the focus
  // instead (fFocusPassword), because it is mounted with the dialog and its ref is not bound yet
  // when the dialog opens.
  function fHandleOpenAutoFocus(pEvent: Event) {
    pEvent.preventDefault();
  }

  // Attachment of the password field: puts the cursor in it whenever a new prompt shows, and again
  // when a failed answer enables it back, so the user can type right away. It waits a frame so the
  // focus trap of the dialog is already in place.
  function fFocusPassword(pInput: HTMLInputElement) {
    if (vVisibleId === '' || vSending) return;
    const vFrame = requestAnimationFrame(() => pInput.focus());
    return () => cancelAnimationFrame(vFrame);
  }
</script>

<Dialog.Root open={dVisible !== null}>
  <Dialog.Content
    showCloseButton={false}
    interactOutsideBehavior="ignore"
    onEscapeKeydown={fHandleEscape}
    onOpenAutoFocus={fHandleOpenAutoFocus}
  >
    {#if dVisible}
      <Dialog.Header class="min-w-0">
        <Dialog.Title class="flex min-w-0 items-center gap-2">
          <KeyRound class="h-5 w-5 shrink-0" />
          <span class="min-w-0">{fTranslate('text_7f84c251ca9a')}</span>
        </Dialog.Title>

        <Dialog.Description class="font-mono [overflow-wrap:anywhere]">
          {dVisible.prompt}
        </Dialog.Description>
      </Dialog.Header>

      {#if dVisible.call.command}
        <div class="min-w-0">
          <div class="mb-1.5 text-xs text-muted-foreground/70">{dVisible.call.toolName}:</div>

          <SyntaxHighlightedCode
            code={dVisible.call.command}
            language={FileTypeText.PLAIN_TEXT}
            maxHeight={cMaxHeightCodeBlock}
          />
        </div>
      {/if}

      <form class="flex min-w-0 flex-col gap-2" onsubmit={fHandleSubmit}>
        <label for="local-shell-askpass-password" class="text-sm font-medium text-muted-foreground">
          {fTranslate('text_e7cf3ef4f17c')}
        </label>

        <Input
          id="local-shell-askpass-password"
          type="password"
          autocomplete="off"
          spellcheck={false}
          disabled={vSending}
          onkeydown={fHandlePasswordKeydown}
          bind:value={vPassword}
          {@attach fFocusPassword}
        />

        {#if vAskingAgain}
          <p class="text-xs text-muted-foreground [overflow-wrap:anywhere]">
            {fTranslate('text_0bcf05a1fdd7')}
          </p>
        {/if}

        {#if vError}
          <p class="text-xs text-destructive [overflow-wrap:anywhere]" role="alert">{vError}</p>
        {/if}

        <p class="text-xs text-muted-foreground/70 [overflow-wrap:anywhere]">
          {fTranslate('text_a690036b9c51')}
        </p>
      </form>

      <Dialog.Footer>
        <Button variant="outline" disabled={vSending} onclick={() => fAnswer(true)}>
          {fTranslate('text_19766ed6ccb2')}
        </Button>

        <Button disabled={vSending || vPassword.length === 0} onclick={() => fAnswer(false)}>
          {fTranslate('text_89713b9c9c1b')}
        </Button>
      </Dialog.Footer>
    {/if}
  </Dialog.Content>
</Dialog.Root>
