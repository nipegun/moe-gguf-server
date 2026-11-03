import { browser } from '$app/environment';
import { cDefaultMobileBreakpoint } from '$lib/constants/viewport';
import { MediaQuery } from 'svelte/reactivity';

export const cViewport = $state({
  width: browser ? window.innerWidth : 0
});

export const cIsMobile = new MediaQuery(`max-width: ${cDefaultMobileBreakpoint - 1}px`);
