<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { Wrench } from '@lucide/svelte';
  import { Badge } from '$lib/components/ui/badge';

  interface Props {
    class?: string;
  }

  let { class: vClassName = '' }: Props = $props();
</script>

<Badge
  variant="secondary"
  class="h-5 bg-orange-100 px-1.5 py-0.5 text-xs text-orange-800 dark:bg-orange-900 dark:text-orange-200 {vClassName}"
>
  <Wrench class="mr-1 h-3 w-3" />
  {fTranslate('text_494ca78f7374')}
</Badge>
