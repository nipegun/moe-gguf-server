import type { PageLoad } from './$types';
import { fValidateApiKey } from '$lib/utils/api-key-validation';

export const load: PageLoad = async ({ fetch: pFetch }) => {
  await fValidateApiKey(pFetch);
};
