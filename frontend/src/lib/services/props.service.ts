import { apiFetchWithParams } from '$lib/utils';

export class PropsService {
  /**
   *
   *
   * Fetching
   *
   *
   */

  /**
   * Fetches global server properties from the `/props` endpoint.
   * In MODEL mode, returns modalities for the single loaded model.
   * In ROUTER mode, returns server-wide settings without model-specific modalities.
   *
   * @param pAutoload - If false, prevents automatic model loading (default: false)
   * @returns Server properties including default generation settings and capabilities
   * @throws {Error} If the request fails or returns invalid data
   */
  static async mFetch(pAutoload = false): Promise<ApiMgsServerProps> {
    const dParams: Record<string, string> = {};
    if (!pAutoload) {
      dParams.autoload = 'false';
    }

    return apiFetchWithParams<ApiMgsServerProps>('./props', dParams, { authOnly: true });
  }

  /**
   * Fetches server properties for a specific model (ROUTER mode only).
   * Required in ROUTER mode because global `/props` does not include per-model modalities.
   *
   * @param pModelId - The model ID to fetch properties for
   * @param pAutoload - If false, prevents automatic model loading (default: false)
   * @returns Server properties specific to the requested model
   * @throws {Error} If the request fails, model not found, or model not loaded
   */
  static async mFetchForModel(pModelId: string, pAutoload = false): Promise<ApiMgsServerProps> {
    const dParams: Record<string, string> = { model: pModelId };
    if (!pAutoload) {
      dParams.autoload = 'false';
    }

    return apiFetchWithParams<ApiMgsServerProps>('./props', dParams, { authOnly: true });
  }
}
