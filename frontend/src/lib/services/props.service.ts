import { fApiFetchWithParams, fApiPost } from '$lib/utils';
import { cShellMcpEnabledEndpoint } from '$lib/constants/api-endpoints';

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
  /**
   * Turns the local shell MCP server on or off (Settings > MCP servers). The server saves the
   * choice in ~/.mgs/config/ui-settings.json and answers with the shell_mcp_server fields of /props.
   */
  static async mSetShellMcpServer(
    pEnabled: boolean
  ): Promise<
    Pick<
      ApiMgsServerProps,
      'shell_mcp_server' | 'shell_mcp_server_forced' | 'shell_mcp_server_toggle'
    >
  > {
    return fApiPost(cShellMcpEnabledEndpoint, { enabled: pEnabled });
  }

  static async mFetch(pAutoload = false): Promise<ApiMgsServerProps> {
    const dParams: Record<string, string> = {};
    if (!pAutoload) {
      dParams.autoload = 'false';
    }

    return fApiFetchWithParams<ApiMgsServerProps>('/props', dParams, { authOnly: true });
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

    return fApiFetchWithParams<ApiMgsServerProps>('/props', dParams, { authOnly: true });
  }
}
