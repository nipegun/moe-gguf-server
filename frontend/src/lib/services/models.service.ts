import { ServerModelStatus } from '$lib/enums';
import { apiFetch, apiPost, normalizeModelName } from '$lib/utils';
import type { ParsedModelId } from '$lib/types/models';
import {
  cModelQuantizationSegmentRe,
  cModelCustomQuantizationPrefixRe,
  cModelParamsRe,
  cModelActivatedParamsRe,
  sModelIgnoredSegments,
  cModelWeightExtensionRe,
  cModelIdNotFound,
  cModelIdOrgSeparator,
  cModelIdSegmentSeparator,
  cModelIdQuantizationSeparator,
  dApiModels
} from '$lib/constants';

export class ModelsService {
  /**
   *
   *
   * Listing
   *
   *
   */

  /**
   * Fetch list of models from OpenAI-compatible endpoint.
   * Works in both MODEL and ROUTER modes.
   *
   * @returns List of available models with basic metadata
   */
  static async mList(): Promise<ApiModelListResponse> {
    return apiFetch<ApiModelListResponse>(dApiModels.LIST);
  }

  /**
   * Fetch list of all models with detailed metadata (ROUTER mode).
   * Returns models with load status, paths, and other metadata
   * beyond what the OpenAI-compatible endpoint provides.
   *
   * @returns List of models with detailed status and configuration info
   */
  static async mListRouter(): Promise<ApiRouterModelsListResponse> {
    return apiFetch<ApiRouterModelsListResponse>(dApiModels.LIST);
  }

  /**
   *
   *
   * Load/Unload
   *
   *
   */

  /**
   * Load a model (ROUTER mode only).
   * Sends POST request to `/models/load`. Note: the endpoint returns success
   * before loading completes — use polling to await actual load status.
   *
   * @param pModelId - Model identifier to load
   * @param pExtraArgs - Optional additional arguments to pass to the model instance
   * @returns Load response from the server
   */
  static async mLoad(
    pModelId: string,
    pExtraArgs?: string[]
  ): Promise<ApiRouterModelsLoadResponse> {
    const aPayload: { model: string; extra_args?: string[] } = { model: pModelId };
    if (pExtraArgs && pExtraArgs.length > 0) {
      aPayload.extra_args = pExtraArgs;
    }

    return apiPost<ApiRouterModelsLoadResponse>(dApiModels.LOAD, aPayload);
  }

  /**
   * Unload a model (ROUTER mode only).
   * Sends POST request to `/models/unload`. Note: the endpoint returns success
   * before unloading completes — use polling to await actual unload status.
   *
   * @param pModelId - Model identifier to unload
   * @returns Unload response from the server
   */
  static async mUnload(pModelId: string): Promise<ApiRouterModelsUnloadResponse> {
    return apiPost<ApiRouterModelsUnloadResponse>(dApiModels.UNLOAD, { model: pModelId });
  }

  /**
   *
   *
   * Status
   *
   *
   */

  /**
   * Check if a model is loaded based on its metadata.
   *
   * @param pModel - Model data entry from the API response
   * @returns True if the model status is LOADED
   */
  static mIsModelLoaded(pModel: ApiModelDataEntry): boolean {
    return pModel.status.value === ServerModelStatus.LOADED;
  }

  /**
   * Check if a model is currently loading.
   *
   * @param pModel - Model data entry from the API response
   * @returns True if the model status is LOADING
   */
  static mIsModelLoading(pModel: ApiModelDataEntry): boolean {
    return pModel.status.value === ServerModelStatus.LOADING;
  }

  /**
   *
   *
   * Parsing
   *
   *
   */

  /**
   * Parse a model ID string into its structured components.
   *
   * Handles conventions like:
   *   `<org>/<ModelName>-<Parameters>(-<ActivatedParameters>)(-<Tags>)(-<Quantization>):<Quantization>`
   *   `<ModelName>.<Quantization>` (dot-separated quantization, e.g. `model.Q4_K_M`)
   *
   * @param pModelId - Raw model identifier string
   * @returns Structured {@link ParsedModelId} with all detected fields
   */
  static mParseModelId(pModelId: string): ParsedModelId {
    const dResult: ParsedModelId = {
      raw: pModelId,
      orgName: null,
      modelName: null,
      params: null,
      activatedParams: null,
      quantization: null,
      tags: []
    };

    // strip directory path and weight extension so a bare `-m /path/file.gguf`
    // parses like a clean repo id; the HF `org/model` form is preserved
    const cSource = normalizeModelName(pModelId).replace(cModelWeightExtensionRe, '');

    // 1. Extract colon-separated quantization (e.g. `model:Q4_K_M`)
    const cColonIdx = cSource.indexOf(cModelIdQuantizationSeparator);
    let vModelPath: string;

    if (cColonIdx !== cModelIdNotFound) {
      dResult.quantization = cSource.slice(cColonIdx + 1) || null;
      vModelPath = cSource.slice(0, cColonIdx);
    } else {
      vModelPath = cSource;
    }

    // 2. Extract org name (e.g. `org/model` -> org = "org")
    const cSlashIdx = vModelPath.indexOf(cModelIdOrgSeparator);
    let vModelStr: string;

    if (cSlashIdx !== cModelIdNotFound) {
      dResult.orgName = vModelPath.slice(0, cSlashIdx);
      vModelStr = vModelPath.slice(cSlashIdx + 1);
    } else {
      vModelStr = vModelPath;
    }

    // 3. Handle dot-separated quantization (e.g. `model-name.Q4_K_M`)
    const cDotIdx = vModelStr.lastIndexOf('.');

    if (cDotIdx !== cModelIdNotFound && !dResult.quantization) {
      const cAfterDot = vModelStr.slice(cDotIdx + 1);

      if (cModelQuantizationSegmentRe.test(cAfterDot)) {
        dResult.quantization = cAfterDot;
        vModelStr = vModelStr.slice(0, cDotIdx);
      }
    }

    const cSegments = vModelStr.split(cModelIdSegmentSeparator);

    // 4. Detect trailing quantization from dash-separated segments
    //    Handle UD-prefixed quantization (e.g. `UD-Q8_K_XL`) and
    //    standalone quantization (e.g. `Q4_K_M`, `BF16`, `F16`, `MXFP4`)
    if (!dResult.quantization && cSegments.length > 1) {
      const cLast = cSegments[cSegments.length - 1];
      const cSecondLast = cSegments.length > 2 ? cSegments[cSegments.length - 2] : null;

      if (cModelQuantizationSegmentRe.test(cLast)) {
        if (cSecondLast && cModelCustomQuantizationPrefixRe.test(cSecondLast)) {
          dResult.quantization = `${cSecondLast}-${cLast}`;
          cSegments.splice(cSegments.length - 2, 2);
        } else {
          dResult.quantization = cLast;
          cSegments.pop();
        }
      }
    }

    // 5. Find params and activated params
    let vParamsIdx = cModelIdNotFound;
    let vActivatedParamsIdx = cModelIdNotFound;

    for (let vI = 0; vI < cSegments.length; vI++) {
      const cSeg = cSegments[vI];

      if (vParamsIdx === cModelIdNotFound && cModelParamsRe.test(cSeg)) {
        vParamsIdx = vI;
        dResult.params = cSeg.toUpperCase();
      } else if (vParamsIdx !== cModelIdNotFound && cModelActivatedParamsRe.test(cSeg)) {
        vActivatedParamsIdx = vI;
        dResult.activatedParams = cSeg.toUpperCase();
      }
    }

    // 6. Model name = segments before params; tags = remaining segments after params
    const cPivotIdx = vParamsIdx !== cModelIdNotFound ? vParamsIdx : cSegments.length;

    dResult.modelName = cSegments.slice(0, cPivotIdx).join(cModelIdSegmentSeparator) || null;

    if (vParamsIdx !== cModelIdNotFound) {
      dResult.tags = cSegments.slice(vParamsIdx + 1).filter((pIgnored, pRelIdx) => {
        const cAbsIdx = vParamsIdx + 1 + pRelIdx;
        if (cAbsIdx === vActivatedParamsIdx) return false;

        return !sModelIgnoredSegments.has(cSegments[cAbsIdx].toUpperCase());
      });
    }

    return dResult;
  }
}
