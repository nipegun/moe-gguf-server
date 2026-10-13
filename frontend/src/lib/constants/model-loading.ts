import { fTranslate } from '../i18n';
/**
 * Labels shown while a model loads, keyed by the stage reported on /models/sse.
 */
export const dModelLoadStageLabels: Record<ApiModelLoadStage, string> = {
  text_model: fTranslate('text_6996f21952a8'),
  spec_model: fTranslate('text_5af36567f9a9'),
  mmproj_model: fTranslate('text_d5b1c506a74f')
};

/**
 * Share of the bar reserved for each load phase after text_model.
 * text_model fills the rest, so a plain model reaches 100% on its own.
 */
export const cModelLoadTailShare = 0.1;
