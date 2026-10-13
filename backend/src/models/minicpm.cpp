#include "models.h"

void mgs_model_minicpm::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);
  pMl.mGetKey(cLlmKvEmbeddingScale, vHparams.vFEmbeddingScale);
  pMl.mGetKey(cLlmKvResidualScale, vHparams.vFResidualScale);
  pMl.mGetKey(cLlmKvLogitScale, vHparams.vFLogitScale);

  // MiniCPM uses rope by default, unlike Granite which uses it as a switch
  vHparams.vRopeFinetuned = true;

  switch (vHparams.mNLayer()) {
  case 52:
    vType = cLlmType1b;
    break;
  case 40:
    vType = cLlmType2b;
    break;
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_minicpm::mLoadArchTensors(mgs_model_loader &) {
  MGS_LOAD_LOCALS;

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  // output
  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, cTensorNotRequired);

  // if output is NULL, init from the input tok embed
  if (vOutput == NULL) {
    vOutput = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, cTensorDuplicated);
  }

  for (int vI = 0; vI < cNLayer; ++vI) {
    auto &vLayer = lLayers[vI];

    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", vI), {cNEmbd}, 0);

    mCreateTensorQkv(vLayer, vI, cNEmbd, cNEmbdHeadK * cNHead, cNEmbdKGqa, cNEmbdVGqa, 0);
    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNEmbdHeadK * cNHead, cNEmbd}, 0);

    // optional bias tensors
    vLayer.vWoB = mCreateTensor(cTn(cLlmTensorAttnOut, "bias", vI), {cNEmbd}, cTensorNotRequired);

    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, 0);

    if (vHparams.vRopeScalingTypeTrain == MGS_ROPE_SCALING_TYPE_LONGROPE) {
      vLayer.vRopeLong = mCreateTensor(cTn(cLlmTensorRopeFactorsLong, "weight", vI), {cNRot / 2},
                                       cTensorNotRequired | (vI != 0 ? cTensorDuplicated : 0));
      vLayer.vRopeShort = mCreateTensor(cTn(cLlmTensorRopeFactorsShort, "weight", vI), {cNRot / 2},
                                        cTensorNotRequired | (vI != 0 ? cTensorDuplicated : 0));
    } else {
      vLayer.vRopeFreqs = mCreateTensor(cTn(cLlmTensorRopeFreqs, "weight", vI), {cNRot / 2},
                                        cTensorNotRequired | (vI != 0 ? cTensorDuplicated : 0));
    }

    if (cNExpert == 0) {
      vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", vI), {cNEmbd, cNFf}, 0);
      vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", vI), {cNFf, cNEmbd}, 0);
      vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", vI), {cNEmbd, cNFf}, 0);

      // optional MLP bias
      vLayer.vFfnGateB = mCreateTensor(cTn(cLlmTensorFfnGate, "bias", vI), {cNFf}, cTensorNotRequired);
      vLayer.vFfnDownB = mCreateTensor(cTn(cLlmTensorFfnDown, "bias", vI), {cNEmbd}, cTensorNotRequired);
      vLayer.vFfnUpB = mCreateTensor(cTn(cLlmTensorFfnUp, "bias", vI), {cNFf}, cTensorNotRequired);
    } else {
      vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, 0);
      vLayer.vFfnGateExps =
          mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbd, cNFf, cNExpert}, cTensorNotRequired);
      vLayer.vFfnDownExps = mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFf, cNEmbd, cNExpert}, 0);
      vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, cNFf, cNExpert}, 0);

      // For Granite MoE Shared
      if (vHparams.vNFfShexp > 0) {
        vLayer.vFfnGateShexp =
            mCreateTensor(cTn(cLlmTensorFfnGateShexp, "weight", vI), {cNEmbd, vHparams.vNFfShexp}, 0);
        vLayer.vFfnUpShexp = mCreateTensor(cTn(cLlmTensorFfnUpShexp, "weight", vI), {cNEmbd, vHparams.vNFfShexp}, 0);
        vLayer.vFfnDownShexp =
            mCreateTensor(cTn(cLlmTensorFfnDownShexp, "weight", vI), {vHparams.vNFfShexp, cNEmbd}, 0);
      }
    }
  }
}

std::unique_ptr<llm_graph_context> mgs_model_minicpm::mBuildArchGraph(const llm_graph_params &pParams) const {
  return std::make_unique<graph>(*this, pParams);
}
