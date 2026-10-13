#include "mgs-memory.h"

mgs_memory_status mgs_memory_status_combine(mgs_memory_status pS0, mgs_memory_status pS1) {
  bool vHasUpdate = false;

  switch (pS0) {
  case cMgsMemoryStatusSuccess: {
    vHasUpdate = true;
    break;
  }
  case cMgsMemoryStatusNoUpdate: {
    break;
  }
  case cMgsMemoryStatusFailedPrepare:
  case cMgsMemoryStatusFailedCompute: {
    return pS0;
  }
  }

  switch (pS1) {
  case cMgsMemoryStatusSuccess: {
    vHasUpdate = true;
    break;
  }
  case cMgsMemoryStatusNoUpdate: {
    break;
  }
  case cMgsMemoryStatusFailedPrepare:
  case cMgsMemoryStatusFailedCompute: {
    return pS1;
  }
  }

  // if either status has an update, then the combined status has an update
  return vHasUpdate ? cMgsMemoryStatusSuccess : cMgsMemoryStatusNoUpdate;
}

bool mgs_memory_status_is_fail(mgs_memory_status pStatus) {
  switch (pStatus) {
  case cMgsMemoryStatusSuccess:
  case cMgsMemoryStatusNoUpdate: {
    return false;
  }
  case cMgsMemoryStatusFailedPrepare:
  case cMgsMemoryStatusFailedCompute: {
    return true;
  }
  }

  return false;
}
