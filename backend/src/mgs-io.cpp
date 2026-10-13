#include "mgs-io.h"

#include <vector>

void mgs_io_write_i::mWriteString(const std::string &pStr) {
  uint32_t vStrSize = pStr.size();

  mWrite(&vStrSize, sizeof(vStrSize));
  mWrite(pStr.data(), vStrSize);
}

void mgs_io_read_i::mReadString(std::string &pStr) {
  uint32_t vStrSize;
  mRead(&vStrSize, sizeof(vStrSize));

  std::vector<char> lBuf(vStrSize);
  mRead(lBuf.data(), vStrSize);

  pStr.assign(lBuf.data(), vStrSize);
}
