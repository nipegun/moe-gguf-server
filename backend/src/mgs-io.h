#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct ggml_tensor;

class mgs_io_write_i {
public:
  mgs_io_write_i() = default;
  virtual ~mgs_io_write_i() = default;

  virtual void mWrite(const void *pSrc, size_t pSize) = 0;
  virtual void mWriteTensor(ggml_tensor *pTensor, size_t pOffset, size_t pSize) = 0;

  // bytes written so far
  virtual size_t mNBytes() = 0;

  void mWriteString(const std::string &pStr);
};

class mgs_io_read_i {
public:
  mgs_io_read_i() = default;
  virtual ~mgs_io_read_i() = default;

  virtual void mRead(void *pDst, size_t pSize) = 0;
  virtual void mReadTensor(ggml_tensor *pTensor, size_t pOffset, size_t pSize) = 0;

  // bytes read so far
  virtual size_t mNBytes() = 0;

  void mReadString(std::string &pStr);
};
