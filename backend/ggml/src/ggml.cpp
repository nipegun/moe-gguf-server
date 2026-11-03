#include "ggml-impl.h"

#include <cstdlib>
#include <exception>

static std::terminate_handler previous_terminate_handler;

GGML_NORETURN static void fGgmlUncaughtException() {
  ggml_print_backtrace();
  if (previous_terminate_handler) {
    previous_terminate_handler();
  }
  abort(); // unreachable unless previous_terminate_handler was nullptr
}

static bool ggml_uncaught_exception_init = [] {
  const char *vGgmlNoBacktrace = getenv("GGML_NO_BACKTRACE");
  if (vGgmlNoBacktrace) {
    return false;
  }
  const auto cPrev{std::get_terminate()};
  GGML_ASSERT(cPrev != fGgmlUncaughtException);
  previous_terminate_handler = cPrev;
  std::set_terminate(fGgmlUncaughtException);
  return true;
}();
