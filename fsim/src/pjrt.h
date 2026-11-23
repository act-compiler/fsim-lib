// The library's PJRT side: the runtime's handles and the error checks.
#pragma once

#include <cstdint>

#include "fsim/runtime.h"
#include "fsim/tensor.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace fsim {

// The api, client and device one runtime owns.
struct Runtime::Impl {
  const PJRT_Api *api = nullptr;
  PJRT_Client *client = nullptr;
  PJRT_Device *device = nullptr;

  ~Impl();

  // Throws PJRT's message on an error, naming the call.
  void check(PJRT_Error *error, const char *call) const;
  // Awaits an event, destroys it, then checks its error.
  void await(PJRT_Event *event) const;
  // The host's available memory, read from /proc/meminfo.
  static int64_t host_limit();
  // The device's byte limit: available host memory on the CPU; throws if the device reports none.
  int64_t bytes_limit() const;
  void destroy(PJRT_Buffer *buffer) const;
  void destroy(PJRT_Executable *executable) const;
  void destroy(PJRT_LoadedExecutable *executable) const;
};

// The PJRT buffer type of an element type.
PJRT_Buffer_Type buffer_type(ElementType type);

} // namespace fsim
