// The runtime: the plugin, one client and its one device.
#pragma once

#include <memory>

namespace fsim {

class Pipeline;

class Runtime {
public:
  // Loads the build-time plugin and creates the client; throws on failure.
  Runtime();
  ~Runtime();
  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;

  struct Impl;

private:
  friend class Pipeline;
  std::unique_ptr<Impl> impl_;
};

} // namespace fsim
