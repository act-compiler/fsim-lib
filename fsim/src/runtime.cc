#include <dlfcn.h>

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

#include "pjrt.h"

namespace fsim {

Runtime::Runtime() : impl_(std::make_unique<Impl>()) {
  // The plugin stays loaded until the process exits.
  void *plugin = dlopen(FSIM_PJRT_PLUGIN, RTLD_NOW | RTLD_LOCAL);
  if (!plugin)
    throw std::runtime_error(std::string("fsim: dlopen: ") + dlerror());
  auto get_api = reinterpret_cast<const PJRT_Api *(*)()>(dlsym(plugin, "GetPjrtApi"));
  if (!get_api)
    throw std::runtime_error(std::string("fsim: dlsym GetPjrtApi: ") + dlerror());
  const PJRT_Api *api = get_api();

  // Header and plugin must agree on the major version.
  int major = api->pjrt_api_version.major_version;
  int minor = api->pjrt_api_version.minor_version;
  std::string versions = "fsim: plugin api " + std::to_string(major) + "." + std::to_string(minor) +
                         ", header " + std::to_string(PJRT_API_MAJOR) + "." +
                         std::to_string(PJRT_API_MINOR);
  if (major != PJRT_API_MAJOR)
    throw std::runtime_error(versions);
  // An older plugin must reach the last PJRT_Api field called.
  if (api->struct_size < offsetof(PJRT_Api, PJRT_Executable_OutputDimensions) +
                             sizeof(api->PJRT_Executable_OutputDimensions))
    throw std::runtime_error(versions + ": plugin lacks PJRT_Executable_OutputDimensions");
  impl_->api = api;

  PJRT_Plugin_Initialize_Args init{};
  init.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
  impl_->check(api->PJRT_Plugin_Initialize(&init), "PJRT_Plugin_Initialize");

  // One client over one cpu device.
  PJRT_NamedValue devices{};
  devices.struct_size = PJRT_NamedValue_STRUCT_SIZE;
  devices.name = "cpu_device_count";
  devices.name_size = std::strlen(devices.name);
  devices.type = PJRT_NamedValue_kInt64;
  devices.int64_value = 1;
  devices.value_size = 1;
  PJRT_Client_Create_Args create{};
  create.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  create.create_options = &devices;
  create.num_options = 1;
  impl_->check(api->PJRT_Client_Create(&create), "PJRT_Client_Create");
  impl_->client = create.client;

  PJRT_Client_AddressableDevices_Args addressable{};
  addressable.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  addressable.client = impl_->client;
  impl_->check(api->PJRT_Client_AddressableDevices(&addressable), "PJRT_Client_AddressableDevices");
  // The first addressable device runs every module.
  if (addressable.num_addressable_devices == 0)
    throw std::runtime_error("fsim: client has no addressable devices");
  impl_->device = addressable.addressable_devices[0];
}

Runtime::~Runtime() = default;

Runtime::Impl::~Impl() {
  if (!client)
    return;
  PJRT_Client_Destroy_Args args{};
  args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  args.client = client;
  check(api->PJRT_Client_Destroy(&args), "PJRT_Client_Destroy");
}

void Runtime::Impl::check(PJRT_Error *error, const char *call) const {
  if (!error)
    return;
  PJRT_Error_Message_Args message{};
  message.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  message.error = error;
  api->PJRT_Error_Message(&message);
  std::string text =
      std::string("fsim: ") + call + ": " + std::string(message.message, message.message_size);
  PJRT_Error_Destroy_Args destroy{};
  destroy.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  destroy.error = error;
  api->PJRT_Error_Destroy(&destroy);
  throw std::runtime_error(text);
}

void Runtime::Impl::await(PJRT_Event *event) const {
  PJRT_Event_Await_Args wait{};
  wait.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  wait.event = event;
  PJRT_Error *error = api->PJRT_Event_Await(&wait);
  PJRT_Event_Destroy_Args destroy{};
  destroy.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
  destroy.event = event;
  PJRT_Error *destroy_error = api->PJRT_Event_Destroy(&destroy);
  if (error && destroy_error) {
    // The await error wins; drop the destroy error.
    PJRT_Error_Destroy_Args drop{};
    drop.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    drop.error = destroy_error;
    api->PJRT_Error_Destroy(&drop);
    destroy_error = nullptr;
  }
  check(error, "PJRT_Event_Await");
  check(destroy_error, "PJRT_Event_Destroy");
}

void Runtime::Impl::destroy(PJRT_Buffer *buffer) const {
  PJRT_Buffer_Destroy_Args args{};
  args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
  args.buffer = buffer;
  check(api->PJRT_Buffer_Destroy(&args), "PJRT_Buffer_Destroy");
}

void Runtime::Impl::destroy(PJRT_Executable *executable) const {
  PJRT_Executable_Destroy_Args args{};
  args.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
  args.executable = executable;
  check(api->PJRT_Executable_Destroy(&args), "PJRT_Executable_Destroy");
}

void Runtime::Impl::destroy(PJRT_LoadedExecutable *executable) const {
  PJRT_LoadedExecutable_Destroy_Args args{};
  args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
  args.executable = executable;
  check(api->PJRT_LoadedExecutable_Destroy(&args), "PJRT_LoadedExecutable_Destroy");
}

} // namespace fsim
