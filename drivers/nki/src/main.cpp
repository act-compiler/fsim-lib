// fsim-nki [--inputs DIR] [--outputs DIR] [--trigger N] [--overlap T]
// [--validate] [<trace>|-]: simulates one trace, a file or stdin.
// --validate parses and checks it only, and takes no directories.
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "Driver.h"
#include "fsim/pipeline.h"

namespace {

int usage() {
  std::cerr << "usage: fsim-nki [--inputs DIR] [--outputs DIR] [--trigger N] [--overlap T] "
               "[--validate] [<trace>|-]\n";
  return 1;
}

} // namespace

int main(int argc, char **argv) {
  argc = fsim::Knobs::parse(argc, argv);
  std::string trace, inputs, outputs;
  bool validate = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--validate") {
      validate = true;
    } else if (arg == "--inputs" || arg == "--outputs") {
      if (i + 1 >= argc)
        return usage();
      (arg == "--inputs" ? inputs : outputs) = argv[++i];
    } else if (arg.starts_with("--") || !trace.empty()) {
      return usage();
    } else {
      trace = arg;
    }
  }
  // A simulation takes both directories, validation neither.
  if (validate ? !inputs.empty() || !outputs.empty() : inputs.empty() || outputs.empty())
    return usage();
  if (trace.empty())
    trace = "-";
  std::ios::sync_with_stdio(false);
  try {
    if (!validate)
      std::filesystem::create_directories(outputs);
    nki::Driver driver(inputs, outputs, validate);
    if (trace == "-") {
      driver.run(std::cin);
    } else {
      std::ifstream in(trace);
      if (!in)
        throw std::runtime_error("cannot read " + trace);
      driver.run(in);
    }
  } catch (const std::exception &e) {
    std::cerr << "fsim-nki: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
