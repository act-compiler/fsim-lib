// fsim-passes: the pass stage over one module file.
// Prints the op counts and each phase's ms to stderr, each pass's with
// --trace and the final op histogram with --histogram, and writes the
// artifact, or the module's text with --text, to -o.
//   fsim-passes <in.mlir> [-o <out>] [--text] [--trace] [--histogram]
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>

#include "stage.h"

namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void usage() {
  std::cerr << "usage: fsim-passes <in.mlir> [-o <out>] [--text] [--trace] [--histogram]\n";
  std::exit(1);
}

} // namespace

int main(int argc, char **argv) {
  const char *in = nullptr;
  const char *out = nullptr;
  bool text = false;
  bool trace = false;
  bool histogram = false;
  for (int i = 1; i < argc; ++i) {
    std::string_view arg = argv[i];
    if (arg == "-o" && i + 1 < argc)
      out = argv[++i];
    else if (arg == "--text")
      text = true;
    else if (arg == "--trace")
      trace = true;
    else if (arg == "--histogram")
      histogram = true;
    else if (!in && !arg.starts_with("-"))
      in = argv[i];
    else
      usage();
  }
  if (!in)
    usage();

  // The whole file as the module's text.
  std::ifstream file(in, std::ios::binary);
  if (!file) {
    std::cerr << "fsim-passes: cannot read " << in << "\n";
    return 1;
  }
  std::string source{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};

  // Times one phase; a failure prints its error and exits 1.
  fsim::Stage stage;
  if (trace)
    stage.trace();
  auto phase = [&](const char *name, auto &&run) {
    Clock::time_point begin = Clock::now();
    bool ok = run();
    double ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    if (!ok) {
      std::cerr << "fsim-passes: " << stage.error() << "\n";
      std::exit(1);
    }
    std::cerr << name << " " << ms << " ms\n";
  };

  phase("parse", [&] { return stage.parse(source); });
  size_t before = stage.ops();
  phase("stock passes", [&] { return stage.stock(); });
  phase("library passes", [&] { return stage.library(); });
  size_t after = stage.ops();
  std::string result;
  if (text)
    phase("print", [&] {
      stage.print(result);
      return true;
    });
  else
    phase("serialize", [&] { return stage.serialize(result); });
  std::cerr << "ops " << before << " -> " << after << ", bytes " << source.size() << " -> "
            << result.size() << "\n";
  if (histogram)
    for (auto &[name, count] : stage.histogram())
      std::cerr << "  " << count << " " << name << "\n";

  if (out) {
    std::ofstream sink(out, std::ios::binary);
    sink.write(result.data(), result.size());
    if (!sink) {
      std::cerr << "fsim-passes: cannot write " << out << "\n";
      return 1;
    }
  }
  return 0;
}
