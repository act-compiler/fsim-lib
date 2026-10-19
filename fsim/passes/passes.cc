// The pass stage and its C entry.
// One single-threaded context per module.
#include "passes.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>

#include "fsim_passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/optimization/Passes.h"
#include "stage.h"
#include "llvm/Support/raw_ostream.h"

namespace fsim {
namespace {

// The artifact's StableHLO version, the one jax 0.11.2's XLA accepts.
constexpr char kTargetVersion[] = "1.20.0";

// The dialects a submitted module uses.
mlir::DialectRegistry dialects() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::stablehlo::StablehloDialect>();
  return registry;
}

// The ops under op's root, the root included.
size_t count(mlir::Operation *op) {
  while (mlir::Operation *parent = op->getParentOp())
    op = parent;
  size_t ops = 0;
  op->walk([&](mlir::Operation *) { ++ops; });
  return ops;
}

// Prints each named pass's ms and the module's op count after it.
class Trace : public mlir::PassInstrumentation {
public:
  void runBeforePass(mlir::Pass *, mlir::Operation *) override {
    begins_.push_back(Clock::now());
  }
  void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override {
    double ms = std::chrono::duration<double, std::milli>(Clock::now() - begins_.back()).count();
    begins_.pop_back();
    // Adaptors over nested passes have no name.
    if (!pass->getArgument().empty())
      llvm::errs() << "  " << pass->getArgument() << " " << ms << " ms, " << count(op) << " ops\n";
  }

private:
  using Clock = std::chrono::steady_clock;
  std::vector<Clock::time_point> begins_;
};

} // namespace

Stage::Stage()
    : context_(dialects(), mlir::MLIRContext::Threading::DISABLED),
      handler_(&context_, [this](mlir::Diagnostic &diagnostic) {
        // Every diagnostic and its notes, one per line.
        llvm::raw_string_ostream os(diagnostics_);
        os << diagnostic.getLocation() << ": " << diagnostic << "\n";
        for (mlir::Diagnostic &note : diagnostic.getNotes())
          os << note.getLocation() << ": note: " << note << "\n";
        return mlir::success();
      }) {
}

bool Stage::parse(std::string_view text) {
  module_ = mlir::parseSourceString<mlir::ModuleOp>(llvm::StringRef(text.data(), text.size()),
                                                    mlir::ParserConfig(&context_), "module");
  return module_ || fail("parse");
}

bool Stage::stock() {
  mlir::PassManager pm(&context_);
  pm.addPass(mlir::createCSEPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createStablehloTargetIndependentOptimizationPass({}, {}));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  return run(pm) || fail("stock passes");
}

bool Stage::library() {
  mlir::PassManager pm(&context_);
  addPasses(pm);
  if (!run(pm))
    return fail("library passes");

  // Argument i aliases result i: the state updates in place.
  auto main = module_->lookupSymbol<mlir::func::FuncOp>("main");
  if (!main) {
    mlir::emitError(module_->getLoc()) << "no @main";
    return fail("aliasing");
  }
  mlir::Builder builder(&context_);
  for (unsigned i = 0; i < main.getNumArguments(); ++i)
    main.setArgAttr(i, "tf.aliasing_output", builder.getI32IntegerAttr(i));
  return true;
}

bool Stage::serialize(std::string &artifact) {
  llvm::raw_string_ostream os(artifact);
  return mlir::succeeded(
             mlir::stablehlo::serializePortableArtifact(*module_, kTargetVersion, os)) ||
         fail("serialize");
}

void Stage::print(std::string &text) {
  llvm::raw_string_ostream os(text);
  module_->print(os);
}

size_t Stage::ops() {
  size_t count = 0;
  module_->walk([&](mlir::Operation *) { ++count; });
  return count;
}

std::vector<std::pair<std::string, size_t>> Stage::histogram() {
  std::map<std::string, size_t> counts;
  module_->walk([&](mlir::Operation *op) { ++counts[op->getName().getStringRef().str()]; });
  std::vector<std::pair<std::string, size_t>> sorted(counts.begin(), counts.end());
  std::stable_sort(sorted.begin(), sorted.end(),
                   [](const auto &a, const auto &b) { return a.second > b.second; });
  return sorted;
}

bool Stage::run(mlir::PassManager &pm) {
  if (trace_)
    pm.addInstrumentation(std::make_unique<Trace>());
  return mlir::succeeded(pm.run(*module_));
}

bool Stage::fail(std::string_view phase) {
  error_ = std::string(phase) + " failed: " + diagnostics_;
  if (error_.back() == '\n')
    error_.pop_back();
  return false;
}

} // namespace fsim

namespace {

// Hands bytes to the C caller as a malloc'd, NUL-terminated buffer.
int give(int status, const std::string &bytes, char **out, size_t *out_size) {
  *out = static_cast<char *>(std::malloc(bytes.size() + 1));
  std::memcpy(*out, bytes.data(), bytes.size());
  (*out)[bytes.size()] = '\0';
  *out_size = bytes.size();
  return status;
}

} // namespace

int fsim_passes_run(const char *text, size_t size, char **out, size_t *out_size) {
  fsim::Stage stage;
  std::string artifact;
  bool ok =
      stage.parse({text, size}) && stage.stock() && stage.library() && stage.serialize(artifact);
  return ok ? give(0, artifact, out, out_size) : give(1, stage.error(), out, out_size);
}

void fsim_passes_free(char *buffer) {
  std::free(buffer);
}
