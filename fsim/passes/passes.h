// The pass stage's C entry, the only symbols its shared object exports.
// The stage parses a StableHLO module's text, simplifies it, aliases @main's
// arguments to its results, and serializes a versioned StableHLO artifact.
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Runs the stage over text; 0 on success, nonzero on failure.
// *out gets a malloc'd buffer of *out_size bytes: the artifact, or the
// failure's diagnostics. Free it with fsim_passes_free.
__attribute__((visibility("default"))) int fsim_passes_run(const char *text, size_t size,
                                                           char **out, size_t *out_size);
// Frees a buffer fsim_passes_run returned.
__attribute__((visibility("default"))) void fsim_passes_free(char *buffer);

#ifdef __cplusplus
}
#endif
