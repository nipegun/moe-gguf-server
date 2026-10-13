#pragma once

// System monitoring for the web UI Monitor and status bar.
//
// CPU usage comes from /proc/stat deltas, RAM from /proc/meminfo and GPU usage from NVML
// (libnvidia-ml.so.1, opened at run time) or from the amdgpu sysfs files. None of these
// readings touch the inference contexts or the task queue, so polling them never wakes a
// sleeping model nor delays its idle timer, and the router reads them without creating a
// CUDA context of its own.
//
// The memory used by the models comes from snapshots that every inference process
// publishes from its main loop (see server_context_impl::mRefreshMonitorSnapshot):
// fMonitorReport adds them to the device and RAM totals of the system.

#include "server-common.h"

#include <vector>

// Memory segment keys, in drawing order. "other" (memory used by anything else) is computed
// by fMonitorReport; the rest of the device or RAM is free.
extern const char *const aMonitorSegments[];
extern const size_t cMonitorSegmentCount;

// CPU usage (global and per physical processor) and GPU usage, for the status bar.
json fMonitorUsageJson();

// Full report: CPU, RAM and GPUs, with the memory segments of every snapshot
// ({"state", "model", "devices", "host", ...}) added to the matching device and to the RAM.
json fMonitorReport(const std::vector<json> &pLSnapshots);

// Resident memory of the calling process, in bytes (0 when unknown).
size_t fMonitorProcessRss();

// Where the kernel DRM driver keeps the GPU memory of the calling process (/proc/self/fdinfo), keyed by
// fMonitorDeviceKey of the PCI id: {"vram_bytes", "gtt_bytes"[, "evicted_vram_bytes"]}. amdgpu, i915 and xe
// publish it; the NVIDIA driver does not, so CUDA devices are absent.
json fMonitorDrmResidency();

// Device identity shared by the GGML backends, NVML, sysfs and fdinfo.
std::string fMonitorDeviceKey(const std::string &pId);
