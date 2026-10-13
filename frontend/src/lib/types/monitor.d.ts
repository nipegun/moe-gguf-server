/**
 * Monitor API types (`/monitor`, `/monitor/usage`).
 * Byte counts and percentages come straight from the server.
 */

/** Memory segment keys published by the server, in drawing order. */
export type MonitorSegmentKey =
  | 'mtp_weights'
  | 'expert_weights'
  | 'dense_weights'
  | 'draft_weights'
  | 'kv_cache'
  | 'draft_kv_cache'
  | 'compute'
  | 'expert_cache'
  | 'prefetch'
  | 'prompt_cache'
  | 'other';

export type MonitorSegments = Partial<Record<MonitorSegmentKey, number>>;

/** One logical CPU of a processor: its number and its usage in percent. */
export interface MonitorCpuCore {
  id: number;
  usage: number;
}

/** One physical processor (socket) with its logical CPUs, in package order. */
export interface MonitorCpuProcessor {
  id: number;
  model: string;
  usage: number | null;
  logical_cpus: number;
  cores: MonitorCpuCore[];
}

/** Global usage of every logical CPU, the system load average and one entry per processor. */
export interface MonitorCpu {
  usage: number | null;
  logical_cpus: number;
  load_average?: [number, number, number] | null;
  processors: MonitorCpuProcessor[];
}

/** A memory pool drawn as one bar: a GPU or the RAM. */
export interface MonitorMemoryPool {
  total_bytes: number;
  used_bytes: number;
  free_bytes: number;
  segments: MonitorSegments;
}

export interface MonitorRam extends MonitorMemoryPool {
  available_bytes: number;
  swap_total_bytes: number;
  swap_used_bytes: number;
}

/** Where the kernel DRM driver keeps what the models allocated on a GPU (amdgpu, i915, xe; not NVIDIA). */
export interface MonitorGpuKernelResidency {
  allocated_bytes: number;
  vram_bytes: number;
  gtt_bytes: number;
  evicted_vram_bytes: number;
  outside_vram_bytes: number;
}

export interface MonitorGpu extends MonitorMemoryPool {
  id: string;
  name: string;
  source: 'nvml' | 'sysfs' | 'backend';
  usage: number | null;
  memory_usage: number | null;
  temperature_c: number | null;
  power_w: number | null;
  power_limit_w: number | null;
  backends: string[];
  kernel?: MonitorGpuKernelResidency;
}

export interface MonitorExpertCache {
  active: boolean;
  size_bytes: number;
  layers: number;
  slots: number;
  experts: number;
  hit_ratio: number;
  uploads: number;
  cpu_experts: number;
  upload_bandwidth_bytes_per_second: number;
  cpu_bandwidth_bytes_per_second: number;
}

export interface MonitorModel {
  name: string;
  state: string;
  rss_bytes?: number;
  expert_cache?: MonitorExpertCache;
}

export interface MonitorReport {
  timestamp_ms: number;
  cpu: MonitorCpu;
  ram: MonitorRam;
  gpus: MonitorGpu[];
  models: MonitorModel[];
  segments: MonitorSegmentKey[];
}

export interface MonitorUsage {
  cpu: {
    usage: number | null;
    processors: { id: number; model: string; usage: number | null }[];
  };
  gpus: { id: string; name: string; usage: number | null }[];
}
