/* The WebGPU calls of sim/gpu.goose. Goose cannot build the webgpu.h
   descriptors (they hold pointers) or take a callback, so this C does it.
   It has no solver logic: sim/gpu.goose decides the buffers, the kernels
   and their sequence. Handles are small indices into the tables below.

   The device comes from the page (Module.preinitializedWebGPUDevice), so
   nothing here waits for a promise. The one asynchronous call, the map
   of a read-back buffer, sets map_state of that buffer from its callback,
   and the page polls it. */
#include <stdint.h>
#include <string.h>
#include <webgpu/webgpu.h>

/* the slices of sim/gpu.goose, as Goose passes them: { data, len }, packed */
#pragma pack(push, 1)
typedef struct { const uint8_t *data; int64_t len; } gw_bytes;
typedef struct { const int32_t *data; int64_t len; } gw_ints;
typedef struct { float *data; int64_t len; } gw_floats;
#pragma pack(pop)

enum { MAXBUF = 32, MAXPIPE = 16, MAXGROUP = 16, MAXENTRY = 16 };

static WGPUDevice device;
static WGPUQueue queue;
static WGPUShaderModule module;
static WGPUBuffer bufs[MAXBUF];
static int nbuf;
static WGPUComputePipeline pipes[MAXPIPE];
static int npipe;
static WGPUBindGroup groups[MAXGROUP];
static int ngroup;
static WGPUCommandEncoder enc;
static WGPUComputePassEncoder pass;
static int map_state[MAXBUF];   /* 0: none or waiting, 1: mapped, -1: failed */
static intptr_t setups; /* gw_reset calls: a map of an older set-up is stale */

static WGPUStringView view(const uint8_t *s, int64_t n) {
    WGPUStringView v = { (const char *)s, (size_t)n };
    return v;
}

int32_t gw_init(gw_bytes wgsl) {
    device = emscripten_webgpu_get_device();
    if (!device) return 0;
    queue = wgpuDeviceGetQueue(device);
    WGPUShaderSourceWGSL src = WGPU_SHADER_SOURCE_WGSL_INIT;
    src.code = view(wgsl.data, wgsl.len);
    WGPUShaderModuleDescriptor md = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    md.nextInChain = &src.chain;
    module = wgpuDeviceCreateShaderModule(device, &md);
    return module != NULL;
}

/* a compute pipeline for one entry point, with the layout from the WGSL */
int32_t gw_pipeline(gw_bytes entry) {
    if (npipe == MAXPIPE) return -1;
    WGPUComputePipelineDescriptor pd = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    pd.compute.module = module;
    pd.compute.entryPoint = view(entry.data, entry.len);
    pipes[npipe] = wgpuDeviceCreateComputePipeline(device, &pd);
    return npipe++;
}

/* Releases the buffers and bind groups of the last set-up. */
void gw_reset(void) {
    for (int i = 0; i < ngroup; i++) wgpuBindGroupRelease(groups[i]);
    for (int i = 0; i < nbuf; i++) { wgpuBufferDestroy(bufs[i]); wgpuBufferRelease(bufs[i]); }
    ngroup = 0;
    nbuf = 0;
    memset(map_state, 0, sizeof map_state);
    setups++;
}

/* kind 0: storage, 1: uniform, 2: read-back. WebGPU zeroes a new buffer.
   An empty array (no panel in the wall run) still gets a buffer, of at
   least one element of the largest struct of the WGSL. */
int32_t gw_buffer(int64_t size, int32_t kind) {
    if (nbuf == MAXBUF) return -1;
    WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
    bd.size = (uint64_t)((size + 15) / 16 * 16);
    if (bd.size < 32) bd.size = 32;
    bd.usage = kind == 0 ? WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc
             : kind == 1 ? WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst
             : WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    bufs[nbuf] = wgpuDeviceCreateBuffer(device, &bd);
    return nbuf++;
}

void gw_write(int32_t buf, int64_t offset, gw_bytes data) {
    wgpuQueueWriteBuffer(queue, bufs[buf], (uint64_t)offset, data.data, (size_t)data.len);
}

/* A bind group for group 0 of a pipeline. pairs: (binding, buffer) ... */
int32_t gw_bind(int32_t pipe, gw_ints pairs) {
    if (ngroup == MAXGROUP || pairs.len > 2 * MAXENTRY) return -1;
    WGPUBindGroupEntry es[MAXENTRY];
    int n = (int)(pairs.len / 2);
    for (int i = 0; i < n; i++) {
        WGPUBindGroupEntry e = WGPU_BIND_GROUP_ENTRY_INIT;
        e.binding = (uint32_t)pairs.data[2 * i];
        e.buffer = bufs[pairs.data[2 * i + 1]];
        es[i] = e;
    }
    WGPUBindGroupDescriptor gd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    gd.layout = wgpuComputePipelineGetBindGroupLayout(pipes[pipe], 0);
    gd.entryCount = (size_t)n;
    gd.entries = es;
    groups[ngroup] = wgpuDeviceCreateBindGroup(device, &gd);
    wgpuBindGroupLayoutRelease(gd.layout);
    return ngroup++;
}

static void end_pass(void) {
    if (!pass) return;
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    pass = NULL;
}

/* Records a dispatch of `n` workgroups. Dispatches go into one compute
   pass, and each one sees the writes of the ones before it. */
void gw_dispatch(int32_t pipe, int32_t group, int64_t n) {
    if (!enc) enc = wgpuDeviceCreateCommandEncoder(device, NULL);
    if (!pass) pass = wgpuCommandEncoderBeginComputePass(enc, NULL);
    if (n <= 0) return;
    wgpuComputePassEncoderSetPipeline(pass, pipes[pipe]);
    wgpuComputePassEncoderSetBindGroup(pass, 0, groups[group], 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(pass, (uint32_t)n, 1, 1);
}

void gw_copy(int32_t src, int64_t src_off, int32_t dst, int64_t dst_off, int64_t size) {
    if (!enc) enc = wgpuDeviceCreateCommandEncoder(device, NULL);
    end_pass();
    if (size > 0) wgpuCommandEncoderCopyBufferToBuffer(enc, bufs[src], (uint64_t)src_off, bufs[dst], (uint64_t)dst_off, (uint64_t)size);
}

void gw_submit(void) {
    if (!enc) return;
    end_pass();
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    enc = NULL;
}

static void mapped(WGPUMapAsyncStatus status, WGPUStringView message, void *u1, void *u2) {
    (void)message;
    if ((intptr_t)u1 != setups) return;
    map_state[(intptr_t)u2] = status == WGPUMapAsyncStatus_Success ? 1 : -1;
}

/* Maps a read-back buffer after the work submitted so far. */
void gw_map(int32_t buf) {
    map_state[buf] = 0;
    WGPUBufferMapCallbackInfo ci = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    ci.mode = WGPUCallbackMode_AllowSpontaneous;
    ci.callback = mapped;
    ci.userdata1 = (void *)setups;
    ci.userdata2 = (void *)(intptr_t)buf;
    wgpuBufferMapAsync(bufs[buf], WGPUMapMode_Read, 0, (size_t)wgpuBufferGetSize(bufs[buf]), ci);
}

int32_t gw_mapped(int32_t buf) { return map_state[buf]; }

void gw_read(int32_t buf, int64_t offset, gw_floats out) {
    wgpuBufferReadMappedRange(bufs[buf], (size_t)offset, out.data, (size_t)out.len * sizeof(float));
}

void gw_unmap(int32_t buf) {
    wgpuBufferUnmap(bufs[buf]);
    map_state[buf] = 0;
}
