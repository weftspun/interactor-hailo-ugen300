// Flat C over HailoRT's C++ InferModel, so a .sigs file and ctypes can reach the device.
//
// WHY. The flat C API HailoRT already exports is rejected by the Hailo-10H:
// hailo_init_configure_params_by_vdevice returns HAILO_NOT_IMPLEMENTED(7) and the library
// prints "Did you try calling create_configure_params on H10? If so, use InferModel
// instead". InferModel returns Expected<std::shared_ptr<InferModel>>, a template over a
// smart pointer, which ctypes cannot call at any level of effort. This is the smallest
// bridge: five extern "C" functions, declared in hailort_shim.sigs like every other C ABI
// in this workspace.
//
// Errors are returned as hailo_status ints rather than thrown, because a C boundary that
// lets an exception escape terminates the process instead of failing.

#define SHIM_API __declspec(dllexport)

#include <cstdio>
#include <cstring>
#include <vector>
#include <memory>
#include <string>

#include "hailo/hailort.hpp"
#include "hailo/infer_model.hpp"
#include "hailo/vdevice.hpp"

using namespace hailort;

namespace {

struct ShimSession {
    std::vector<uint8_t> hef_blob;
    std::unique_ptr<VDevice> vdevice;
    std::shared_ptr<InferModel> model;
    std::unique_ptr<ConfiguredInferModel> configured;
    std::string last_error;
    size_t in_bytes = 0;
    size_t out_bytes = 0;
};

thread_local std::string g_last_error;

}  // namespace

extern "C" {

// Opens a HEF on a fresh VDevice and configures it. Returns NULL on failure; the reason is
// available from hs_last_error.
// float_io=1 asks the device for float32 tensors instead of the quantised uint8 default,
// so a result can be compared against the same graph in torch rather than only timed.
SHIM_API void *hs_open_tuned(const char *hef_path, int float_io,
                             int ultra_power, int batch_size) {
    try {
        auto session = std::make_unique<ShimSession>();

        auto vdev = VDevice::create();
        if (!vdev) { g_last_error = "VDevice::create failed"; return nullptr; }
        session->vdevice = std::move(vdev.release());

        std::FILE *f = std::fopen(hef_path, "rb");
        if (!f) { g_last_error = "cannot open HEF"; return nullptr; }
        std::fseek(f, 0, SEEK_END);
        session->hef_blob.resize(static_cast<size_t>(std::ftell(f)));
        std::fseek(f, 0, SEEK_SET);
        size_t got = std::fread(session->hef_blob.data(), 1, session->hef_blob.size(), f);
        std::fclose(f);
        if (got != session->hef_blob.size()) { g_last_error = "short read on HEF"; return nullptr; }

        auto model = session->vdevice->create_infer_model(
            MemoryView::create_const(session->hef_blob.data(), session->hef_blob.size()));
        if (!model) { g_last_error = "create_infer_model failed"; return nullptr; }
        session->model = model.release();

        auto in = session->model->input();
        auto out = session->model->output();
        if (!in || !out) { g_last_error = "input()/output() failed"; return nullptr; }
        if (float_io) {
            for (const auto &nm : session->model->get_input_names())
                session->model->input(nm)->set_format_type(HAILO_FORMAT_TYPE_FLOAT32);
            for (const auto &nm : session->model->get_output_names())
                session->model->output(nm)->set_format_type(HAILO_FORMAT_TYPE_FLOAT32);
        }
        if (ultra_power) session->model->set_power_mode(HAILO_POWER_MODE_ULTRA_PERFORMANCE);
        if (batch_size > 0) session->model->set_batch_size(static_cast<uint16_t>(batch_size));

        auto configured = session->model->configure();
        if (!configured) { g_last_error = "configure failed"; return nullptr; }
        session->configured =
            std::make_unique<ConfiguredInferModel>(configured.release());

        session->in_bytes = in->get_frame_size();
        session->out_bytes = out->get_frame_size();

        return session.release();
    } catch (const std::exception &e) {
        g_last_error = e.what();
        return nullptr;
    } catch (...) {
        g_last_error = "unknown exception in hs_open";
        return nullptr;
    }
}

SHIM_API size_t hs_input_size(void *handle) {
    return handle ? static_cast<ShimSession *>(handle)->in_bytes : 0;
}

SHIM_API size_t hs_output_size(void *handle) {
    return handle ? static_cast<ShimSession *>(handle)->out_bytes : 0;
}

// One synchronous inference. Returns HAILO_SUCCESS (0) or a hailo_status.
SHIM_API int hs_infer(void *handle, const void *in_buf, size_t in_n,
                        void *out_buf, size_t out_n, unsigned timeout_ms) {
    if (!handle || !in_buf || !out_buf) return HAILO_INVALID_ARGUMENT;
    auto *s = static_cast<ShimSession *>(handle);
    if (in_n != s->in_bytes || out_n != s->out_bytes) return HAILO_INVALID_ARGUMENT;
    try {
        auto bindings = s->configured->create_bindings();
        if (!bindings) { g_last_error = "create_bindings failed"; return bindings.status(); }
        auto b = bindings.release();

        auto in = b.input();
        if (!in) { g_last_error = "bindings.input failed"; return in.status(); }
        auto st = in->set_buffer(MemoryView(const_cast<void *>(in_buf), in_n));
        if (st != HAILO_SUCCESS) { g_last_error = "set_buffer(in) failed"; return st; }

        auto out = b.output();
        if (!out) { g_last_error = "bindings.output failed"; return out.status(); }
        st = out->set_buffer(MemoryView(out_buf, out_n));
        if (st != HAILO_SUCCESS) { g_last_error = "set_buffer(out) failed"; return st; }

        return s->configured->run(b, std::chrono::milliseconds(timeout_ms));
    } catch (const std::exception &e) {
        g_last_error = e.what();
        return HAILO_INTERNAL_FAILURE;
    } catch (...) {
        g_last_error = "unknown exception in hs_infer";
        return HAILO_INTERNAL_FAILURE;
    }
}

SHIM_API void hs_close(void *handle) {
    delete static_cast<ShimSession *>(handle);
}

SHIM_API const char *hs_last_error(void) {
    return g_last_error.c_str();
}


// --- the llama.cpp mtmd contract: 4 same-shaped outputs and an embedded config ----------

SHIM_API int hs_num_outputs(void *handle) {
    if (!handle) return -1;
    auto *s = static_cast<ShimSession *>(handle);
    try {
        return static_cast<int>(s->model->get_output_names().size());
    } catch (...) { g_last_error = "get_output_names failed"; return -1; }
}

// Writes output i's stream name into buf. Returns the length, or the needed length if buf
// is too small, so a caller can size its buffer without guessing.
SHIM_API int hs_output_name(void *handle, int i, char *buf, size_t n) {
    if (!handle) return -1;
    auto *s = static_cast<ShimSession *>(handle);
    try {
        auto names = s->model->get_output_names();
        if (i < 0 || static_cast<size_t>(i) >= names.size()) return -1;
        const std::string &nm = names[i];
        if (buf && n > nm.size()) { std::memcpy(buf, nm.c_str(), nm.size() + 1); }
        return static_cast<int>(nm.size());
    } catch (...) { g_last_error = "hs_output_name failed"; return -1; }
}

SHIM_API int hs_output_shape(void *handle, int i, unsigned *h, unsigned *w, unsigned *f) {
    if (!handle || !h || !w || !f) return -1;
    auto *s = static_cast<ShimSession *>(handle);
    try {
        auto names = s->model->get_output_names();
        if (i < 0 || static_cast<size_t>(i) >= names.size()) return -1;
        auto o = s->model->output(names[i]);
        if (!o) { g_last_error = "output() failed"; return -1; }
        auto sh = o->shape();
        *h = sh.height; *w = sh.width; *f = sh.features;
        return 0;
    } catch (...) { g_last_error = "hs_output_shape failed"; return -1; }
}

// Reads a named external resource out of the HEF -- hailo-config.json is the one mtmd
// requires. Returns its byte length, -1 when absent, so "missing" and "empty" stay distinct.
SHIM_API int hs_resource(void *handle, const char *name, char *buf, size_t n) {
    if (!handle || !name) return -1;
    auto *s = static_cast<ShimSession *>(handle);
    try {
        auto view = s->model->hef().get_external_resources(std::string(name));
        if (!view) { g_last_error = "resource absent"; return -1; }
        auto v = view.release();
        if (buf && n >= v.size()) std::memcpy(buf, v.data(), v.size());
        return static_cast<int>(v.size());
    } catch (const std::exception &e) { g_last_error = e.what(); return -1; }
      catch (...) { g_last_error = "hs_resource failed"; return -1; }
}


// --- pipelined inference -------------------------------------------------------------
// One synchronous call pays a full USB round trip before the next starts. This queues
// n_frames of work and waits once, so the transfers of one frame overlap the compute of
// another. Returns the device's async queue depth from hs_queue_size, since queueing more
// than that cannot help.

SHIM_API int hs_queue_size(void *handle) {
    if (!handle) return -1;
    auto *s = static_cast<ShimSession *>(handle);
    auto q = s->configured->get_async_queue_size();
    return q ? static_cast<int>(q.value()) : -1;
}

SHIM_API int hs_infer_batch(void *handle, const void *in_buf, void *out_buf,
                            int n_frames, unsigned timeout_ms) {
    if (!handle || !in_buf || !out_buf || n_frames < 1) return HAILO_INVALID_ARGUMENT;
    auto *s = static_cast<ShimSession *>(handle);
    try {
        const auto timeout = std::chrono::milliseconds(timeout_ms);
        std::vector<ConfiguredInferModel::Bindings> held;
        held.reserve(n_frames);
        AsyncInferJob last;

        for (int i = 0; i < n_frames; ++i) {
            auto st = s->configured->wait_for_async_ready(timeout, 1);
            if (st != HAILO_SUCCESS) { g_last_error = "wait_for_async_ready failed"; return st; }

            auto bindings = s->configured->create_bindings();
            if (!bindings) { g_last_error = "create_bindings failed"; return bindings.status(); }
            auto b = bindings.release();

            auto in = b.input();
            if (!in) return in.status();
            st = in->set_buffer(MemoryView(
                const_cast<uint8_t *>(static_cast<const uint8_t *>(in_buf)) + i * s->in_bytes,
                s->in_bytes));
            if (st != HAILO_SUCCESS) { g_last_error = "set_buffer(in) failed"; return st; }

            auto out = b.output();
            if (!out) return out.status();
            st = out->set_buffer(MemoryView(
                static_cast<uint8_t *>(out_buf) + i * s->out_bytes, s->out_bytes));
            if (st != HAILO_SUCCESS) { g_last_error = "set_buffer(out) failed"; return st; }

            auto job = s->configured->run_async(b);
            if (!job) { g_last_error = "run_async failed"; return job.status(); }
            last = job.release();
            held.push_back(std::move(b));
        }
        last.detach();
        return last.wait(timeout);
    } catch (const std::exception &e) {
        g_last_error = e.what();
        return HAILO_INTERNAL_FAILURE;
    }
}


// Pins caller memory for DMA. Without this HailoRT copies every frame through a bounce
// buffer, which is invisible at 16 KB and is most of the cost at 33 MB.
SHIM_API int hs_dma_map(void *handle, void *buf, size_t n, int to_device) {
    if (!handle || !buf) return HAILO_INVALID_ARGUMENT;
    auto *s = static_cast<ShimSession *>(handle);
    return s->vdevice->dma_map(buf, n, to_device ? HAILO_DMA_BUFFER_DIRECTION_H2D
                                                 : HAILO_DMA_BUFFER_DIRECTION_D2H);
}

SHIM_API int hs_dma_unmap(void *handle, void *buf, size_t n, int to_device) {
    if (!handle || !buf) return HAILO_INVALID_ARGUMENT;
    auto *s = static_cast<ShimSession *>(handle);
    return s->vdevice->dma_unmap(buf, n, to_device ? HAILO_DMA_BUFFER_DIRECTION_H2D
                                                   : HAILO_DMA_BUFFER_DIRECTION_D2H);
}

SHIM_API void *hs_open_ex(const char *hef_path, int float_io) {
    return hs_open_tuned(hef_path, float_io, 0, 0);
}

SHIM_API void *hs_open(const char *hef_path) { return hs_open_ex(hef_path, 0); }

}  // extern "C"
