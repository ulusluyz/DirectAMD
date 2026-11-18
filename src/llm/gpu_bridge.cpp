#include "directamd/command.hpp"
#include "directamd/device.hpp"
#include "directamd/isa.hpp"
#include "directamd/memory.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <drm/amdgpu_drm.h>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace DirectAMD;
thread_local std::string last_error;
constexpr size_t guard = 64;
constexpr float sentinel = -123456.0f;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check(const std::error_code& ec) {
    if (ec) throw std::system_error(ec);
}

struct State {
    std::unique_ptr<Device> device;
    VAManager va;
    uint32_t context = 0;
    uint64_t resident = 0;
    uint64_t peak = 0;
    uint64_t vram_bytes = 0;
    uint64_t gtt_bytes = 0;
    uint64_t submissions = 0;
    uint64_t waits = 0;
    uint64_t budget = 0;
    bool poisoned = false;
    std::map<int, std::unique_ptr<BufferObject>> kernels;

    explicit State(const char* path, uint64_t limit) {
        std::error_code ec;
        device = Device::open_path(path, ec);
        check(ec);
        require(device && device->get_properties().arch_family == ArchitectureFamily::GCN4_POLARIS,
                "LLM backend requires Polaris gfx803");
        budget = std::min<uint64_t>(limit, device->get_properties().vram_size_bytes - uint64_t{256 * 1024 * 1024});
        require(budget > 0, "Invalid GPU memory budget");
        context = device->create_context(ec);
        check(ec);
    }
    ~State() {
        kernels.clear();
        if (device) {
            std::error_code ec;
            device->destroy_context(context, ec);
        }
    }
};

struct Allocation {
    std::shared_ptr<State> state;
    std::unique_ptr<BufferObject> bo;
    size_t count;
    bool vram;

    Allocation(std::shared_ptr<State> owner, size_t elements, uint32_t domain)
        : state(std::move(owner)), count(elements), vram(domain == AMDGPU_GEM_DOMAIN_VRAM) {
        require(count > 0 && count <= (size_t{1} << 30) - 2 * guard, "Invalid GPU tensor size");
        const uint64_t bytes = (count + 2 * guard) * sizeof(float);
        const uint64_t aligned = (bytes + 4095) & ~uint64_t{4095};
        require(aligned <= state->budget && state->resident <= state->budget - aligned,
                "GPU memory budget exceeded");
        std::error_code ec;
        bo = BufferObject::create(*state->device, state->va, bytes, domain, ec, !vram);
        check(ec);
        require(bo && (vram || bo->get_cpu_ptr()), "GPU allocation has no host mapping");
        state->resident += bo->get_size();
        (vram ? state->vram_bytes : state->gtt_bytes) += bo->get_size();
        state->peak = std::max(state->peak, state->resident);
        if (vram) return; // Device-local storage is initialized through bounded GTT staging.
        auto* host = static_cast<float*>(bo->get_cpu_ptr());
        std::fill_n(host, guard, sentinel);
        std::fill_n(host + guard, count, 0.0f);
        std::fill_n(host + guard + count, guard, sentinel);
    }
    ~Allocation() {
        if (bo) {
            state->resident -= bo->get_size();
            (vram ? state->vram_bytes : state->gtt_bytes) -= bo->get_size();
        }
    }
    void validate() const;
};

struct Tensor {
    std::shared_ptr<Allocation> allocation;
    size_t offset;
    size_t count;
    uint64_t address() const {
        return allocation->bo->get_gpu_va() + (guard + offset) * sizeof(float);
    }
    float* host() const {
        require(!allocation->vram, "Device-local VRAM requires staged transfer");
        return static_cast<float*>(allocation->bo->get_cpu_ptr()) + guard + offset;
    }
};

using Handle = std::shared_ptr<State>;
Tensor& tensor(void* handle) {
    require(handle != nullptr, "Null tensor handle");
    return *static_cast<Tensor*>(handle);
}

void same_device(const State& state, const Tensor& t) {
    require(t.allocation->state.get() == &state, "Tensors belong to different GPU sessions");
}

std::vector<uint32_t> code_for(int kind) {
    switch (kind) {
        case 0: return ISA::PolarisEncoder::emit_vector_add_kernel();
        case 1: return ISA::PolarisEncoder::emit_vector_mul_kernel();
        case 2: return ISA::PolarisEncoder::emit_gemv_kernel();
        case 3: return ISA::PolarisEncoder::emit_gemm_kernel();
        case 4: return ISA::PolarisEncoder::emit_rmsnorm_kernel();
        case 5: return ISA::PolarisEncoder::emit_rope_kernel();
        case 6: return ISA::PolarisEncoder::emit_silu_kernel();
        case 7: return ISA::PolarisEncoder::emit_softmax_kernel();
        default: throw std::runtime_error("Invalid operation");
    }
}

template<class Build>
void dispatch(State& state, int kind, const std::vector<Tensor*>& tensors, Build build, bool validate_guards = true) {
    require(!state.poisoned, "GPU session is poisoned after a failed command");
    std::error_code ec;
    if (!state.kernels.contains(kind)) {
        const auto code = code_for(kind);
        auto kernel = BufferObject::create(*state.device, state.va, code.size() * sizeof(uint32_t),
                                            AMDGPU_GEM_DOMAIN_GTT, ec);
        check(ec);
        require(kernel != nullptr, "Kernel allocation failed");
        kernel->copy_to_gpu(code.data(), code.size() * sizeof(uint32_t), ec);
        check(ec);
        state.kernels.emplace(kind, std::move(kernel));
    }
    auto& kernel = state.kernels.at(kind);
    PM4Builder pm4;
    build(pm4, kernel->get_gpu_va());
    auto ib = BufferObject::create(*state.device, state.va, pm4.get_size_bytes(), AMDGPU_GEM_DOMAIN_GTT, ec);
    check(ec);
    require(ib != nullptr, "IB allocation failed");
    ib->copy_to_gpu(pm4.get_packets().data(), pm4.get_size_bytes(), ec);
    check(ec);
    std::vector<BufferObject*> buffers{kernel.get()};
    for (auto* t : tensors) {
        same_device(state, *t);
        if (validate_guards) t->allocation->validate();
        if (std::find(buffers.begin(), buffers.end(), t->allocation->bo.get()) == buffers.end()) {
            buffers.push_back(t->allocation->bo.get());
        }
    }
    const uint64_t fence = CommandSubmission::submit_compute(*state.device, state.context, *ib, buffers, ec);
    if (ec || fence == 0) {
        state.poisoned = true;
        throw std::runtime_error("Compute submission failed: " + ec.message());
    }
    ++state.submissions;
    if (!CommandSubmission::wait_fence(*state.device, state.context, fence, 5'000'000'000ULL, ec)) {
        state.poisoned = true;
        throw std::runtime_error("Compute fence failed: " + ec.message());
    }
    ++state.waits;
    for (auto* t : tensors) if (validate_guards) t->allocation->validate();
}

// Transfers reuse the existing GPU vector-add kernel with a zero operand.
// No CPU mapping of VRAM weights, avoiding the 256 MiB visible-BAR limitation.
void staged_transfer(Allocation& allocation, size_t physical_offset, float* data,
                     size_t count, bool upload) {
    auto owner = allocation.state;
    auto staging = std::make_shared<Allocation>(owner, count, AMDGPU_GEM_DOMAIN_GTT);
    auto zeros = std::make_shared<Allocation>(owner, count, AMDGPU_GEM_DOMAIN_GTT);
    Tensor stage{staging, 0, count}, zero{zeros, 0, count};
    if (upload) std::memcpy(stage.host(), data, count * sizeof(float));
    // Keep the target allocation alive through the caller; the alias never owns it.
    auto alias = std::shared_ptr<Allocation>(&allocation, [](Allocation*) {});
    Tensor device{alias, 0, allocation.count};
    const auto device_address = allocation.bo->get_gpu_va() + physical_offset * sizeof(float);
    dispatch(*owner, 0, {&stage, &zero, &device}, [&](PM4Builder& pm4, uint64_t program) {
        pm4.add_polaris_vector_add(program, {upload ? stage.address() : device_address,
            zero.address(), upload ? device_address : stage.address(), static_cast<uint32_t>(count)});
    }, false);
    staging->validate(); zeros->validate();
    if (!upload) std::memcpy(data, stage.host(), count * sizeof(float));
}

void Allocation::validate() const {
    if (vram) {
        float values[guard];
        for (const size_t offset : {size_t{0}, guard + count}) {
            staged_transfer(*const_cast<Allocation*>(this), offset, values, guard, false);
            for (const float value : values) require(value == sentinel, "GPU tensor guard corrupted");
        }
    } else {
        const auto* p = static_cast<const float*>(bo->get_cpu_ptr());
        for (size_t i = 0; i < guard; ++i)
            require(p[i] == sentinel && p[guard + count + i] == sentinel, "GPU tensor guard corrupted");
    }
}

template<class Function>
int safely(Function operation) {
    try {
        operation();
        last_error.clear();
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

} // namespace

extern "C" {

const char* damd_error() { return last_error.c_str(); }

void* damd_open(const char* path, uint64_t budget) {
    Handle* result = nullptr;
    safely([&] { result = new Handle(std::make_shared<State>(path, budget)); });
    return result;
}
void damd_close(void* handle) { delete static_cast<Handle*>(handle); }

void* damd_alloc(void* handle, uint64_t count, int vram) {
    Tensor* result = nullptr;
    safely([&] {
        require(handle != nullptr, "Null GPU handle");
        auto allocation = std::make_shared<Allocation>(*static_cast<Handle*>(handle), count,
                                vram ? AMDGPU_GEM_DOMAIN_VRAM : AMDGPU_GEM_DOMAIN_GTT);
        if (vram) {
            float values[guard]; std::fill_n(values, guard, sentinel);
            staged_transfer(*allocation, 0, values, guard, true);
            staged_transfer(*allocation, guard + count, values, guard, true);
            allocation->validate();
        }
        result = new Tensor{std::move(allocation), 0, static_cast<size_t>(count)};
    });
    return result;
}
void* damd_view(void* handle, uint64_t offset, uint64_t count) {
    Tensor* result = nullptr;
    safely([&] {
        auto& parent = tensor(handle);
        require(count > 0 && offset <= parent.count && count <= parent.count - offset,
                "Tensor view exceeds allocation");
        result = new Tensor{parent.allocation, parent.offset + offset, static_cast<size_t>(count)};
    });
    return result;
}
void damd_free(void* handle) { delete static_cast<Tensor*>(handle); }

int damd_upload(void* handle, const float* data, uint64_t count) {
    return safely([&] {
        auto& t = tensor(handle);
        require(data && count == t.count, "Upload length mismatch");
        if (t.allocation->vram) {
            for (size_t i = 0; i < count; i += 262144) {
                const size_t length = std::min(size_t{262144}, size_t{count} - i);
                staged_transfer(*t.allocation, guard + t.offset + i, const_cast<float*>(data + i), length, true);
            }
        } else std::memcpy(t.host(), data, count * sizeof(float));
        t.allocation->validate();
    });
}
int damd_upload_f16(void* handle, const uint16_t* data, uint64_t count) {
    return safely([&] {
        auto& t = tensor(handle);
        require(data && count == t.count && count <= 262144, "FP16 upload chunk size mismatch");
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i) {
            _Float16 value;
            std::memcpy(&value, data + i, sizeof(value));
            values[i] = static_cast<float>(value);
            require(std::isfinite(values[i]), "Nonfinite FP16 weight");
        }
        require(damd_upload(handle, values.data(), count) == 0, "Staged FP16 upload failed");
    });
}

int damd_read(void* handle, float* data, uint64_t count) {
    return safely([&] {
        auto& t = tensor(handle);
        require(data && count == t.count, "Readback length mismatch");
        t.allocation->validate();
        if (t.allocation->vram) {
            for (size_t i = 0; i < count; i += 262144)
                staged_transfer(*t.allocation, guard + t.offset + i, data + i, std::min(size_t{262144}, size_t{count} - i), false);
        } else std::memcpy(data, t.host(), count * sizeof(float));
    });
}

int damd_binary(int kind, void* a_handle, void* b_handle, void* c_handle,
                uint32_t rows, uint32_t columns, uint32_t inner) {
    return safely([&] {
        auto& a = tensor(a_handle); auto& b = tensor(b_handle); auto& c = tensor(c_handle);
        auto& state = *a.allocation->state;
        if (kind == 0 || kind == 1) {
            require(a.count == b.count && a.count == c.count, "Elementwise shape mismatch");
        } else if (kind == 2) {
            require(GemvKernelParams::valid_shape(rows, columns) &&
                    a.count == uint64_t{rows} * columns && b.count == columns && c.count == rows,
                    "GEMV shape mismatch");
        } else if (kind == 3) {
            require(GemmKernelParams::valid_shape(rows, columns, inner) &&
                    a.count == uint64_t{rows} * inner && b.count == uint64_t{inner} * columns &&
                    c.count == uint64_t{rows} * columns, "GEMM shape mismatch");
        } else {
            throw std::runtime_error("Invalid binary operation");
        }
        require(c.allocation.get() != a.allocation.get() && c.allocation.get() != b.allocation.get(),
                "GPU binary output must use a separate allocation");
        dispatch(state, kind, {&a, &b, &c}, [&](PM4Builder& pm4, uint64_t program) {
            if (kind == 0) pm4.add_polaris_vector_add(program, {a.address(), b.address(), c.address(),
                                                               static_cast<uint32_t>(c.count)});
            if (kind == 1) pm4.add_polaris_vector_mul(program, {a.address(), b.address(), c.address(),
                                                               static_cast<uint32_t>(c.count)});
            if (kind == 2) pm4.add_polaris_gemv(program, {a.address(), b.address(), c.address(), rows, columns});
            if (kind == 3) pm4.add_polaris_gemm(program, {a.address(), b.address(), c.address(), rows, columns, inner});
        });
    });
}

int damd_unary(int kind, void* x_handle, void* y_handle, void* aux_handle, float epsilon) {
    return safely([&] {
        auto& x = tensor(x_handle); auto& y = tensor(y_handle);
        require(x.count == y.count && x.allocation.get() != y.allocation.get(), "Unary shape or alias mismatch");
        auto& state = *x.allocation->state;
        std::vector<Tensor*> tensors{&x, &y};
        Tensor* aux = nullptr;
        if (kind == 4) {
            aux = &tensor(aux_handle);
            require(aux->count == x.count && std::isfinite(epsilon) && epsilon > 0, "Invalid RMSNorm parameters");
            tensors.push_back(aux);
        }
        require(kind == 4 || kind == 6 || kind == 7, "Invalid unary operation");
        dispatch(state, kind, tensors, [&](PM4Builder& pm4, uint64_t program) {
            if (kind == 4) pm4.add_polaris_rmsnorm(program, {x.address(), aux->address(), y.address(),
                                                           static_cast<uint32_t>(x.count), epsilon});
            if (kind == 6) pm4.add_polaris_silu(program, {x.address(), y.address(), static_cast<uint32_t>(x.count)});
            if (kind == 7) pm4.add_polaris_softmax(program, {x.address(), y.address(), static_cast<uint32_t>(x.count)});
        });
    });
}

int damd_rope(void* x_handle, void* phase_handle, void* y_handle) {
    return safely([&] {
        auto& x = tensor(x_handle); auto& phase = tensor(phase_handle); auto& y = tensor(y_handle);
        require(x.count == y.count && phase.count == x.count / 2 &&
                RopeKernelParams::valid_shape(x.count, x.count) &&
                x.allocation.get() != y.allocation.get(), "Invalid full-head RoPE tensors");
        auto& state = *x.allocation->state;
        dispatch(state, 5, {&x, &phase, &y}, [&](PM4Builder& pm4, uint64_t program) {
            pm4.add_polaris_rope(program, {x.address(), phase.address(), y.address(),
                    static_cast<uint32_t>(x.count), static_cast<uint32_t>(x.count), RopeLayout::HalfSplit});
        });
    });
}

int damd_stats(void* handle, uint64_t* values) {
    return safely([&] {
        require(handle && values, "Invalid stats pointer");
        auto& state = **static_cast<Handle*>(handle);
        values[0] = state.resident; values[1] = state.peak;
        values[2] = state.submissions; values[3] = state.waits; values[4] = state.budget;
        values[5] = state.vram_bytes; values[6] = state.gtt_bytes;
    });
}

} // extern "C"
