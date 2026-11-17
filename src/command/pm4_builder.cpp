#include "directamd/command.hpp"
#include "directamd/isa.hpp"

#include <bit>
#include <cmath>
#include <stdexcept>

namespace DirectAMD {

uint32_t PM4Builder::build_type3_header(uint8_t opcode, uint16_t count) {
    // PM4 Type 3 Header: [31:30]=11, [29:16]=count, [15:8]=opcode, [7:0]=predicate
    return (0b11U << 30) | ((static_cast<uint32_t>(count) & 0x3FFF) << 16) | ((static_cast<uint32_t>(opcode) & 0xFF) << 8);
}

void PM4Builder::add_set_sh_reg(uint32_t register_address, const std::vector<uint32_t>& values) {
    if (values.empty()) return;
    uint16_t count = static_cast<uint16_t>(values.size());
    packets_.push_back(build_type3_header(0x76, count)); // 0x76 = PACKET3_SET_SH_REG
    // Shader register addresses are dword indices relative to 0x2c00 in PM4.
    packets_.push_back(register_address - 0x2C00);
    for (uint32_t v : values) {
        packets_.push_back(v);
    }
}

void PM4Builder::add_acquire_mem(uint32_t coherence_control) {
    // GFX8 ACQUIRE_MEM has six payload dwords.
    packets_.push_back(build_type3_header(0x58, 5));
    packets_.push_back(coherence_control); // CP_COHER_CNTL
    packets_.push_back(0xFFFFFFFF); // Coherence bytes low
    packets_.push_back(0xFF);       // Coherence bytes high
    packets_.push_back(0);          // Base address low
    packets_.push_back(0);          // Base address high
    packets_.push_back(0x0000000A); // Poll interval
}

void PM4Builder::add_dispatch_direct(uint32_t gx, uint32_t gy, uint32_t gz, uint32_t dispatch_initiator) {
    packets_.push_back(build_type3_header(0x15, 3) | (1u << 1)); // 0x15 = PACKET3_DISPATCH_DIRECT
    packets_.push_back(gx);
    packets_.push_back(gy);
    packets_.push_back(gz);
    packets_.push_back(dispatch_initiator);
}

void PM4Builder::add_polaris_vector_add(uint64_t program_va, const VectorAddKernelParams& params) {
    // The kernel computes each byte offset in a single 32-bit VGPR.
    if (params.count == 0 || params.count > (1u << 30) || program_va % 256 != 0) {
        throw std::invalid_argument("Invalid Polaris vector-add dispatch");
    }

    add_polaris_dispatch(program_va, params.count, 2, {
        static_cast<uint32_t>(params.buffer_a_va), static_cast<uint32_t>(params.buffer_a_va >> 32),
        static_cast<uint32_t>(params.buffer_b_va), static_cast<uint32_t>(params.buffer_b_va >> 32),
        static_cast<uint32_t>(params.buffer_c_va), static_cast<uint32_t>(params.buffer_c_va >> 32),
        params.count
    });
}

void PM4Builder::add_polaris_gemv(uint64_t program_va, const GemvKernelParams& params) {
    if (!GemvKernelParams::valid_shape(params.rows, params.columns) || program_va % 256 != 0) {
        throw std::invalid_argument("Invalid Polaris GEMV dispatch");
    }

    add_polaris_dispatch(program_va, params.rows, 3, {
        static_cast<uint32_t>(params.matrix_va), static_cast<uint32_t>(params.matrix_va >> 32),
        static_cast<uint32_t>(params.vector_va), static_cast<uint32_t>(params.vector_va >> 32),
        static_cast<uint32_t>(params.output_va), static_cast<uint32_t>(params.output_va >> 32),
        params.rows, params.columns
    });
}

void PM4Builder::add_polaris_gemm(uint64_t program_va, const GemmKernelParams& params) {
    if (!GemmKernelParams::valid_shape(params.rows, params.columns, params.inner) ||
        program_va % 256 != 0) {
        throw std::invalid_argument("Invalid Polaris GEMM dispatch");
    }

    add_polaris_dispatch(program_va, params.columns, 3, {
        static_cast<uint32_t>(params.a_va), static_cast<uint32_t>(params.a_va >> 32),
        static_cast<uint32_t>(params.b_va), static_cast<uint32_t>(params.b_va >> 32),
        static_cast<uint32_t>(params.c_va), static_cast<uint32_t>(params.c_va >> 32),
        params.rows, params.columns, params.inner
    }, params.rows);
}

void PM4Builder::add_polaris_softmax(uint64_t program_va, const SoftmaxKernelParams& params) {
    if (!SoftmaxKernelParams::valid_count(params.count) || program_va % 256 != 0) {
        throw std::invalid_argument("Invalid Polaris Softmax dispatch");
    }

    // The wave loops over the full vector. Enable FP32 subnormal inputs and outputs.
    constexpr uint32_t preserve_denormals_mode = 0xf0;
    add_polaris_dispatch(program_va, ISA::PolarisEncoder::wave_size, 3, {
        static_cast<uint32_t>(params.input_va), static_cast<uint32_t>(params.input_va >> 32),
        static_cast<uint32_t>(params.output_va), static_cast<uint32_t>(params.output_va >> 32),
        params.count
    }, 0, preserve_denormals_mode);
}

void PM4Builder::add_polaris_rmsnorm(uint64_t program_va, const RmsNormKernelParams& params) {
    if (!RmsNormKernelParams::valid_count(params.count) || program_va % 256 != 0 ||
        !std::isfinite(params.epsilon) || params.epsilon <= 0.0f) {
        throw std::invalid_argument("Invalid Polaris RMSNorm dispatch");
    }

    constexpr uint32_t preserve_denormals_mode = 0xf0;
    add_polaris_dispatch(program_va, ISA::PolarisEncoder::wave_size, 3, {
        static_cast<uint32_t>(params.input_va), static_cast<uint32_t>(params.input_va >> 32),
        static_cast<uint32_t>(params.weights_va), static_cast<uint32_t>(params.weights_va >> 32),
        static_cast<uint32_t>(params.output_va), static_cast<uint32_t>(params.output_va >> 32),
        params.count, std::bit_cast<uint32_t>(params.epsilon)
    }, 0, preserve_denormals_mode);
}

void PM4Builder::add_polaris_rope(uint64_t program_va, const RopeKernelParams& params) {
    if (!RopeKernelParams::valid_shape(params.count, params.rotary_dim) ||
        !RopeKernelParams::valid_layout(params.layout) || program_va % 256 != 0) {
        throw std::invalid_argument("Invalid Polaris RoPE dispatch");
    }

    constexpr uint32_t preserve_denormals_mode = 0xf0;
    // One work item per rotated pair or per unrotated element.
    add_polaris_dispatch(program_va, params.count - params.rotary_dim / 2, 4, {
        static_cast<uint32_t>(params.input_va), static_cast<uint32_t>(params.input_va >> 32),
        static_cast<uint32_t>(params.phase_va), static_cast<uint32_t>(params.phase_va >> 32),
        static_cast<uint32_t>(params.output_va), static_cast<uint32_t>(params.output_va >> 32),
        params.count, params.rotary_dim, static_cast<uint32_t>(params.layout)
    }, 0, preserve_denormals_mode);
}

void PM4Builder::add_polaris_dispatch(uint64_t program_va, uint32_t items_x, uint32_t vgpr_blocks,
                                     const std::vector<uint32_t>& user_data, uint32_t groups_y,
                                     uint32_t float_mode) {
    constexpr uint32_t wave_size = ISA::PolarisEncoder::wave_size;
    constexpr uint32_t ib_alignment_dw = 8;
    constexpr uint32_t nop = 0xffff1000; // Single-dword PACKET3_NOP
    const uint32_t rsrc1 = (vgpr_blocks - 1) | (1u << 6) | (float_mode << 12); // 16 SGPRs, FP mode
    uint32_t rsrc2 = (static_cast<uint32_t>(user_data.size()) << 1) | (1u << 7); // Workgroup ID X
    if (groups_y != 0) rsrc2 |= 1u << 8; // Workgroup ID Y for the 2D grid

    add_set_sh_reg(0x2E04, {0, 0, 0}); // COMPUTE_START_X/Y/Z
    add_set_sh_reg(0x2E07, {wave_size, 1, 1}); // COMPUTE_NUM_THREAD_X/Y/Z
    add_set_sh_reg(0x2E0C, {static_cast<uint32_t>(program_va >> 8),
                           static_cast<uint32_t>(program_va >> 40)});
    add_set_sh_reg(0x2E12, {rsrc1, rsrc2}); // COMPUTE_PGM_RSRC1/2
    add_set_sh_reg(0x2E15, {0}); // COMPUTE_RESOURCE_LIMITS
    add_set_sh_reg(0x2E18, {0}); // COMPUTE_TMPRING_SIZE, no scratch
    add_set_sh_reg(0x2E40, user_data);
    const uint32_t groups_x = (static_cast<uint64_t>(items_x) + wave_size - 1) / wave_size;
    add_dispatch_direct(groups_x, groups_y == 0 ? 1 : groups_y, 1);
    const size_t padding = (ib_alignment_dw - packets_.size() % ib_alignment_dw) % ib_alignment_dw;
    packets_.insert(packets_.end(), padding, nop);
}

} // namespace DirectAMD
