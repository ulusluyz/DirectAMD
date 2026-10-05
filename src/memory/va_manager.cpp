#include "directamd/memory.hpp"

namespace DirectAMD {

VAManager::VAManager(uint64_t start_address) : current_va_(start_address) {}

uint64_t VAManager::allocate_range(size_t bytes, size_t alignment) {
    // Round size up to alignment boundary (default 4KB)
    size_t aligned_size = (bytes + alignment - 1) & ~(alignment - 1);
    uint64_t allocated_va = current_va_.fetch_add(aligned_size, std::memory_order_relaxed);
    return allocated_va;
}

void VAManager::free_range(uint64_t /*gpu_va*/, size_t /*bytes*/) {
    // Bump allocator cleanup strategy for fast execution
}

} // namespace DirectAMD
