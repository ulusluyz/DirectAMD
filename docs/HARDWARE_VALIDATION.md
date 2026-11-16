# DirectAMD Physical GPU Validation

The direct DRM vector-add example targets Polaris 10 (GCN4 / `gfx803`,
PCI ID `1002:67df`). Use a Linux system with the `amdgpu` driver and
read/write access to the GPU's DRM render node.

## Build and run host tests

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DENABLE_WERROR=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Debug keeps the host-test assertions enabled.

## Select each physical card

```bash
lspci -nnk -d 1002:67df
ls -l /dev/dri/by-path/*-render
```

Pass one render node per invocation. For the two cards in the
[validation report](FINAL_VALIDATION_REPORT.md):

```bash
./build/examples/vector_add_example /dev/dri/by-path/pci-0000:0a:00.0-render
./build/examples/vector_add_example /dev/dri/by-path/pci-0000:0b:00.0-render
```

Use your system's PCI addresses. To capture direct DRM calls for a run,
replace the PCI path in this example and resolve it to the render node:

```bash
node=$(readlink -f /dev/dri/by-path/pci-0000:0a:00.0-render)
strace -yy -e trace=ioctl -P "$node" ./build/examples/vector_add_example "$node"
```

A successful run reports both GPU vector-add cases and exits 0. It checks
the 4-element output `[6, 8, 10, 12]`, every element of the 257-element
output, and the sentinel values in the padded output slots. The trace
should show successful command submission and fence waits on the selected
node. A submission, fence, or readback failure returns nonzero. If there
is no usable render node, record hardware execution as untested.
