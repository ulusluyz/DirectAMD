# DirectAMD Physical Hardware Validation Procedure

## Overview
This document specifies the step-by-step physical hardware validation procedure for DirectAMD Milestone 1.

---

## 1. Prerequisites & Environment Setup

### Hardware Requirements
* Physical AMD GPU (RDNA2 GFX1030 / Navi 21/22/23 or RDNA3 GFX1100 / Navi 31/32/33).
* Host system connected via PCIe.

### Operating System & Driver Requirements
* Linux OS (Kernel >= 5.4 with active `amdgpu` driver).
* User added to `render` and `video` groups to access `/dev/dri/renderD128`:
  ```bash
  sudo usermod -aG render,video $USER
  ```

### Diagnostic Commands
Before running DirectAMD, verify that the kernel exposes a valid DRM render node:
```bash
ls -l /dev/dri/renderD*
# Expected output: /dev/dri/renderD128

dmesg | grep -i amdgpu
# Expected output: [drm] amdgpu kernel driver initialized
```

---

## 2. Compilation
Build DirectAMD with standard compiler flags (-Wall -Wextra -Wpedantic -Werror):

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_WERROR=ON ..
cmake --build . --parallel $(nproc)
```

---

## 3. Running Physical Vector-Add Validation

Execute the compiled vector-add example:
```bash
./examples/vector_add_example
```

### Expected Hardware Success Output
When running on physical hardware with `/dev/dri/renderD128`:
```
====================================================
          DirectAMD Vector-Add Prototype
====================================================
[+] Host Input Vector A: [1, 2, 3, 4]
[+] Host Input Vector B: [5, 6, 7, 8]
[+] AMD GPU Found at: /dev/dri/renderD128
[+] Architecture: RDNA2/RDNA3
MILESTONE 1: PASS
```

### Expected Output in Non-Hardware Environments
When running in environments lacking physical AMD GPUs:
```
====================================================
          DirectAMD Vector-Add Prototype
====================================================
[+] Host Input Vector A: [1, 2, 3, 4]
[+] Host Input Vector B: [5, 6, 7, 8]

----------------------------------------------------
PHYSICAL GPU DETECTED: NO (/dev/dri/renderD* missing)
STATUS: MILESTONE 1: NOT TESTED — PHYSICAL AMD GPU REQUIRED
----------------------------------------------------
[+] Verifying CPU Reference Math:
    C[0] = 6 (Expected: 6)
    C[1] = 8 (Expected: 8)
    C[2] = 10 (Expected: 10)
    C[3] = 12 (Expected: 12)
CPU REFERENCE PASS
```

---

## 4. Hardware Milestone PASS / FAIL Criteria

* **MILESTONE 1: PASS:** Requires `vector_add_example` to execute on a physical AMD GPU via DRM ioctls and produce $C = [6.0, 8.0, 10.0, 12.0]$ matching host reference values.
* **MILESTONE 1: FAIL:** Occurs if `DRM_IOCTL_AMDGPU_CS` returns submission errors, fence timeout occurs, or readback values mismatch.
* **MILESTONE 1: NOT TESTED — PHYSICAL AMD GPU REQUIRED:** Reported when `/dev/dri/renderD*` is missing.
