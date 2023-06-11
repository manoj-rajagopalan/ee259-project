# Instructions to build and run

## System Requirements
- Ubuntu 22.04+ system with CUDA 11+ installed.
- While developing, I used an AWS g4dn system with Tesla T4 GPU (RTX capable).

## Build
- Install `gcc-10` and `g++10`
- Change symlinks `/usr/bin/g`{`cc`,`++`} to point to the above
- `mkdir _build && cd _build`
- In addition to running `cmake ..`, also run `ccmake ..` and make the following changes:
  - Set `CMAKE_CUDA_FLAGS` = `-std=c++14 -ccbin g++-10` (note that `/usr/bin/g++-10` must symlink properly)
  - Set `CMAKE_CXX_COMPILER`=`/usr/bin/g++-10`
  - Set `CMAKE_CXX_COMPILER_RANLIB`=`/usr/bin/gcc-ranlib-10`
  - Set `CMAKE_CXX_COMPILER_AR`=`/usr/bin/gcc-ar-10`
  - Set `CMAKE_CXX_FLAGS` = `-std=c++14`
  - Remember to run the generation step inside `ccmake`
- Despite the above changes, CMake issues a warning during generation but this is the best that was possible.
- Run `make` - this will create an executable named `ee259`.
   Do not run this yet.
- Separately build PTX for ray-tracing code for dynamic loading:
  `nvcc ee259.cu -g --ptx --ccbin g++-10 -I .. -I <path to OptiX include/ dir>`
  This creates `ee295.ptx` in the `_build/` dir.
  This will be dynamically loaded at program runtime.

# Run
- Run `ee259` executable from within the build dir.
  Ensure that `ee259.ptx` is in the same dir.

# Notes
- `nvcc` v11 has a problem when running with `gcc` v11: compile failures are reported inside `std::functional` (some difficulty in unpacking template parameters with `...` (in the C++11+ sense)). Notes on the internet say switching to `gcc` v10 helps but it is NOT sufficient to simply `apt install g++-10`: `nvcc` does NOT pick it up automatically. The `g++-10` value of the `-ccbin` argument actually indicates the `g++10` executable (in `PATH` due to `/usr/bin/`) that `nvcc` must use. This removes the problematic version of the `<functional>` header.
- Including `ee259.cu` into CMakeLists.txt will result in `nvcc` linker failure to resolve `optix...` symbols. Looks like this resolution is dynamically, by the OptiX runtime, at the time the module is built. Hence we build it separately and present it as an asset to the running program.


