#include "userd_observer.h"
#include <alloc/alloc_channel.h>
#include <class/clc56f.h>
#include <ctrl/ctrl2080/ctrl2080gpu.h>
#include <nv-unix-nvos-params-wrappers.h>
#include <nv_escape.h>
#include <nvos.h>
#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstddef>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#if !defined(__linux__) || !defined(__x86_64__)
#error "The passive forwarding ABI was reviewed only for Linux x86_64"
#endif

// The only device calls here forward calls already made by libcuda. No private
// memory dereference: only pinned, successful, known RM argument structures.
static_assert(sizeof(nv_ioctl_nvos33_parameters_with_fd) == 56);
static_assert(sizeof(nv_ioctl_nvos02_parameters_with_fd) == 56);
static_assert(offsetof(Nvc56fControl, GPGet) == 0x88);
static_assert(offsetof(Nvc56fControl, GPPut) == 0x8c);
namespace {
using Ioctl = int(*)(int, unsigned long, ...);
Ioctl next_ioctl = nullptr;
std::atomic<bool> recording{false};
thread_local bool inside = false;
struct Guard { Guard() { inside = true; } ~Guard() { inside = false; } };
struct State { std::recursive_mutex mutex; int log = -1; pid_t owner = 0; uint64_t seq = 0; bool failed = false; };
State& state() { static auto* s = new State; return *s; }
uint64_t ns() { timespec ts{}; clock_gettime(CLOCK_MONOTONIC_RAW, &ts); return uint64_t(ts.tv_sec) * 1000000000 + ts.tv_nsec; }
uint64_t address(const void* p) { return reinterpret_cast<uintptr_t>(p); }
struct Fd {
    bool nvidia = false;
    uint64_t dev = 0, ino = 0, rdev = 0;
    std::string path;
};
Fd fd_info(int fd) {
    Fd result; struct stat st{}; char path[64], target[256];
    if (fd < 0 || fstat(fd, &st)) return result;
    snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
    const auto n = readlink(path, target, sizeof(target) - 1);
    if (n < 0) return result;
    target[n] = 0; result.path = target;
    result.nvidia = S_ISCHR(st.st_mode) && result.path.rfind("/dev/nvidia", 0) == 0;
    result.dev = st.st_dev; result.ino = st.st_ino; result.rdev = st.st_rdev;
    return result;
}
std::string fd_fields(const Fd& f, const char* prefix = "") {
    std::ostringstream s;
    s << ",\"" << prefix << "fd_dev\":" << f.dev << ",\"" << prefix << "fd_ino\":" << f.ino
      << ",\"" << prefix << "fd_rdev\":" << f.rdev;
    // Only the already-checked NVIDIA device basename is logged here.
    if (f.nvidia) s << ",\"" << prefix << "fd_path\":\"" << f.path << '"';
    return s.str();
}
uint64_t emit(const char* kind, uint64_t begin, uint64_t end, const std::string& fields) {
    auto& s = state();
    if (s.owner != getpid() || s.log < 0 || s.seq >= 20000) { s.failed = true; return 0; }
    std::ostringstream row;
    row << "{\"seq\":" << ++s.seq << ",\"pid\":" << s.owner << ",\"tid\":" << syscall(SYS_gettid)
        << ",\"begin_ns\":" << begin << ",\"end_ns\":" << end << ",\"kind\":\"" << kind << '"' << fields << "}\n";
    const std::string bytes = row.str();
    if (write(s.log, bytes.data(), bytes.size()) != static_cast<ssize_t>(bytes.size())) s.failed = true;
    return s.failed ? 0 : s.seq;
}
void decode(int fd, unsigned long request, void* arg, int rc, int error, uint64_t begin, uint64_t end, const Fd& file) {
    std::ostringstream s;
    const unsigned nr = _IOC_NR(request), size = _IOC_SIZE(request);
    s << ",\"fd\":" << fd << fd_fields(file) << ",\"number\":" << nr << ",\"size\":" << size
      << ",\"rc\":" << rc << ",\"errno\":" << (rc < 0 ? error : 0);
    if (rc || !arg || (file.path != "/dev/nvidiactl" && nr != NV_ESC_RM_ALLOC_MEMORY)) { emit("ioctl", begin, end, s.str()); return; }
    if (nr == NV_ESC_RM_ALLOC) {
        uint32_t client = 0, object = 0, parent = 0, cls = 0, status = 0, bytes = 0, flags = 0; void* payload = nullptr;
        if (size == sizeof(NVOS21_PARAMETERS)) {
            const auto& a = *static_cast<NVOS21_PARAMETERS*>(arg);
            client = a.hRoot; object = a.hObjectNew; parent = a.hObjectParent; cls = a.hClass;
            status = a.status; bytes = a.paramsSize; payload = a.pAllocParms;
        } else if (size == sizeof(NVOS64_PARAMETERS)) {
            const auto& a = *static_cast<NVOS64_PARAMETERS*>(arg);
            client = a.hRoot; object = a.hObjectNew; parent = a.hObjectParent; cls = a.hClass;
            status = a.status; bytes = a.paramsSize; payload = a.pAllocParms; flags = a.flags;
        } else { state().failed = true; emit("unknown_alloc_abi", begin, end, s.str()); return; }
        s << ",\"client\":" << client << ",\"object\":" << object << ",\"parent\":" << parent
          << ",\"class\":" << cls << ",\"status\":" << status << ",\"params_size\":" << bytes << ",\"flags\":" << flags;
        if (cls == AMPERE_CHANNEL_GPFIFO_A && !status) {
            // Legacy paramsSize==0 means class-defined fixed layout in NVOS21.
            // FINN/unknown encodings are never reinterpreted as plain structs.
            if (flags || !payload || (bytes && bytes != sizeof(NV_CHANNEL_ALLOC_PARAMS))) {
                state().failed = true; emit("unknown_channel_abi", begin, end, s.str()); return;
            }
            const auto& p = *static_cast<NV_CHANNEL_ALLOC_PARAMS*>(payload);
            s << ",\"channel_flags\":" << p.flags << ",\"engine\":" << p.engineType
              << ",\"gpfifo_gpu_va\":" << p.gpFifoOffset << ",\"gpfifo_entries\":" << p.gpFifoEntries
              << ",\"vaspace\":" << p.hVASpace << ",\"userd_handles\":[";
            for (unsigned i = 0; i < NV_MAX_SUBDEVICES; ++i) { if (i) s << ','; s << p.hUserdMemory[i]; }
            s << "],\"userd_offsets\":[";
            for (unsigned i = 0; i < NV_MAX_SUBDEVICES; ++i) { if (i) s << ','; s << p.userdOffset[i]; }
            s << ']';
        }
        emit("alloc", begin, end, s.str());
    } else if (nr == NV_ESC_RM_MAP_MEMORY && size == sizeof(nv_ioctl_nvos33_parameters_with_fd)) {
        const auto& w = *static_cast<nv_ioctl_nvos33_parameters_with_fd*>(arg); const auto& p = w.params;
        s << ",\"client\":" << p.hClient << ",\"device\":" << p.hDevice << ",\"memory\":" << p.hMemory
          << ",\"offset\":" << p.offset << ",\"length\":" << p.length << ",\"linear\":" << address(p.pLinearAddress)
          << ",\"status\":" << p.status << ",\"flags\":" << p.flags << ",\"map_fd\":" << w.fd << fd_fields(fd_info(w.fd), "map_");
        emit("map_memory", begin, end, s.str());
    } else if (nr == NV_ESC_RM_ALLOC_MEMORY && size == sizeof(nv_ioctl_nvos02_parameters_with_fd)) {
        const auto& w = *static_cast<nv_ioctl_nvos02_parameters_with_fd*>(arg); const auto& p = w.params;
        s << ",\"client\":" << p.hRoot << ",\"object\":" << p.hObjectNew << ",\"parent\":" << p.hObjectParent
          << ",\"class\":" << p.hClass << ",\"flags\":" << p.flags << ",\"status\":" << p.status
          << ",\"linear\":" << address(p.pMemory) << ",\"limit\":" << p.limit << ",\"map_fd\":" << w.fd << fd_fields(fd_info(w.fd), "map_");
        emit("alloc_memory", begin, end, s.str());
    } else if (nr == NV_ESC_RM_UNMAP_MEMORY && size == sizeof(NVOS34_PARAMETERS)) {
        const auto& p = *static_cast<NVOS34_PARAMETERS*>(arg);
        s << ",\"client\":" << p.hClient << ",\"memory\":" << p.hMemory << ",\"linear\":" << address(p.pLinearAddress) << ",\"status\":" << p.status;
        emit("unmap_memory", begin, end, s.str());
    } else if (nr == NV_ESC_RM_FREE && size == sizeof(NVOS00_PARAMETERS)) {
        const auto& p = *static_cast<NVOS00_PARAMETERS*>(arg);
        s << ",\"client\":" << p.hRoot << ",\"object\":" << p.hObjectOld << ",\"status\":" << p.status;
        emit("free", begin, end, s.str());
    } else if (nr == NV_ESC_RM_CONTROL && size == sizeof(NVOS54_PARAMETERS)) {
        const auto& p = *static_cast<NVOS54_PARAMETERS*>(arg);
        s << ",\"client\":" << p.hClient << ",\"object\":" << p.hObject << ",\"command\":" << p.cmd
          << ",\"status\":" << p.status << ",\"flags\":" << p.flags << ",\"params_size\":" << p.paramsSize;
        if (!p.status && !p.flags && p.cmd == NV2080_CTRL_CMD_GPU_GET_GID_INFO && p.params && p.paramsSize == sizeof(NV2080_CTRL_GPU_GET_GID_INFO_PARAMS)) {
            const auto& gid = *static_cast<NV2080_CTRL_GPU_GET_GID_INFO_PARAMS*>(p.params);
            s << ",\"gid_flags\":" << gid.flags << ",\"gid_length\":" << gid.length << ",\"gid_bytes\":\"";
            if (gid.length <= sizeof(gid.data))
                for (unsigned i = 0; i < gid.length; ++i) s << std::hex << std::setfill('0') << std::setw(2) << unsigned(gid.data[i]);
            s << '"' << std::dec;
        }
        emit("control", begin, end, s.str());
    } else emit("ioctl", begin, end, s.str());
}
void* observe_mmap(void* addr, size_t length, int prot, int flags, int fd, off64_t offset) {
    // The Linux libc wrappers use this syscall too. Forward exactly the original
    // application request; never manufacture a mapping or alter protections.
    if (!recording || inside) return reinterpret_cast<void*>(syscall(SYS_mmap, addr, length, prot, flags, fd, offset));
    Guard guard; auto& s = state(); std::lock_guard<std::recursive_mutex> lock(s.mutex);
    const auto file = fd_info(fd); const auto begin = ns();
    void* result = reinterpret_cast<void*>(syscall(SYS_mmap, addr, length, prot, flags, fd, offset));
    const auto end = ns(); const int error = errno;
    try {
        // MAP_FIXED can invalidate a previous mapping even for an anonymous fd.
        if (file.nvidia || (flags & MAP_FIXED)) {
            std::ostringstream row;
            row << ",\"fd\":" << fd << fd_fields(file) << ",\"requested_address\":" << address(addr)
                << ",\"address\":" << address(result) << ",\"length\":" << length << ",\"prot\":" << prot
                << ",\"flags\":" << flags << ",\"offset\":" << offset << ",\"rc\":" << (result == MAP_FAILED ? -1 : 0);
            emit("mmap", begin, end, row.str());
        }
    } catch (...) { s.failed = true; }
    errno = error; return result;
}
}
extern "C" int userd_observer_begin(const char* path) {
    if (!path || recording || inside) return -1;
    Guard guard; auto& s = state();
    next_ioctl = reinterpret_cast<Ioctl>(dlsym(RTLD_NEXT, "ioctl"));
    Dl_info info{};
    if (!next_ioctl || !dladdr(reinterpret_cast<void*>(next_ioctl), &info) || !info.dli_fname ||
        !strstr(info.dli_fname, "librm_control.so")) return -2;
    s.log = open(path, O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
    if (s.log < 0) return -3;
    s.owner = getpid(); recording = true;
    const auto now = ns();
    return emit("start", now, now, ",\"abi\":1,\"bridge_hook_next\":true,\"channel_alloc_size\":" + std::to_string(sizeof(NV_CHANNEL_ALLOC_PARAMS))) ? 0 : -4;
}
extern "C" uint64_t userd_observer_mark(const char* label) {
    if (!recording || inside || !label || strspn(label, "abcdefghijklmnopqrstuvwxyz_") != strlen(label)) return 0;
    Guard guard; auto& s = state(); std::lock_guard<std::recursive_mutex> lock(s.mutex);
    const auto now = ns();
    return emit("mark", now, now, ",\"label\":\"" + std::string(label) + "\"");
}
extern "C" int userd_observer_ok() { auto& s = state(); std::lock_guard<std::recursive_mutex> lock(s.mutex); return recording && !s.failed && s.owner == getpid(); }
extern "C" int ioctl(int fd, unsigned long request, ...) noexcept {
    va_list args; va_start(args, request); void* arg = va_arg(args, void*); va_end(args);
    if (!next_ioctl) next_ioctl = reinterpret_cast<Ioctl>(dlsym(RTLD_NEXT, "ioctl"));
    if (!recording || inside || _IOC_TYPE(request) != 'F')
        return next_ioctl ? next_ioctl(fd, request, arg) : static_cast<int>(syscall(SYS_ioctl, fd, request, arg));
    Guard guard; auto& s = state(); std::lock_guard<std::recursive_mutex> lock(s.mutex);
    const auto file = fd_info(fd); const auto begin = ns();
    const int result = next_ioctl(fd, request, arg), error = errno; const auto end = ns();
    try { if (file.nvidia) decode(fd, request, arg, result, error, begin, end, file); }
    catch (...) { s.failed = true; }
    errno = error; return result;
}
extern "C" void* mmap(void* a, size_t n, int p, int f, int fd, off_t o) noexcept { return observe_mmap(a, n, p, f, fd, o); }
extern "C" void* mmap64(void* a, size_t n, int p, int f, int fd, off64_t o) noexcept { return observe_mmap(a, n, p, f, fd, o); }
extern "C" int munmap(void* addr, size_t length) noexcept {
    if (!recording || inside) return static_cast<int>(syscall(SYS_munmap, addr, length));
    Guard guard; auto& s = state(); std::lock_guard<std::recursive_mutex> lock(s.mutex); const auto begin = ns();
    const int result = static_cast<int>(syscall(SYS_munmap, addr, length)), error = errno;
    try { emit("munmap", begin, ns(), ",\"address\":" + std::to_string(address(addr)) + ",\"length\":" + std::to_string(length) + ",\"rc\":" + std::to_string(result)); }
    catch (...) { s.failed = true; }
    errno = error; return result;
}
extern "C" int close(int fd) {
    if (!recording || inside) return static_cast<int>(syscall(SYS_close, fd));
    Guard guard; auto& s = state(); std::lock_guard<std::recursive_mutex> lock(s.mutex);
    const auto file = fd_info(fd); const auto begin = ns();
    const int result = static_cast<int>(syscall(SYS_close, fd)), error = errno;
    try { if (file.nvidia) emit("close", begin, ns(), ",\"fd\":" + std::to_string(fd) + fd_fields(file) + ",\"rc\":" + std::to_string(result)); }
    catch (...) { s.failed = true; }
    errno = error; return result;
}
extern "C" void* mremap(void* old, size_t old_size, size_t new_size, int flags, ...) noexcept {
    void* target = nullptr;
    if (flags & MREMAP_FIXED) { va_list args; va_start(args, flags); target = va_arg(args, void*); va_end(args); }
    if (!recording || inside) return reinterpret_cast<void*>(syscall(SYS_mremap, old, old_size, new_size, flags, target));
    Guard guard; auto& s = state(); std::lock_guard<std::recursive_mutex> lock(s.mutex); const auto begin = ns();
    void* result = reinterpret_cast<void*>(syscall(SYS_mremap, old, old_size, new_size, flags, target));
    const int error = errno;
    try { emit("mremap", begin, ns(), ",\"address\":" + std::to_string(address(old)) + ",\"length\":" + std::to_string(old_size) +
               ",\"new_address\":" + std::to_string(address(result)) + ",\"new_length\":" + std::to_string(new_size) +
               ",\"rc\":" + std::to_string(result == MAP_FAILED ? -1 : 0)); }
    catch (...) { s.failed = true; }
    errno = error; return result;
}
