// mandelbrot_opencl.cpp
//
// Simulasi berbagai fraktal ala-Mandelbrot menggunakan OpenCL, dengan preset
// khusus untuk mereproduksi gaya "satellite antenna" (zoom ke Seahorse
// Valley, palet warna oranye-biru-putih klasik).
//
// Cara compile (Linux, dengan OpenCL ICD terpasang):
//   g++ -std=c++17 mandelbrot_opencl.cpp -o mandelbrot -lOpenCL
// macOS:
//   g++ -std=c++17 mandelbrot_opencl.cpp -o mandelbrot -framework OpenCL
//
// ============================== CARA PAKAI ==============================
//
//   ./mandelbrot info                 -> info device OpenCL, lalu keluar
//
//   ./mandelbrot [--key=value ...]     -> render (semua flag opsional)
//
//   Flag umum:
//     --width=N --height=N --iter=N --localx=N --localy=N
//     --xmin=F --xmax=F --ymin=F --ymax=F
//     --type=mandelbrot|julia|burningship|tricorn
//     --power=F        (default 2.0, boleh pecahan)
//     --cre=F --cim=F  (konstanta Julia)
//     --palette=F      panjang 1 siklus warna oranye-biru (default 20)
//     --out=NAMA.ppm
//
//   Flag KHUSUS untuk meniru gaya "satellite antenna" (Seahorse Valley):
//     --preset=satellite_antenna
//         Mengatur center zoom ke titik terverifikasi (Misiurewicz point)
//         Re=-0.7436438870371587, Im=0.1318259042053119 (area Seahorse
//         Valley yang terkenal berisi struktur "antena" satelit).
//     --zoomlevel=N   (default 18)
//         Seberapa dalam zoom di sekitar titik itu: lebar area =
//         3.0769 / 2^N. Coba beberapa nilai N (mis. 12..24) untuk menemukan
//         level zoom yang paling mirip dengan gambar referensimu -- karena
//         kombinasi persis lebar/posisi frame aslinya tidak saya temukan
//         datanya, ini pendekatan yang paling aman & bisa diverifikasi.
//
//   Contoh:
//     ./mandelbrot --preset=satellite_antenna --zoomlevel=16 --iter=3000
//     ./mandelbrot --preset=satellite_antenna --zoomlevel=20 --iter=5000 --out=coba2.ppm
//
// Output: file .ppm (bisa dibuka GIMP/IrfanView, atau dikonversi ke PNG --
// lihat README.md)
// ==========================================================================

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <cmath>

enum FractalType {
    TYPE_MANDELBROT   = 0,
    TYPE_JULIA        = 1,
    TYPE_BURNING_SHIP = 2,
    TYPE_TRICORN      = 3
};

static std::string readFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Gagal membuka file kernel: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

static void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        std::cerr << "Error OpenCL saat '" << operation << "': kode " << err << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static void savePPM(const std::string& filename, const std::vector<unsigned char>& rgba,
                     int width, int height) {
    std::ofstream out(filename, std::ios::binary);
    out << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; i++) {
        unsigned char rgb[3] = { rgba[i * 4 + 0], rgba[i * 4 + 1], rgba[i * 4 + 2] };
        out.write(reinterpret_cast<char*>(rgb), 3);
    }
    out.close();
}

static std::unordered_map<std::string, std::string> parseFlags(int argc, char** argv) {
    std::unordered_map<std::string, std::string> flags;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.rfind("--", 0) != 0) continue;
        arg = arg.substr(2);
        size_t eq = arg.find('=');
        if (eq == std::string::npos) {
            flags[arg] = "1";
        } else {
            flags[arg.substr(0, eq)] = arg.substr(eq + 1);
        }
    }
    return flags;
}

static std::string getFlag(const std::unordered_map<std::string, std::string>& f,
                            const std::string& key, const std::string& def) {
    auto it = f.find(key);
    return (it != f.end()) ? it->second : def;
}

static FractalType parseFractalType(const std::string& name) {
    std::string n = name;
    std::transform(n.begin(), n.end(), n.begin(), ::tolower);
    if (n == "julia") return TYPE_JULIA;
    if (n == "burningship" || n == "burning_ship" || n == "ship") return TYPE_BURNING_SHIP;
    if (n == "tricorn" || n == "mandelbar") return TYPE_TRICORN;
    return TYPE_MANDELBROT;
}

static std::string fractalTypeName(FractalType t) {
    switch (t) {
        case TYPE_JULIA: return "Julia";
        case TYPE_BURNING_SHIP: return "Burning Ship";
        case TYPE_TRICORN: return "Tricorn";
        default: return "Mandelbrot";
    }
}

static void printAllDeviceInfo() {
    cl_uint numPlatforms = 0;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (numPlatforms == 0) {
        std::cout << "Tidak ada platform OpenCL yang terdeteksi di sistem ini." << std::endl;
        return;
    }
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    for (cl_uint p = 0; p < numPlatforms; p++) {
        char platName[256] = {0};
        clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, sizeof(platName), platName, nullptr);
        std::cout << "Platform #" << p << " : " << platName << std::endl;

        cl_uint numDevices = 0;
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
        if (numDevices == 0) { std::cout << "  (tidak ada device)" << std::endl; continue; }
        std::vector<cl_device_id> devices(numDevices);
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr);

        for (cl_uint d = 0; d < numDevices; d++) {
            char devName[256] = {0};
            size_t maxWorkGroupSize = 0;
            size_t maxWorkItemSizes[3] = {0,0,0};
            clGetDeviceInfo(devices[d], CL_DEVICE_NAME, sizeof(devName), devName, nullptr);
            clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);
            clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxWorkItemSizes), maxWorkItemSizes, nullptr);
            std::cout << "  Device #" << d << ": " << devName
                      << " | max work-group size: " << maxWorkGroupSize
                      << " | max work-item: (" << maxWorkItemSizes[0] << "," << maxWorkItemSizes[1] << "," << maxWorkItemSizes[2] << ")"
                      << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    if (argc > 1 && (std::string(argv[1]) == "info" || std::string(argv[1]) == "--info")) {
        printAllDeviceInfo();
        return EXIT_SUCCESS;
    }

    auto flags = parseFlags(argc, argv);

    int width   = std::stoi(getFlag(flags, "width", "1600"));
    int height  = std::stoi(getFlag(flags, "height", "1200"));
    int maxIter = std::stoi(getFlag(flags, "iter", "1000"));
    int localX  = std::stoi(getFlag(flags, "localx", "0"));
    int localY  = std::stoi(getFlag(flags, "localy", "0"));

    double minReal = -2.5, maxReal = 1.0, minImag = -1.25, maxImag = 1.25;
    bool userSetArea = flags.count("xmin") || flags.count("xmax") ||
                        flags.count("ymin") || flags.count("ymax");

    std::string preset = getFlag(flags, "preset", "");

    if (preset == "satellite_antenna") {
        // Titik Seahorse Valley yang terverifikasi (Misiurewicz point) --
        // dipakai sebagai pusat zoom untuk mendekati gaya render
        // "satellite antenna". Lebar area ditentukan oleh --zoomlevel.
        double cx = -0.7436438870371587;
        double cy =  0.1318259042053119;
        int zoomLevel = std::stoi(getFlag(flags, "zoomlevel", "18"));
        double baseWidth = 3.0769; // lebar seluruh Mandelbrot set
        double w = baseWidth / std::pow(2.0, (double)zoomLevel);
        double h = w * ((double)height / (double)width);
        minReal = cx - w / 2.0; maxReal = cx + w / 2.0;
        minImag = cy - h / 2.0; maxImag = cy + h / 2.0;
        std::cout << "[preset satellite_antenna] zoomlevel=" << zoomLevel
                   << " -> lebar area = " << w << std::endl;
        std::cout << "Catatan: ini pendekatan berbasis titik terverifikasi "
                     "di Seahorse Valley, BUKAN koordinat pasti dari file "
                     "aslinya (tidak ditemukan datanya). Coba beberapa nilai "
                     "--zoomlevel (mis. 12..24) untuk mencari yang paling mirip."
                  << std::endl;
    }

    if (userSetArea) {
        minReal = std::stod(getFlag(flags, "xmin", std::to_string(minReal)));
        maxReal = std::stod(getFlag(flags, "xmax", std::to_string(maxReal)));
        minImag = std::stod(getFlag(flags, "ymin", std::to_string(minImag)));
        maxImag = std::stod(getFlag(flags, "ymax", std::to_string(maxImag)));
    }

    FractalType fractalType = parseFractalType(getFlag(flags, "type", "mandelbrot"));
    double power = std::stod(getFlag(flags, "power", "2.0"));
    double juliaCRe = std::stod(getFlag(flags, "cre", "-0.7"));
    double juliaCIm = std::stod(getFlag(flags, "cim", "0.27015"));
    double paletteScale = std::stod(getFlag(flags, "palette", "20.0"));
    std::string outFile = getFlag(flags, "out", "mandelbrot.ppm");

    if (fractalType == TYPE_JULIA && !userSetArea && preset != "satellite_antenna") {
        minReal = -1.5; maxReal = 1.5; minImag = -1.5; maxImag = 1.5;
    }

    std::cout << "Jenis fraktal : " << fractalTypeName(fractalType) << std::endl;
    std::cout << "Power (z^n+c) : " << power << std::endl;
    std::cout << "Resolusi      : " << width << " x " << height << std::endl;
    std::cout << "Iterasi maks  : " << maxIter << std::endl;
    std::cout << "Palette scale : " << paletteScale << std::endl;
    std::cout << "Area (real)   : [" << minReal << ", " << maxReal << "]" << std::endl;
    std::cout << "Area (imag)   : [" << minImag << ", " << maxImag << "]" << std::endl;

    cl_int err;
    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    checkError(err, "clGetPlatformIDs (count)");
    if (numPlatforms == 0) {
        std::cerr << "Tidak ada platform OpenCL yang ditemukan." << std::endl;
        return EXIT_FAILURE;
    }
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    cl_device_id device = nullptr;
    for (auto& plat : platforms) {
        cl_uint numDevices = 0;
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices) == CL_SUCCESS && numDevices > 0) {
            std::vector<cl_device_id> devices(numDevices);
            clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
            device = devices[0];
            break;
        }
    }
    if (device == nullptr) {
        for (auto& plat : platforms) {
            cl_uint numDevices = 0;
            if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_CPU, 0, nullptr, &numDevices) == CL_SUCCESS && numDevices > 0) {
                std::vector<cl_device_id> devices(numDevices);
                clGetDeviceIDs(plat, CL_DEVICE_TYPE_CPU, numDevices, devices.data(), nullptr);
                device = devices[0];
                break;
            }
        }
    }
    if (device == nullptr) {
        std::cerr << "Tidak ada device OpenCL (GPU/CPU) yang cocok ditemukan." << std::endl;
        return EXIT_FAILURE;
    }

    char deviceName[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    std::cout << "Device OpenCL : " << deviceName << std::endl;

    size_t maxWorkGroupSize = 0;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);
    size_t maxWorkItemSizes[3] = {0,0,0};
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxWorkItemSizes), maxWorkItemSizes, nullptr);

    if (localX > 0 && localY > 0) {
        if (static_cast<size_t>(localX) * static_cast<size_t>(localY) > maxWorkGroupSize ||
            static_cast<size_t>(localX) > maxWorkItemSizes[0] ||
            static_cast<size_t>(localY) > maxWorkItemSizes[1]) {
            std::cerr << "Peringatan: local work size melebihi batas device. Beralih otomatis." << std::endl;
            localX = 0; localY = 0;
        }
    }

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    checkError(err, "clCreateContext");
#ifdef CL_VERSION_2_0
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
#else
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
#endif
    checkError(err, "clCreateCommandQueue");

    std::string kernelSource = readFile("mandelbrot_kernel.cl");
    const char* sourcePtr = kernelSource.c_str();
    size_t sourceLen = kernelSource.size();
    cl_program program = clCreateProgramWithSource(context, 1, &sourcePtr, &sourceLen, &err);
    checkError(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Gagal compile kernel:\n" << log.data() << std::endl;
        return EXIT_FAILURE;
    }

    cl_kernel kernel = clCreateKernel(program, "mandelbrot", &err);
    checkError(err, "clCreateKernel");

    size_t numPixels = static_cast<size_t>(width) * height;
    size_t bufferSize = numPixels * 4 * sizeof(unsigned char);
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, &err);
    checkError(err, "clCreateBuffer");

    int fractalTypeInt = static_cast<int>(fractalType);
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &outputBuffer);
    err |= clSetKernelArg(kernel, 1, sizeof(int), &width);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &height);
    err |= clSetKernelArg(kernel, 3, sizeof(double), &minReal);
    err |= clSetKernelArg(kernel, 4, sizeof(double), &maxReal);
    err |= clSetKernelArg(kernel, 5, sizeof(double), &minImag);
    err |= clSetKernelArg(kernel, 6, sizeof(double), &maxImag);
    err |= clSetKernelArg(kernel, 7, sizeof(int), &maxIter);
    err |= clSetKernelArg(kernel, 8, sizeof(int), &fractalTypeInt);
    err |= clSetKernelArg(kernel, 9, sizeof(double), &power);
    err |= clSetKernelArg(kernel, 10, sizeof(double), &juliaCRe);
    err |= clSetKernelArg(kernel, 11, sizeof(double), &juliaCIm);
    err |= clSetKernelArg(kernel, 12, sizeof(double), &paletteScale);
    checkError(err, "clSetKernelArg");

    size_t globalSize[2];
    size_t localSize[2];
    size_t* localSizePtr = nullptr;
    if (localX > 0 && localY > 0) {
        localSize[0] = static_cast<size_t>(localX);
        localSize[1] = static_cast<size_t>(localY);
        localSizePtr = localSize;
        globalSize[0] = ((static_cast<size_t>(width)  + localSize[0] - 1) / localSize[0]) * localSize[0];
        globalSize[1] = ((static_cast<size_t>(height) + localSize[1] - 1) / localSize[1]) * localSize[1];
    } else {
        globalSize[0] = static_cast<size_t>(width);
        globalSize[1] = static_cast<size_t>(height);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, localSizePtr, 0, nullptr, nullptr);
    checkError(err, "clEnqueueNDRangeKernel");
    clFinish(queue);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Waktu komputasi kernel: "
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms" << std::endl;

    std::vector<unsigned char> pixels(numPixels * 4);
    err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, bufferSize, pixels.data(), 0, nullptr, nullptr);
    checkError(err, "clEnqueueReadBuffer");

    savePPM(outFile, pixels, width, height);
    std::cout << "Gambar tersimpan: " << outFile << std::endl;

    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return EXIT_SUCCESS;
}