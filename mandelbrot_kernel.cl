// mandelbrot_kernel.cl
//
// Kernel OpenCL untuk fraktal ala-Mandelbrot (Mandelbrot, Julia, Burning Ship,
// Tricorn) dengan eksponen bebas, DAN palet warna "klasik" oranye-biru-putih
// yang mendekati gaya render populer (mis. seri render Wikipedia "satellite
// antenna" / Seahorse Valley: cincin oranye di luar, spiral biru/putih di
// dalam, hitam solid untuk titik yang masuk ke dalam set).

#define TYPE_MANDELBROT   0
#define TYPE_JULIA        1
#define TYPE_BURNING_SHIP 2
#define TYPE_TRICORN      3

inline double2 complexPow(double ar, double ai, double power) {
    double r = sqrt(ar * ar + ai * ai);
    if (r < 1e-300) {
        return (double2)(0.0, 0.0);
    }
    double theta   = atan2(ai, ar);
    double newR     = pow(r, power);
    double newTheta = theta * power;
    return (double2)(newR * cos(newTheta), newR * sin(newTheta));
}

// ------------------------------------------------------------------
// Palet warna "klasik" oranye-biru-putih (mirip gradient default Ultra
// Fractal / banyak render Mandelbrot populer): interior gelap-navy ->
// biru -> cyan/putih -> kuning -> oranye -> merah-coklat gelap -> ulang.
// t dinormalisasi 0..1 lalu diinterpolasi linear antar "stop" warna.
// ------------------------------------------------------------------
inline double3 classicPalette(double t) {
    // 7 stop warna (R,G,B masing-masing 0..1), akan diulang secara siklik
    const double3 stops[7] = {
        (double3)(0.00, 0.01, 0.15),  // navy sangat gelap
        (double3)(0.05, 0.15, 0.55),  // biru
        (double3)(0.30, 0.65, 0.95),  // cyan terang
        (double3)(0.95, 0.98, 1.00),  // putih
        (double3)(1.00, 0.85, 0.20),  // kuning
        (double3)(1.00, 0.45, 0.05),  // oranye
        (double3)(0.35, 0.08, 0.02)   // coklat-merah gelap
    };
    const int n = 7;

    t = t - floor(t); // bungkus ke [0,1)
    double pos = t * (double)n;
    int i0 = (int)floor(pos);
    int i1 = (i0 + 1) % n;
    double frac = pos - (double)i0;

    double3 c0 = stops[i0];
    double3 c1 = stops[i1];
    return (double3)(
        c0.x + (c1.x - c0.x) * frac,
        c0.y + (c1.y - c0.y) * frac,
        c0.z + (c1.z - c0.z) * frac
    );
}

__kernel void mandelbrot(
    __global uchar4* output,
    const int width,
    const int height,
    const double minReal,
    const double maxReal,
    const double minImag,
    const double maxImag,
    const int maxIter,
    const int fractalType,
    const double power,
    const double juliaCRe,
    const double juliaCIm,
    const double paletteScale   // panjang periode 1 siklus warna (dalam satuan iterasi smooth)
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    double real = minReal + (maxReal - minReal) * ((double)x / (double)(width - 1));
    double imag = minImag + (maxImag - minImag) * ((double)y / (double)(height - 1));

    double zr, zi, cr, ci;
    if (fractalType == TYPE_JULIA) {
        zr = real; zi = imag;
        cr = juliaCRe; ci = juliaCIm;
    } else {
        zr = 0.0; zi = 0.0;
        cr = real; ci = imag;
    }

    const double bailout = 256.0;
    int iter = 0;

    while (zr * zr + zi * zi <= bailout && iter < maxIter) {
        double ar = zr;
        double ai = zi;

        if (fractalType == TYPE_BURNING_SHIP) {
            ar = fabs(zr);
            ai = fabs(zi);
        } else if (fractalType == TYPE_TRICORN) {
            ai = -zi;
        }

        double2 zp = complexPow(ar, ai, power);
        zr = zp.x + cr;
        zi = zp.y + ci;
        iter++;
    }

    uchar4 color;

    if (iter >= maxIter) {
        color = (uchar4)(0, 0, 0, 255);
    } else {
        // Smooth iteration count -- TIDAK dinormalisasi ke maxIter (supaya
        // tidak "hitam semua" saat maxIter dinaikkan tinggi).
        double modZ = sqrt(zr * zr + zi * zi);
        double nu   = log(log(modZ) / log(2.0)) / log(2.0);
        double smoothIter = (double)iter + 1.0 - nu;

        double t = smoothIter / paletteScale; // posisi dalam siklus warna
        double3 rgb = classicPalette(t);

        color = (uchar4)((uchar)(rgb.x * 255.0), (uchar)(rgb.y * 255.0), (uchar)(rgb.z * 255.0), 255);
    }

    int idx = y * width + x;
    output[idx] = color;
}
