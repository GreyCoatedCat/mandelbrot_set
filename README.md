# Simulasi Fraktal (Mandelbrot, Julia, Burning Ship, Tricorn) dengan OpenCL

Program ini menghitung fraktal ala-Mandelbrot secara paralel di GPU (atau
CPU) menggunakan OpenCL, dan bisa mereproduksi gaya render populer "satellite
antenna" (Seahorse Valley) dengan palet warna oranye-biru-putih klasik.

## Isi
- `mandelbrot_kernel.cl` — kernel OpenCL (4 jenis fraktal + palet warna klasik)
- `mandelbrot_opencl.cpp` — program host (parsing argumen, device, render, simpan gambar)

## 1. Install OpenCL & compile

```bash
sudo apt install ocl-icd-opencl-dev nvidia-opencl-dev   # sesuaikan vendor GPU Anda
g++ -std=c++17 mandelbrot_opencl.cpp -o mandelbrot -lOpenCL
```
macOS: ganti `-lOpenCL` dengan `-framework OpenCL`.

Cek device: `./mandelbrot info`

## 2. Mereproduksi gambar bergaya "satellite antenna" (Seahorse Valley)

Gambar semacam itu (satu blob hitam besar dengan "antena" tipis + spiral
biru/putih dan cincin oranye di luar) adalah bagian dari seri zoom
Mandelbrot yang sangat populer, hasil zoom ke area **Seahorse Valley** —
tepatnya di sekitar sebuah **titik Misiurewicz yang terverifikasi**:
```
Re(c) = -0.7436438870371587
Im(c) =  0.1318259042053119
```
(titik ini dipakai berulang kali di berbagai render Mandelbrot terkenal,
karena area di sekitarnya penuh struktur "satelit" mini-Mandelbrot dengan
"antena").

**Catatan penting:** koordinat & level zoom PERSIS dari file gambar aslinya
tidak saya temukan datanya secara publik, jadi ini adalah **pendekatan**,
bukan reproduksi pixel-per-pixel. Gunakan flag `--preset` + `--zoomlevel` di
bawah, coba beberapa nilai `zoomlevel`, dan bandingkan hasilnya dengan
gambar referensimu:

```bash
./mandelbrot --preset=satellite_antenna --zoomlevel=14 --iter=2000 --out=coba_14.ppm
./mandelbrot --preset=satellite_antenna --zoomlevel=16 --iter=3000 --out=coba_16.ppm
./mandelbrot --preset=satellite_antenna --zoomlevel=18 --iter=4000 --out=coba_18.ppm
./mandelbrot --preset=satellite_antenna --zoomlevel=20 --iter=5000 --out=coba_20.ppm
./mandelbrot --preset=satellite_antenna --zoomlevel=22 --iter=6000 --out=coba_22.ppm
```

`--zoomlevel=N` mengatur lebar area = `3.0769 / 2^N` (jadi makin besar N,
makin dalam zoom-nya). Semakin dalam, semakin butuh `--iter` besar biar
detail tetap tajam.

Kalau sudah ketemu level yang paling mendekati, kamu bisa geser posisinya
sedikit dengan menimpa area manual (`--xmin/--xmax/--ymin/--ymax`) di sekitar
hasil `zoomlevel` yang paling cocok — program mencetak area (`xmin, xmax,
ymin, ymax`) yang dipakai di setiap run, jadi kamu punya titik awal yang
jelas untuk menyesuaikan (pan) secara manual.

### Kalau ingin coba palet warnanya saja (tanpa preset)

Palet oranye-biru-putih ini sekarang jadi default untuk SEMUA render (bukan
cuma preset satellite_antenna) — jadi cukup jalankan Mandelbrot standar dan
warnanya sudah bergaya serupa:
```bash
./mandelbrot --iter=2000
```
Atur `--palette=N` untuk mengubah lebar 1 siklus warna (default 20; makin
kecil = pita warna lebih rapat/banyak, makin besar = lebih lebar/halus):
```bash
./mandelbrot --preset=satellite_antenna --zoomlevel=16 --palette=8 --iter=3000
```

## 3. Semua flag yang tersedia

| Flag        | Arti                                              | Default          |
|-------------|-----------------------------------------------------|------------------|
| `--width` `--height` | Resolusi gambar                            | 1600 x 1200      |
| `--iter`    | Iterasi maksimum per piksel                          | 1000             |
| `--localx` `--localy` | Local work size OpenCL                    | 0 (otomatis)     |
| `--xmin` `--xmax` `--ymin` `--ymax` | Area bidang kompleks manual  | -2.5..1.0, -1.25..1.25 |
| `--type`    | `mandelbrot`, `julia`, `burningship`, `tricorn`      | mandelbrot       |
| `--power`   | Eksponen `z^power+c` (boleh pecahan)                 | 2.0              |
| `--cre` `--cim` | Konstanta Julia                                 | -0.7, 0.27015    |
| `--palette` | Panjang 1 siklus warna oranye-biru                   | 20               |
| `--preset`  | `satellite_antenna` (lihat bagian 2)                 | (kosong)         |
| `--zoomlevel` | Kedalaman zoom untuk preset (lebar = 3.0769/2^N)   | 18               |
| `--cpu_only`  | Paksa pakai device CPU (tidak fallback ke GPU)     | nonaktif         |
| `--gpu_use`   | Paksa pakai device GPU (error kalau GPU tidak ada) | nonaktif         |
| `--out`     | Nama file output                                     | mandelbrot.ppm   |

Catatan `--cpu_only` / `--gpu_use`:
- Tidak dipakai keduanya sekaligus (program akan menolak).
- Kalau tidak ada flag ini sama sekali, perilaku default: coba pakai GPU dulu, kalau tidak ada baru fallback otomatis ke CPU (seperti sebelumnya).
- `--cpu_only` berguna untuk membandingkan performa CPU vs GPU, atau kalau driver GPU sedang bermasalah.
- `--gpu_use` berguna kalau Anda ingin memastikan render benar-benar jalan di GPU dan mau tahu (lewat error) kalau ternyata OpenCL cuma mendeteksi CPU.

## Soal local work size (`--localx`/`--localy`) yang tidak ditentukan

Kalau `--localx`/`--localy` **tidak diisi** (atau diisi 0 — ini default),
program memanggil `clEnqueueNDRangeKernel` dengan parameter *local_work_size*
= `NULL`. Ini artinya:

- **Driver/runtime OpenCL yang memilihkan ukuran work-group-nya sendiri**,
  bukan Anda. OpenCL spec mengizinkan ini secara eksplisit — device akan
  otomatis membagi `global_work_size` (jumlah total piksel) ke dalam
  work-group berukuran yang menurut driver paling optimal untuk kernel dan
  hardware tersebut.
- Ukuran yang dipilih biasanya berdasarkan heuristik vendor: mempertimbangkan
  jumlah register/local memory yang dipakai kernel, jumlah compute
  unit/SIMD width device, dan bentuk `global_work_size` itu sendiri. Nilainya
  bisa berbeda-beda antara NVIDIA, AMD, Intel, bahkan CPU vs GPU pada vendor
  yang sama.
- **Anda tidak tahu persis angka yang dipakai** kecuali menelusurinya lewat
  profiler (mis. `nvidia-nsight`, `CodeXL`, atau `CL_KERNEL_WORK_GROUP_SIZE`
  lewat `clGetKernelWorkGroupInfo`) — program ini sendiri tidak mencetak
  angka tersebut karena OpenCL tidak mewajibkan driver melaporkannya balik.
- **Tidak ada kewajiban `global_work_size` habis dibagi apa pun** ketika
  local size NULL — beda dengan kalau Anda menentukan sendiri (di mana
  program ini membulatkan global size ke kelipatan local size).
- Untuk kernel sederhana seperti Mandelbrot (setiap piksel independen, tidak
  ada local memory/synchronization antar work-item), driver otomatis
  BIASANYA sudah cukup baik — menentukan sendiri lewat `--localx/--localy`
  paling berguna untuk micro-optimization / eksperimen, dibandingkan
  kernel yang memang butuh ukuran local tertentu (mis. yang memakai
  `__local` memory atau `barrier()`).

Singkatnya: local size `0`/tidak diisi = "biarkan OpenCL yang atur", bukan
berarti "tidak ada work-group sama sekali" — work-group tetap ada, cuma
ukurannya dipilihkan otomatis, bukan oleh Anda.

## 4. Melihat / mengonversi hasil

```bash
convert mandelbrot.ppm mandelbrot.png     # ImageMagick
python3 -c "from PIL import Image; Image.open('mandelbrot.ppm').save('mandelbrot.png')"
```

## Kenapa gambar tampak full hitam saat iterasi tinggi?

Warna dihitung dari *smooth iteration count*, TIDAK dinormalisasi dengan
`iter/maxIter` -- jadi tetap kaya warna berapa pun `--iter` yang dipakai.
Titik yang benar-benar masuk ke dalam set tetap hitam (itu benar secara
matematis).

## Kustomisasi lanjutan
- **Warna**: edit array `stops[]` di `classicPalette()` pada kernel untuk skema warna lain.
- **Jenis fraktal baru**: tambah nilai enum `TYPE_...` + cabang `if` pada kernel (pola sama seperti Burning Ship/Tricorn).
- **Presisi**: kernel pakai `double`. Kalau device tidak dukung `cl_khr_fp64` (cek `./mandelbrot info`), ganti semua `double` di kernel jadi `float` -- tapi ini akan membatasi seberapa dalam zoom bisa dilakukan sebelum muncul artefak pixelated.
