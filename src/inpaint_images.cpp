// =============================================================================
// inpaint_tiff — EXAMPLE / DEMONSTRATION CODE ONLY
//
// This program is NOT production software. It is NOT supported. It exists only
// to show one possible approach (OpenCV cv::inpaint). Your team must validate
// results for your instrument and replace this with a proper solution if needed.
//
// Pipeline order: run this AFTER normalize_tiff (flat-field correction). Input
// TIFFs should be normalized float images, not raw detector frames.
//
// Mask rule (matches supplied detector masks):
//   Value 0     = good pixel (leave as measured).
//   Value > 0   = bad pixel or gap (inpaint here).
//
// NaN / Inf: gaps or bad math upstream often contain non-finite floats. OpenCV’s
// float inpaint can leave NaNs if neighbours are bad; we OR those pixels into the
// inpaint mask and replace non-finite values before/after inpainting.
//
// Borderless vs full-frame: if width and height each differ by exactly 2 (one pixel
// off per side), the smaller of {image, mask} is treated as borderless. We crop the
// outer 1-pixel rim from the larger so sizes match.
//
// Optional: --inpaint-noise-scale adds Gaussian noise only on inpainted pixels,
// with σ = scale × sqrt(max(|pixel|, tiny)) after inpainting (shot-noise style).
//
// Build: same CMake as normalize_images — target inpaint_tiff (OpenCV core, imgcodecs, imgproc, photo).
// =============================================================================

#include "linear_inpaint.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

static void print_usage(const char *exe) {
    std::cerr
        << "Usage:\n"
        << "  " << exe
        << " --mask <mask.tif> (--in <one.tif> --out <one_out.tif>)\n"
        << "               OR (--in <folder> --out <folder>)\n"
        << "Optional:\n"
        << "  --radius <pixels>   inpaint radius (default: 3). Try 2–5 for small "
           "defects;\n"
        << "                      larger gaps may need larger radius (slower).\n"
        << "  --algorithm telea|ns|linear   (default: telea; linear ignores "
           "--radius)\n"
        << "  --nan-fill <float>  replace NaN/Inf before inpaint and fix stragglers "
           "after (default: 1.0)\n"
        << "  --inpaint-noise-scale <float>  if > 0, add Gaussian noise on inpainted "
           "pixels only;\n"
        << "                      std dev = scale × sqrt(max(|pixel|, 1e-12)) "
           "(default: 0 = off)\n";
}

// Replace NaN and Inf so OpenCV inpaint does not spread garbage; typical
// transmission maps use 1.0 in “empty” regions — tune with --nan-fill.
static void replace_nonfinite(cv::Mat &m32f, float fill) {
    CV_Assert(m32f.type() == CV_32FC1);
    for (int r = 0; r < m32f.rows; ++r) {
        float *row = m32f.ptr<float>(r);
        for (int c = 0; c < m32f.cols; ++c) {
            if (!std::isfinite(row[c])) {
                row[c] = fill;
            }
        }
    }
}

// 255 where the image is not finite — OR into the gap mask so those pixels are
// always inpainted, even if the file mask missed them.
static void nonfinite_mask_u8(const cv::Mat &m32f, cv::Mat *out_u8) {
    *out_u8 = cv::Mat::zeros(m32f.size(), CV_8UC1);
    for (int r = 0; r < m32f.rows; ++r) {
        const float *row = m32f.ptr<float>(r);
        uchar *mrow = out_u8->ptr<uchar>(r);
        for (int c = 0; c < m32f.cols; ++c) {
            if (!std::isfinite(row[c])) {
                mrow[c] = 255;
            }
        }
    }
}

// Inpainted pixels: mask_use != 0. Noise ~ N(0, σ²) with σ = scale * sqrt(|v|).
static void add_sqrt_scaled_noise_on_mask(cv::Mat &filled32,
                                          const cv::Mat &mask_u8,
                                          float scale,
                                          std::mt19937 &rng) {
    if (scale <= 0.f) {
        return;
    }
    CV_Assert(filled32.type() == CV_32FC1 && mask_u8.type() == CV_8UC1);
    CV_Assert(filled32.size() == mask_u8.size());
    std::normal_distribution<float> normal(0.f, 1.f);
    constexpr float eps = 1e-12f;
    for (int r = 0; r < filled32.rows; ++r) {
        float *row = filled32.ptr<float>(r);
        const uchar *mrow = mask_u8.ptr<uchar>(r);
        for (int c = 0; c < filled32.cols; ++c) {
            if (mrow[c] == 0) {
                continue;
            }
            float v = row[c];
            float sigma = scale * std::sqrt(std::max(std::fabs(v), eps));
            row[c] += sigma * normal(rng);
        }
    }
}

static bool is_tiff_extension(const fs::path &p) {
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".tif" || ext == ".tiff";
}

static std::vector<fs::path> list_tiffs_sorted(const fs::path &folder) {
    std::vector<fs::path> out;
    if (!fs::exists(folder) || !fs::is_directory(folder)) {
        return out;
    }
    for (const auto &entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (is_tiff_extension(entry.path())) {
            out.push_back(entry.path());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

// Stem with "_inpaint" before a trailing digit run (e.g. img_00001 -> img_inpaint_00001).
// No trailing digits: append "_inpaint" (e.g. frame -> frame_inpaint).
static std::string inpaint_output_stem(const fs::path &in_file) {
    std::string stem = in_file.stem().string();
    std::size_t j = stem.size();
    while (j > 0 && std::isdigit(static_cast<unsigned char>(stem[j - 1]))) {
        --j;
    }
    if (j == stem.size()) {
        return stem + "_inpaint";
    }
    std::string number = stem.substr(j);
    std::string prefix = stem.substr(0, j);
    while (!prefix.empty() && prefix.back() == '_') {
        prefix.pop_back();
    }
    if (prefix.empty()) {
        return std::string("_inpaint_") + number;
    }
    return prefix + "_inpaint_" + number;
}

// Load mask: single channel — any pixel != 0 means "inpaint here".
// OpenCV's compare() returns an 8-bit mask (0 or 255) for inpaint().
static bool load_mask_u8(const fs::path &path, cv::Mat *mask_u8) {
    cv::Mat m = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (m.empty()) {
        std::cerr << "Error: could not read mask: " << path << "\n";
        return false;
    }
    if (m.channels() != 1) {
        std::cerr << "Error: mask must be single-channel.\n";
        return false;
    }
    if (m.type() != CV_8UC1 && m.type() != CV_16UC1 && m.type() != CV_32FC1) {
        std::cerr << "Error: unsupported mask type (use 8/16/32-bit single channel).\n";
        return false;
    }
    cv::compare(m, 0, *mask_u8, cv::CMP_NE);
    return true;
}

// Convert any supported single-channel image to float32 for inpaint.
static bool to_float32(const cv::Mat &src, cv::Mat *dst) {
    if (src.channels() != 1) {
        std::cerr << "Error: expected single-channel TIFF.\n";
        return false;
    }
    if (src.type() == CV_32FC1) {
        *dst = src;
        return true;
    }
    src.convertTo(*dst, CV_32F);
    return true;
}

static bool same_size(const cv::Mat &a, const cv::Mat &b) {
    return a.rows == b.rows && a.cols == b.cols;
}

// If |Δw|==2 and |Δh|==2, crop one pixel from each side of the larger mat.
// mask_u8 may be updated in place when the mask is the larger asset (shared across batch).
static bool align_borderless_image_mask(cv::Mat *img32, cv::Mat *mask_u8) {
    const int iw = img32->cols;
    const int ih = img32->rows;
    const int mw = mask_u8->cols;
    const int mh = mask_u8->rows;
    if (iw == mw && ih == mh) {
        return true;
    }
    static bool warned_crop_img = false;
    static bool warned_crop_mask = false;
    if (iw == mw + 2 && ih == mh + 2) {
        if (!warned_crop_img) {
            std::cerr
                << "Note: images are 1 px larger per side than the mask; cropping "
                   "the outer rim from each image (borderless mask / full-frame "
                   "images).\n";
            warned_crop_img = true;
        }
        *img32 = (*img32)(cv::Rect(1, 1, mw, mh)).clone();
        return true;
    }
    if (mw == iw + 2 && mh == ih + 2) {
        if (!warned_crop_mask) {
            std::cerr
                << "Note: mask is 1 px larger per side than the images; cropping "
                   "the outer rim from the mask (borderless images / full-frame "
                   "mask).\n";
            warned_crop_mask = true;
        }
        *mask_u8 = (*mask_u8)(cv::Rect(1, 1, iw, ih)).clone();
        return true;
    }
    return false;
}

// Sentinel: use separable linear + Jacobi inpaint (not an OpenCV enum value).
static constexpr int kInpaintAlgoLinear = -1;

static int process_one(const fs::path &in_path,
                       const fs::path &out_path,
                       const cv::Mat &mask_u8,
                       double radius,
                       int inpaint_algo,
                       float nan_fill,
                       float inpaint_noise_scale,
                       uint64_t rng_seed) {
    cv::Mat mask_work = mask_u8.clone();
    std::mt19937 rng(rng_seed);

    cv::Mat img = cv::imread(in_path.string(), cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "Error: could not read: " << in_path << "\n";
        return 1;
    }
    cv::Mat img32;
    if (!to_float32(img, &img32)) {
        return 1;
    }
    if (!align_borderless_image_mask(&img32, &mask_work)) {
        std::cerr << "Error: size mismatch image vs mask for: " << in_path << "\n";
        std::cerr << "  image: " << img32.cols << " x " << img32.rows << "   mask: "
                  << mask_work.cols << " x " << mask_work.rows << "\n";
        std::cerr << "  (Borderless alignment allows exactly 2 pixels difference in "
                     "width and height; otherwise sizes must match.)\n";
        return 1;
    }

    cv::Mat nan_mask;
    nonfinite_mask_u8(img32, &nan_mask);
    cv::Mat mask_use;
    cv::bitwise_or(mask_work, nan_mask, mask_use);

    replace_nonfinite(img32, nan_fill);

    cv::Mat filled;
    if (inpaint_algo == kInpaintAlgoLinear) {
        img32.copyTo(filled);
        linearInpaintMaskedRowsThenCols(filled, mask_use, nullptr);
    } else {
        cv::inpaint(img32, mask_use, filled, radius, inpaint_algo);
    }
    replace_nonfinite(filled, nan_fill);
    add_sqrt_scaled_noise_on_mask(filled, mask_use, inpaint_noise_scale, rng);
    replace_nonfinite(filled, nan_fill);

    if (!cv::imwrite(out_path.string(), filled)) {
        std::cerr << "Error: failed to write: " << out_path << "\n";
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    fs::path mask_path;
    fs::path in_path;
    fs::path out_path;
    double radius = 3.0;
    std::string algo_name = "telea";
    float nan_fill = 1.0f;
    float inpaint_noise_scale = 0.f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mask" && i + 1 < argc) {
            mask_path = argv[++i];
        } else if (arg == "--in" && i + 1 < argc) {
            in_path = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (arg == "--radius" && i + 1 < argc) {
            radius = std::strtod(argv[++i], nullptr);
        } else if (arg == "--algorithm" && i + 1 < argc) {
            algo_name = argv[++i];
        } else if (arg == "--nan-fill" && i + 1 < argc) {
            nan_fill = static_cast<float>(std::strtod(argv[++i], nullptr));
        } else if (arg == "--inpaint-noise-scale" && i + 1 < argc) {
            inpaint_noise_scale =
                static_cast<float>(std::strtod(argv[++i], nullptr));
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (mask_path.empty() || in_path.empty() || out_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }
    if (inpaint_noise_scale < 0.f) {
        std::cerr << "Error: --inpaint-noise-scale must be >= 0\n";
        return 1;
    }

    std::random_device rd;
    const uint64_t base_seed =
        (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());

    int inpaint_algo = cv::INPAINT_TELEA;
    std::transform(algo_name.begin(), algo_name.end(), algo_name.begin(), ::tolower);
    if (algo_name == "telea" || algo_name == "fast") {
        inpaint_algo = cv::INPAINT_TELEA;
    } else if (algo_name == "ns" || algo_name == "navier" || algo_name == "slow") {
        inpaint_algo = cv::INPAINT_NS;
    } else if (algo_name == "linear") {
        inpaint_algo = kInpaintAlgoLinear;
    } else {
        std::cerr << "Error: --algorithm must be telea, ns, or linear\n";
        return 1;
    }

    cv::Mat mask_u8;
    if (!load_mask_u8(mask_path, &mask_u8)) {
        return 1;
    }

    const bool in_is_dir = fs::is_directory(in_path);
    const bool out_is_dir = fs::is_directory(out_path) ||
                            (!fs::exists(out_path) && in_is_dir);

    if (in_is_dir) {
        std::error_code ec;
        fs::create_directories(out_path, ec);
        if (ec) {
            std::cerr << "Error: could not create output folder: " << out_path
                      << "\n";
            return 1;
        }
        const std::vector<fs::path> files = list_tiffs_sorted(in_path);
        if (files.empty()) {
            std::cerr << "Error: no TIFF files in: " << in_path << "\n";
            return 1;
        }
        const int nfiles = static_cast<int>(files.size());
        std::atomic<int> failed{0};
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) if (nfiles > 1)
#endif
        for (int ii = 0; ii < nfiles; ++ii) {
            const fs::path &f = files[static_cast<size_t>(ii)];
            const fs::path dest = out_path / (inpaint_output_stem(f) + ".tiff");
            const uint64_t seed =
                base_seed ^
                (static_cast<uint64_t>(ii + 1) * 0xD6E8FEB866D9D43DULL);
            if (process_one(f, dest, mask_u8, radius, inpaint_algo, nan_fill,
                            inpaint_noise_scale, seed) != 0) {
                failed.store(1, std::memory_order_relaxed);
            }
        }
        if (failed.load(std::memory_order_relaxed) != 0) {
            return 1;
        }
        std::cout << "Done -> " << out_path.string() << "\n";
        return 0;
    }

    // Single file mode: create parent folder of output if needed.
    {
        std::error_code ec;
        fs::create_directories(out_path.parent_path(), ec);
    }
    if (process_one(in_path, out_path, mask_u8, radius, inpaint_algo, nan_fill,
                    inpaint_noise_scale, base_seed) != 0) {
        return 1;
    }
    std::cout << "Wrote: " << out_path.string() << "\n";
    return 0;
}
