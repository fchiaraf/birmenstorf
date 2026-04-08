#include "linear_inpaint.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

static constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// Linear interpolation along a 1D strip: gapPos / goodPos are sorted indices;
// good positions keep vals[]; gaps filled between consecutive good samples.
// Leading / trailing gaps get nearest good value (constant extrapolation).
static void inpaintStrip(float *vals, const std::vector<int> &gapPos,
                         const std::vector<int> &goodPos) {
    if (gapPos.empty() || goodPos.empty()) {
        return;
    }

    if (goodPos.size() == 1) {
        const float v = vals[goodPos[0]];
        for (int g : gapPos) {
            vals[g] = v;
        }
        return;
    }
    const int firstGood = goodPos.front();
    const int lastGood = goodPos.back();
    int gi = 0;
    const int ngaps = static_cast<int>(gapPos.size());

    while (gi < ngaps && gapPos[gi] < firstGood) {
        vals[gapPos[gi++]] = vals[firstGood];
    }

    for (size_t k = 0; k + 1 < goodPos.size() && gi < ngaps; ++k) {
        const int x0 = goodPos[k];
        const int x1 = goodPos[k + 1];
        const float v0 = vals[x0];
        const float v1 = vals[x1];
        const float inv_span = 1.f / static_cast<float>(x1 - x0);
        while (gi < ngaps && gapPos[gi] < x1) {
            const int g = gapPos[gi++];
            vals[g] = v0 + (v1 - v0) * static_cast<float>(g - x0) * inv_span;
        }
    }

    while (gi < ngaps) {
        vals[gapPos[gi++]] = vals[lastGood];
    }
}

// Thick cross: fully masked row ∩ fully masked column (vertical ∧ horizontal gap).
static void thickCrossingRelaxMask_u8(const cv::Mat &mask_u8, cv::Mat *out_u8) {
    CV_Assert(mask_u8.type() == CV_8UC1 && out_u8 != nullptr);
    const int h = mask_u8.rows;
    const int w = mask_u8.cols;
    std::vector<uchar> row_all_masked(static_cast<size_t>(h), 1);
    std::vector<uchar> col_all_masked(static_cast<size_t>(w), 1);
    for (int y = 0; y < h; ++y) {
        const uchar *mr = mask_u8.ptr<uchar>(y);
        for (int x = 0; x < w; ++x) {
            if (mr[x] == 0) {
                row_all_masked[static_cast<size_t>(y)] = 0;
                break;
            }
        }
    }
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            if (mask_u8.ptr<uchar>(y)[x] == 0) {
                col_all_masked[static_cast<size_t>(x)] = 0;
                break;
            }
        }
    }
    *out_u8 = cv::Mat::zeros(h, w, CV_8UC1);
    for (int y = 0; y < h; ++y) {
        if (!row_all_masked[static_cast<size_t>(y)]) {
            continue;
        }
        uchar *orow = out_u8->ptr<uchar>(y);
        for (int x = 0; x < w; ++x) {
            if (col_all_masked[static_cast<size_t>(x)]) {
                orow[x] = 255;
            }
        }
    }
}

// Jacobi-style 4-neighbour averaging on sparse sites only.
static void relaxSitesJacobiSparse(cv::Mat &frame, const cv::Mat &relax_u8,
                                   int max_passes) {
    CV_Assert(frame.type() == CV_32FC1 && relax_u8.type() == CV_8UC1);
    CV_Assert(frame.size() == relax_u8.size());
    if (max_passes <= 0) {
        return;
    }
    const int h = frame.rows;
    const int w = frame.cols;
    std::vector<std::pair<int, int>> coords;
    coords.reserve((static_cast<size_t>(h) * static_cast<size_t>(w)) / 16u + 64u);
    for (int y = 0; y < h; ++y) {
        const uchar *ru = relax_u8.ptr<uchar>(y);
        for (int x = 0; x < w; ++x) {
            if (ru[x] != 0) {
                coords.emplace_back(y, x);
            }
        }
    }
    if (coords.empty()) {
        return;
    }

    std::vector<float> nextv(coords.size());

    const float tol = 1e-6f;
    for (int pass = 0; pass < max_passes; ++pass) {
#pragma omp parallel for schedule(static)
        for (int64_t ii = 0; ii < static_cast<int64_t>(coords.size()); ++ii) {
            const auto &rc = coords[static_cast<size_t>(ii)];
            const int y = rc.first;
            const int x = rc.second;
            float sum = 0.f;
            int cnt = 0;
            if (y > 0) {
                const float v = frame.ptr<float>(y - 1)[x];
                // non-finite neighbours excluded from the average
                if (std::isfinite(v)) {
                    sum += v;
                    ++cnt;
                }
            }
            if (y + 1 < h) {
                const float v = frame.ptr<float>(y + 1)[x];
                if (std::isfinite(v)) {
                    sum += v;
                    ++cnt;
                }
            }
            if (x > 0) {
                const float v = frame.ptr<float>(y)[x - 1];
                if (std::isfinite(v)) {
                    sum += v;
                    ++cnt;
                }
            }
            if (x + 1 < w) {
                const float v = frame.ptr<float>(y)[x + 1];
                if (std::isfinite(v)) {
                    sum += v;
                    ++cnt;
                }
            }
            float *roww = frame.ptr<float>(y);
            nextv[static_cast<size_t>(ii)] =
                cnt > 0 ? sum / static_cast<float>(cnt) : roww[x];
        }

        bool changed = false;
        for (size_t ii = 0; ii < coords.size(); ++ii) {
            const auto &rc = coords[ii];
            float *roww = frame.ptr<float>(rc.first);
            const int x = rc.second;
            const float nv = nextv[ii];
            const float oldv = roww[x];
            roww[x] = nv;
            if (std::isfinite(oldv) && std::isfinite(nv)) {
                if (std::abs(nv - oldv) > tol) {
                    changed = true;
                }
            } else if (std::isfinite(nv) != std::isfinite(oldv)) {
                changed = true;
            }
        }
        if (!changed) {
            break;
        }
    }
}

} // namespace

void linearInpaintMaskedRowsThenCols(cv::Mat &frame_f32c1,
                                     const cv::Mat &bad_mask_u8,
                                     const cv::Mat *relax_nan_sites_u8) {
    CV_Assert(frame_f32c1.type() == CV_32FC1 && bad_mask_u8.type() == CV_8UC1);
    CV_Assert(frame_f32c1.size() == bad_mask_u8.size());

    const int h = frame_f32c1.rows;
    const int w = frame_f32c1.cols;
    if (cv::countNonZero(bad_mask_u8) == 0) {
        return;
    }

#pragma omp parallel
    {
        std::vector<int> gapPos;
        std::vector<int> goodPos;
        gapPos.reserve(static_cast<size_t>(w));
        goodPos.reserve(static_cast<size_t>(w));

#pragma omp for schedule(dynamic, 4)
        for (int y = 0; y < h; ++y) {
            gapPos.clear();
            goodPos.clear();
            float *row = frame_f32c1.ptr<float>(y);
            const uchar *rowMask = bad_mask_u8.ptr<uchar>(y);
            for (int x = 0; x < w; ++x) {
                // treat NaN as gap even if not in file mask
                if (rowMask[x] || !std::isfinite(row[x])) {
                    gapPos.push_back(x);
                } else {
                    goodPos.push_back(x);
                }
            }
            if (gapPos.empty()) {
                continue;
            }
            if (goodPos.empty()) {
                for (int x = 0; x < w; ++x) {
                    float sum = 0.f;
                    int cnt = 0;
                    if (y > 0) {
                        const float v = frame_f32c1.ptr<float>(y - 1)[x];
                        if (std::isfinite(v)) {
                            sum += v;
                            ++cnt;
                        }
                    }
                    if (y + 1 < h) {
                        const float v = frame_f32c1.ptr<float>(y + 1)[x];
                        if (std::isfinite(v)) {
                            sum += v;
                            ++cnt;
                        }
                    }
                    row[x] = cnt > 0 ? sum / static_cast<float>(cnt) : kNaN;
                }
            } else {
                inpaintStrip(row, gapPos, goodPos);
            }
        }
    }

#pragma omp parallel
    {
        std::vector<int> gapPos;
        std::vector<int> goodPos;
        std::vector<float> col;
        gapPos.reserve(static_cast<size_t>(h));
        goodPos.reserve(static_cast<size_t>(h));
        col.resize(static_cast<size_t>(h));

#pragma omp for schedule(dynamic, 4)
        for (int x = 0; x < w; ++x) {
            gapPos.clear();
            goodPos.clear();
            for (int y = 0; y < h; ++y) {
                col[y] = frame_f32c1.ptr<float>(y)[x];
                if (bad_mask_u8.ptr<uchar>(y)[x] || !std::isfinite(col[y])) {
                    gapPos.push_back(y);
                } else {
                    goodPos.push_back(y);
                }
            }
            if (gapPos.empty()) {
                for (int y = 0; y < h; ++y) {
                    frame_f32c1.ptr<float>(y)[x] = col[y];
                }
                continue;
            }
            if (goodPos.empty()) {
                for (int y = 0; y < h; ++y) {
                    float sum = 0.f;
                    int cnt = 0;
                    if (x > 0) {
                        const float v = frame_f32c1.ptr<float>(y)[x - 1];
                        if (std::isfinite(v)) {
                            sum += v;
                            ++cnt;
                        }
                    }
                    if (x + 1 < w) {
                        const float v = frame_f32c1.ptr<float>(y)[x + 1];
                        if (std::isfinite(v)) {
                            sum += v;
                            ++cnt;
                        }
                    }
                    col[y] = cnt > 0 ? sum / static_cast<float>(cnt) : kNaN;
                }
            } else {
                inpaintStrip(col.data(), gapPos, goodPos);
            }
            for (int y = 0; y < h; ++y) {
                frame_f32c1.ptr<float>(y)[x] = col[y];
            }
        }
    }

    // Jacobi only on thick-mask crossings (full gap row ∩ full gap column), not on
    // the whole inpaint mask. Optional relax_nan_sites_u8 is OR'd into that set.
    cv::Mat jacobi_sites;
    thickCrossingRelaxMask_u8(bad_mask_u8, &jacobi_sites);
    if (relax_nan_sites_u8 != nullptr && !relax_nan_sites_u8->empty()) {
        CV_Assert(relax_nan_sites_u8->type() == CV_8UC1 &&
                  relax_nan_sites_u8->size() == bad_mask_u8.size());
        cv::bitwise_or(jacobi_sites, *relax_nan_sites_u8, jacobi_sites);
    }
    if (cv::countNonZero(jacobi_sites) > 0) {
        const int relax_pass_cap =
            std::min(512, std::max(128, (std::max(w, h) * 3 + 2) / 2));
        relaxSitesJacobiSparse(frame_f32c1, jacobi_sites, relax_pass_cap);
    }

    // One-shot 8-neighbour mean for any masked pixel still non-finite (not Jacobi).

    for (int y = 0; y < h; ++y) {
        const uchar *m = bad_mask_u8.ptr<uchar>(y);
        float *rowf = frame_f32c1.ptr<float>(y);
        for (int x = 0; x < w; ++x) {
            if (m[x] != 0 && !std::isfinite(rowf[x])) {
                float sum = 0.f;
                int cnt = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    const int yy = y + dy;
                    if (yy < 0 || yy >= h) {
                        continue;
                    }
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) {
                            continue;
                        }
                        const int xx = x + dx;
                        if (xx < 0 || xx >= w) {
                            continue;
                        }
                        const float v = frame_f32c1.ptr<float>(yy)[xx];
                        if (std::isfinite(v)) {
                            sum += v;
                            ++cnt;
                        }
                    }
                }
                if (cnt > 0) {
                    rowf[x] = sum / static_cast<float>(cnt);
                }
            }
        }
    }
}

