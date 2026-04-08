#pragma once

#include <opencv2/core.hpp>

// Row-wise then column-wise 1D linear inpaint. Jacobi (4-neighbour) runs only on
// thick crossings: fully masked row ∩ fully masked column, plus sites OR'd in via
// relax_nan_sites_u8. Remaining masked non-finite pixels get a single 8-neighbour
// mean (no iterative smoothing there).
void linearInpaintMaskedRowsThenCols(cv::Mat &frame_f32c1,
                                     const cv::Mat &bad_mask_u8,
                                     const cv::Mat *relax_nan_sites_u8);
