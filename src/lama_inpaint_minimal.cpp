/**
 * LaMa inference (namespace lama) — used by inpaint_tiff (--algorithm lama).
 * Declarations in lama_inpaint.hpp. No CLI; see lama_inpaint.cpp.
 */

#include "lama_inpaint.hpp"

#include <torch/script.h>
#include <torch/torch.h>
#include <c10/core/InferenceMode.h>
#include <c10/util/Exception.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::string lowercase_string(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

namespace lama {

torch::Device selectDevice(const std::string &hint) {
    if (hint == "cpu") {
        std::cout << "[device] using CPU (--device cpu)\n" << std::flush;
        return torch::kCPU;
    }
    if (hint == "cuda") {
        std::cout << "[device] checking CUDA (--device cuda) ...\n" << std::flush;
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA requested but not available.");
        }
        std::cout << "[device] using CUDA\n" << std::flush;
        return torch::kCUDA;
    }
    std::cout << "[device] auto: probing CUDA ...\n" << std::flush;
    if (torch::cuda::is_available()) {
        std::cout << "[device] CUDA available - using GPU\n" << std::flush;
        return torch::kCUDA;
    }
    std::cout << "[device] CUDA not available - using CPU\n" << std::flush;
    return torch::kCPU;
}

bool loadTracedModel(const fs::path &model_path,
                     torch::Device device,
                     std::unique_ptr<torch::jit::script::Module> *out_model,
                     std::string *err_msg) {
    if (!out_model) {
        return false;
    }
    std::error_code ec;
    if (!fs::is_regular_file(model_path, ec)) {
        if (err_msg) {
            *err_msg = "Model path is not a readable file: " + model_path.string();
        }
        return false;
    }
    const std::string ext = lowercase_string(model_path.extension().string());
    if (ext == ".ckpt" || ext == ".pth") {
        if (err_msg) {
            *err_msg = "Model must be a TorchScript export (.pt), not a checkpoint (" + ext + ").";
        }
        return false;
    }
    try {
        std::ifstream ifs(model_path, std::ios::binary);
        if (!ifs) {
            if (err_msg) {
                *err_msg = "Could not open model file for reading: " + model_path.string();
            }
            return false;
        }
        torch::jit::script::Module loaded =
            torch::jit::load(ifs, std::optional<c10::Device>{device});
        loaded.eval();
        loaded = torch::jit::optimize_for_inference(loaded);
        *out_model = std::make_unique<torch::jit::script::Module>(std::move(loaded));
    } catch (const c10::Error &e) {
        if (err_msg) {
            *err_msg = std::string("Error loading model: ") + e.what();
        }
        return false;
    }
    return true;
}

namespace {

torch::Tensor hwc_bgr_float01_to_chw_rgb_tensor(const cv::Mat &bgr_hwc_f32) {
    cv::Mat rgb = bgr_hwc_f32.clone();
    cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
    if (!rgb.isContinuous()) {
        rgb = rgb.clone();
    }
    auto tensor = torch::from_blob(rgb.data, {1, rgb.rows, rgb.cols, 3}, torch::kFloat32).clone();
    return tensor.permute({0, 3, 1, 2});
}

torch::Tensor mask_u8_to_float01_tensor(const cv::Mat &mask_u8) {
    CV_Assert(mask_u8.type() == CV_8UC1);
    cv::Mat mask_f;
    mask_u8.convertTo(mask_f, CV_32FC1, 1.0f / 255.0f);
    cv::threshold(mask_f, mask_f, 0.5f, 1.0f, cv::THRESH_BINARY);
    if (!mask_f.isContinuous()) {
        mask_f = mask_f.clone();
    }
    auto tensor =
        torch::from_blob(mask_f.data, {1, mask_f.rows, mask_f.cols, 1}, torch::kFloat32).clone();
    return tensor.permute({0, 3, 1, 2});
}

} // namespace

bool inpaintGray32f(const cv::Mat &img_32fc1,
                    const cv::Mat &mask_u8c1,
                    const torch::jit::script::Module &model,
                    const torch::Device &device,
                    cv::Mat *out_32fc1,
                    std::string *err_msg) {
    if (!out_32fc1) {
        return false;
    }
    try {
        CV_Assert(img_32fc1.type() == CV_32FC1);
        CV_Assert(mask_u8c1.type() == CV_8UC1);
        CV_Assert(img_32fc1.size() == mask_u8c1.size());

        const int orig_h = img_32fc1.rows;
        const int orig_w = img_32fc1.cols;

        cv::Mat good_u8;
        cv::compare(mask_u8c1, 0, good_u8, cv::CMP_EQ);

        double minv = 0.0;
        double maxv = 1.0;
        cv::minMaxLoc(img_32fc1, &minv, &maxv, nullptr, nullptr, good_u8);
        if (!(maxv > minv)) {
            cv::minMaxLoc(img_32fc1, &minv, &maxv);
            if (!(maxv > minv)) {
                minv = 0.0;
                maxv = 1.0;
            }
        }
        const float denom = static_cast<float>(maxv - minv);
        const float inv_denom = denom > 0.f ? (1.f / denom) : 1.f;
        const float shift = static_cast<float>(-minv * inv_denom);

        cv::Mat n01;
        img_32fc1.convertTo(n01, CV_32F, static_cast<double>(inv_denom),
                            static_cast<double>(shift));
        cv::max(n01, 0.f, n01);
        cv::min(n01, 1.f, n01);

        cv::Mat bgr3;
        cv::cvtColor(n01, bgr3, cv::COLOR_GRAY2BGR);

        torch::Tensor image = hwc_bgr_float01_to_chw_rgb_tensor(bgr3);
        torch::Tensor mask = mask_u8_to_float01_tensor(mask_u8c1);

        image = image.contiguous().to(device);
        mask = mask.contiguous().to(device);

        torch::Tensor output;
        {
            c10::InferenceMode infer_guard;
            std::vector<torch::jit::IValue> inputs = {image, mask};
            auto forward_method = model.get_method("forward");
            output = forward_method(inputs).toTensor();
        }

        output = output.slice(2, 0, orig_h).slice(3, 0, orig_w).cpu().contiguous();

        auto chw = output.squeeze(0);
        torch::Tensor gray01 = chw.mean(/*dim=*/0).contiguous();

        cv::Mat out_mat(orig_h, orig_w, CV_32FC1, gray01.data_ptr<float>());
        out_mat = out_mat.clone();
        out_mat *= denom;
        out_mat += static_cast<float>(minv);

        *out_32fc1 = out_mat;
        return true;
    } catch (const c10::Error &e) {
        if (err_msg) {
            *err_msg = e.what();
        }
        return false;
    } catch (const std::exception &e) {
        if (err_msg) {
            *err_msg = e.what();
        }
        return false;
    }
}

} // namespace lama
