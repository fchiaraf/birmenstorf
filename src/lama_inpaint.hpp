#pragma once

#ifdef WITH_LAMA_INPAINT

#include <filesystem>
#include <memory>
#include <string>

#include <opencv2/core.hpp>
#include <torch/script.h>

namespace lama {

torch::Device selectDevice(const std::string &hint);

bool loadTracedModel(const std::filesystem::path &model_path,
                      torch::Device device,
                      std::unique_ptr<torch::jit::script::Module> *out_model,
                      std::string *err_msg);

bool inpaintGray32f(const cv::Mat &img_32fc1,
                    const cv::Mat &mask_u8c1,
                    const torch::jit::script::Module &model,
                    const torch::Device &device,
                    cv::Mat *out_32fc1,
                    std::string *err_msg);

} // namespace lama

#endif // WITH_LAMA_INPAINT
