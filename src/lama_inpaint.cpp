/**
 * lama_inpaint.cpp
 * ----------------
 * LaMa inpainting inference using LibTorch + OpenCV.
 *
 * Build:  see CMakeLists.txt
 * Usage:
 *   ./lama_inpaint --model lama_traced.pt \
 *                  --image photo.tiff \
 *                  --mask  mask.tiff  \
 *                  --output result.tiff   (file path, or an existing directory → inpainted.tiff inside it)
 *                  [--device cpu|cuda]  (default: cuda if available)
 *
 *  H/W padding to a multiple of 8 is done inside the exported TorchScript wrapper; do not pre-pad in C++.
 *
 * Mask: 0 = keep original, 255 = inpaint gap (same as LaMa / inference.py).
 */

 #include <torch/script.h>
 #include <torch/torch.h>
 #include <c10/util/Exception.h>
 #include <opencv2/opencv.hpp>


 #include <algorithm>
 #include <cctype>
 #include <iostream>
 #include <fstream>
 #include <string>
 #include <stdexcept>
 #include <chrono>
 #include <filesystem>
 #include <optional>

 #include <queue>
 #include <mutex>
 #include <condition_variable>
 #include <future>
 #include <thread>
 
 namespace fs = std::filesystem;
 
 static std::string lowercase_string(std::string s) {
     std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
     return s;
 }
 
 // ---------------------------------------------------------------------------
 // CLI argument parsing (simple, no external deps)
 // ---------------------------------------------------------------------------
 struct Args {
     std::string model_path  = "lama_traced_test.pt";
     std::string image_path;
     std::string mask_path;
     std::string output_path = "result.tiff";
     std::string device_str  = "auto";   // "cpu", "cuda", "auto"
 };
 
 static bool looks_like_cli_flag(const std::string& v) {
     return v.size() >= 2 && v[0] == '-' && v[1] == '-';
 }
 
 // Consumes argv[i+1]; exits if missing or if value is another flag (e.g. `--output --output path`).
 static std::string take_path_arg(const char* flag_name, int& i, int argc, char** argv) {
     if (i + 1 >= argc) {
         std::cerr << "Error: missing value after " << flag_name << "\n";
         std::exit(1);
     }
     std::string v = argv[++i];
     if (looks_like_cli_flag(v)) {
         std::cerr << "Error: " << flag_name << " was followed by \"" << v
                   << "\" - that looks like another flag.\n"
                   << "  Fix: use " << flag_name << " only once, e.g. "
                   << flag_name << " \"C:\\\\path\\\\out.tiff\"\n"
                   << "  (Typing --output twice makes the first one eat the second --output as the path.)\n";
         std::exit(1);
     }
     return v;
 }
 
 Args parse_args(int argc, char** argv) {
     Args a;
     for (int i = 1; i < argc; ++i) {
         std::string key = argv[i];
         if (key == "--model")
             a.model_path = take_path_arg("--model", i, argc, argv);
         else if (key == "--image")
             a.image_path = take_path_arg("--image", i, argc, argv);
         else if (key == "--mask")
             a.mask_path = take_path_arg("--mask", i, argc, argv);
         else if (key == "--output")
             a.output_path = take_path_arg("--output", i, argc, argv);
         else if (key == "--device" && i + 1 < argc)
             a.device_str = argv[++i];
         else {
             std::cerr << "Unknown argument: " << key << "\n";
         }
     }
     if (a.image_path.empty() || a.mask_path.empty()) {
         std::cerr << "Usage: lama_inpaint --model lama.pt --image img.tiff "
                      "--mask mask.tiff --output result.tiff [--device cpu|cuda]\n";
         std::exit(1);
     }
     return a;
 }
 
 // ---------------------------------------------------------------------------
 // Device selection
 // ---------------------------------------------------------------------------
 torch::Device select_device(const std::string& hint) {
     if (hint == "cpu") {
         std::cout << "[device] using CPU (--device cpu)\n" << std::flush;
         return torch::kCPU;
     }
     if (hint == "cuda") {
         std::cout << "[device] checking CUDA (--device cuda) ...\n" << std::flush;
         if (!torch::cuda::is_available())
             throw std::runtime_error("CUDA requested but not available.");
         std::cout << "[device] using CUDA\n" << std::flush;
         return torch::kCUDA;
     }
     // auto: probe CUDA only after a line is printed (probing can hang on broken drivers)
     std::cout << "[device] auto: probing CUDA ...\n" << std::flush;
     if (torch::cuda::is_available()) {
         std::cout << "[device] CUDA available - using GPU\n" << std::flush;
         return torch::kCUDA;
     }
     std::cout << "[device] CUDA not available - using CPU\n" << std::flush;
     return torch::kCPU;
 }
 
 // ---------------------------------------------------------------------------
 // Image / mask I/O
 // ---------------------------------------------------------------------------
 
 // Load an RGB image as float32 tensor [1, 3, H, W] in [0, 1]
 torch::Tensor load_image(const std::string& path) {
     cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
     if (img.empty())
         throw std::runtime_error("Cannot read image: " + path);
 
     // BGR -> RGB, uint8 -> float32 [0,1]
     cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
     img.convertTo(img, CV_32FC3, 1.0 / 255.0);
 
     if (!img.isContinuous())
         img = img.clone();
 
     // HWC -> CHW
     auto tensor = torch::from_blob(img.data,
         {1, img.rows, img.cols, 3}, torch::kFloat32).clone();
     tensor = tensor.permute({0, 3, 1, 2});   // [1,3,H,W]
     return tensor;
 }
 
 // Load a mask as float32 tensor [1, 1, H, W], values in {0, 1}.
 // 0 = keep original, 255 (bright) = inpaint gap — same as LaMa / torchscript.py.
 torch::Tensor load_mask(const std::string& path) {
     cv::Mat raw = cv::imread(path, cv::IMREAD_UNCHANGED);
     if (raw.empty())
         throw std::runtime_error("Cannot read mask: " + path);
 
     cv::Mat gray;
     if (raw.channels() == 1)
         gray = raw;
     else if (raw.channels() == 3)
         cv::cvtColor(raw, gray, cv::COLOR_BGR2GRAY);
     else if (raw.channels() == 4)
         cv::cvtColor(raw, gray, cv::COLOR_BGRA2GRAY);
     else
         throw std::runtime_error("Unsupported mask channel count: " + path);
 
     cv::Mat binary_f;
 
     // 8-bit 0/255: map with fixed scale (avoids min-max edge cases on near-constant strips).
     if (gray.depth() == CV_8U) {
         cv::Mat gray_f;
         gray.convertTo(gray_f, CV_32FC1, 1.0f / 255.0f);
         cv::threshold(gray_f, binary_f, 0.5f, 1.0f, cv::THRESH_BINARY);
     } else {
         cv::Mat gray_f;
         gray.convertTo(gray_f, CV_32FC1);
         double minv = 0.0, maxv = 0.0;
         cv::minMaxLoc(gray_f, &minv, &maxv);
         if (maxv <= minv)
             throw std::runtime_error("Empty or flat mask: " + path);
         gray_f = (gray_f - static_cast<float>(minv)) / static_cast<float>(maxv - minv);
         cv::threshold(gray_f, binary_f, 0.5f, 1.0f, cv::THRESH_BINARY);
     }
 
     if (!binary_f.isContinuous())
         binary_f = binary_f.clone();
 
     auto tensor = torch::from_blob(binary_f.data,
         {1, binary_f.rows, binary_f.cols, 1}, torch::kFloat32).clone();
     tensor = tensor.permute({0, 3, 1, 2});   // [1,1,H,W]
     return tensor;
 }
 
 // Save float32 tensor [1, 3, H, W] in [0,1] to tiff
 // If path is a directory (existing, or clearly meant as a dir via trailing slash), write
 // inpainted.tiff inside (creating the directory if needed). Otherwise path is the image file path.
 std::string resolve_output_file(std::string path) {
     const bool trailing_sep = !path.empty() && (path.back() == '/' || path.back() == '\\');
     while (!path.empty() && (path.back() == '/' || path.back() == '\\'))
         path.pop_back();
     std::error_code ec;
     if (path.empty())
         throw std::runtime_error("Empty --output path");
     fs::path p(path);
     const bool is_dir = fs::is_directory(p, ec);
     if (trailing_sep || is_dir) {
         fs::create_directories(p, ec);
         if (ec)
             throw std::runtime_error("Cannot create output directory: " + p.string());
         return (p / "inpainted.tiff").string();
     }
     if (p.has_parent_path()) {
         fs::create_directories(p.parent_path(), ec);
         if (ec)
             throw std::runtime_error("Cannot create output directory: " + p.parent_path().string());
     }
     return path;
 }
 
 void save_image(const torch::Tensor& t, const std::string& path) {
     // CHW -> HWC, to CPU, contiguous
     auto img_t = t.squeeze(0).permute({1, 2, 0}).contiguous().cpu();
     cv::Mat img(img_t.size(0), img_t.size(1), CV_32FC3, img_t.data_ptr<float>());
     img.convertTo(img, CV_8UC3, 255.0);
     cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
     if (!cv::imwrite(path, img))
         throw std::runtime_error("Failed to write: " + path);
 }
 
 // ---------------------------------------------------------------------------
 // Main
 // ---------------------------------------------------------------------------
 int main(int argc, char** argv) {
     // Unbuffered-ish startup so the console shows something even if LibTorch/CUDA blocks later.
     std::cerr << "lama_inpaint: starting (parse args, load torch, ...)\n" << std::flush;
 
     Args args = parse_args(argc, argv);

    // Use all available CPU cores for intra-op parallelism
    // Physical cores only (divide logical count by 2 to strip hyperthreads)
     int logical  = static_cast<int>(std::thread::hardware_concurrency());
     int physical = std::max(1, logical * 2);
     torch::set_num_threads(physical);
     torch::set_num_interop_threads(1);  // LaMa is sequential, no inter-op parallelism
     std::cout << "[threads] intra-op=" << physical << " inter-op=1\n";
 
     // ---- Device ----
     torch::Device device = select_device(args.device_str);
     torch::globalContext().setBenchmarkCuDNN(true);  // enables oneDNN tuning on CPU too
 
     const fs::path model_file = fs::absolute(args.model_path);
 
     // ---- Load TorchScript model ----
     {
         std::error_code ec;
         if (!fs::is_regular_file(model_file, ec)) {
             std::cerr << "[ERROR] --model is not a readable file: " << model_file.string() << "\n";
             return 1;
         }
         const std::string ext = lowercase_string(model_file.extension().string());
         if (ext == ".ckpt" || ext == ".pth") {
             std::cerr << "[ERROR] --model must be a TorchScript export (.pt), not a training checkpoint (" << ext << ").\n"
                       << "  In Python, export first, e.g.:\n"
                       << "    python torchscript.py --model-path ./big-lama --output lama_traced.pt\n"
                       << "  Then: --model .../lama_traced.pt\n";
             return 1;
         }
     }
     std::cout << "Loading model: " << model_file.string() << " ...\n";
     torch::jit::script::Module model;
     try {
         // Avoid torch::jit::load(path, device) on Windows: it uses fopen on a narrow path and can
         // fail with errno 22 (EINVAL) even for valid paths. std::ifstream(fs::path) uses the wide
         // Win32 API; istream-based jit::load then reads the archive reliably.
         std::ifstream ifs(model_file, std::ios::binary);
         if (!ifs) {
             std::cerr << "[ERROR] Could not open model file for reading: " << model_file.string() << "\n";
             return 1;
         }
         model = torch::jit::load(ifs, std::optional<c10::Device>{device});
     } catch (const c10::Error& e) {
         std::cerr << "Error loading model: " << e.what() << "\n"
                   << "  Hint: export the .pt with PyTorch whose major.minor matches this LibTorch build.\n";
         return 1;
     }
     model.eval();
     model = torch::jit::optimize_for_inference(model);
     std::cout << "      Model loaded OK\n";
 
     // ---- Load inputs ----
     std::cout << "Loading image & mask ...\n";
     torch::Tensor image, mask;
     try {
         image = load_image(args.image_path);
         mask  = load_mask(args.mask_path);
     } catch (const std::exception& e) {
         std::cerr << e.what() << "\n";
         return 1;
     }
 
     // Sanity check: image and mask must be the same H x W
     if (image.size(2) != mask.size(2) || image.size(3) != mask.size(3)) {
         std::cerr << "Image and mask must have the same resolution.\n"
                   << "  Image: " << image.size(3) << "x" << image.size(2) << "\n"
                   << "  Mask:  " << mask.size(3)  << "x" << mask.size(2)  << "\n";
         return 1;
     }
     std::cout << "      Input size: " << image.size(3) << " x " << image.size(2) << "\n";

 
     const int orig_h = static_cast<int>(image.size(2));
     const int orig_w = static_cast<int>(image.size(3));
 
     // ---- Preprocessing: only move to device. The traced LaMa wrapper pads/crops internally
     // (same as torchscript.py). Extra reflect-pad here was redundant and could assert/crash
     // on some LibTorch builds for certain H×W.
     std::cout << "Preprocessing (move tensors to device) ...\n";
     try {
         {
             double hole_frac = mask.mean().item<double>();
             std::cout << "      Mask inpaint pixel fraction: " << (hole_frac * 100.0) << "%\n";
         }
         image = image.contiguous().to(device);
         mask  = mask.contiguous().to(device);
         
     } catch (const c10::Error& e) {
         std::cerr << "[ERROR] Preprocessing failed: " << e.what() << "\n";
         return 1;
     }
 
     // ---- Inference ----
     std::cout << "Running inference ...\n";
     auto t_start = std::chrono::high_resolution_clock::now();
 
     torch::Tensor output;
     try {
         torch::NoGradGuard no_grad;
         std::vector<torch::jit::IValue> inputs = {image, mask};
         auto forward_method = model.get_method("forward");
         output = forward_method(inputs).toTensor();
     } catch (const c10::Error& e) {
         std::cerr << "[ERROR] Inference failed: " << e.what() << "\n";
         return 1;
     }
 
     auto t_end = std::chrono::high_resolution_clock::now();
     double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
     std::cout << "      Inference time: " << ms << " ms\n";
 
     // ---- Crop back to original size & save ----
     std::string out_file;
     try {
         out_file = resolve_output_file(args.output_path);
     } catch (const std::exception& e) {
         std::cerr << e.what() << "\n";
         return 1;
     }
     std::cout << "Saving result to " << out_file << " ...\n";
     output = output.slice(2, 0, orig_h).slice(3, 0, orig_w);  // crop H, W
     output = output.cpu();
 
     try {
         save_image(output, out_file);
     } catch (const std::exception& e) {
         std::cerr << e.what() << "\n";
         return 1;
     }
 
     std::cout << "Done! Result saved to: " << out_file << "\n";
     return 0;
 }
 