#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include "cmdline.h"
#include "models/phi3v/configuration_phi3v.hpp"
#include "models/phi3v/modeling_phi3v.hpp"
#include "models/phi3v/processing_phi3v.hpp"
#include "processor/PostProcess.hpp"

namespace fs = std::filesystem;
using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/phi3v_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/phi-3-vision-instruct-q4_k.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 2500);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    int thread_num = cmdParser.get<int>("thread");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    ParamLoader param_loader(model_path);
    auto processor = Phi3VProcessor(vocab_path);
    Phi3VConfig config(tokens_limit, "3.8B");
    auto model_config = Phi3VConfig(config);
    auto model = Phi3VModel(model_config);
    model.load(model_path);

    // Collect all .jpg images from ../images folder
    vector<string> in_imgs;
    for (const auto &entry : fs::directory_iterator("../images")) {
        if (entry.path().extension() == ".jpg") {
            in_imgs.push_back(entry.path().string());
        }
    }

    // Input string for all images
    string in_str = "<|image_1|>\nDescribe the image in detail.";

    // Open output file
    std::ofstream output_file("output.txt");
    if (!output_file.is_open()) {
        std::cerr << "Failed to open output.txt for writing!" << std::endl;
        return 1;
    }

    for (const auto &img_path : in_imgs) {
        std::cout << "[Processing Image] " << img_path << std::endl;
        output_file << "[Image] " << img_path << std::endl;

        auto processed_str = processor.tokenizer->apply_chat_template(in_str);
        auto input_tensor = processor.process(processed_str, img_path);
        input_tensor[1].saveData<float>();
        input_tensor[2].saveData<float>();

        output_file << "[Q] " << in_str << std::endl;
        output_file << "[A] ";

        for (int step = 0; step < 100; step++) {
            auto result = model(input_tensor);
            auto outputs = processor.detokenize(result[0]);
            auto out_string = outputs.first;
            auto out_token = outputs.second;
            auto [not_end, output_string] = processor.tokenizer->postprocess(out_string);

            output_file << output_string;
            std::cout << output_string << std::flush;

            if (!not_end) {
                break;
            }
            chatPostProcessing(out_token, input_tensor[0], {&input_tensor[1], &input_tensor[2]});
        }

        output_file << "\n\n";
        std::cout << std::endl;
    }

    output_file.close();
    return 0;
}
