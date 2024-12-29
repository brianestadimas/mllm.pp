// #include <iostream>
// #include "cmdline.h"
// #include "models/phi3v/configuration_phi3v.hpp"
// #include "models/phi3v/modeling_phi3v.hpp"
// #include "models/phi3v/processing_phi3v.hpp"
// #include "processor/PostProcess.hpp"

// using namespace mllm;
// int main(int argc, char **argv) {
//     cmdline::parser cmdParser;
//     cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/phi3v_vocab.mllm");
//     cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/phi-3-vision-instruct-q4_k.mllm");
//     cmdParser.add<int>("limits", 'l', "max KV cache size", false, 2500);
//     cmdParser.add<int>("thread", 't', "num of threads", false, 4);
//     cmdParser.parse_check(argc, argv);

//     string vocab_path = cmdParser.get<string>("vocab");
//     string model_path = cmdParser.get<string>("model");
//     int tokens_limit = cmdParser.get<int>("limits");
//     int thread_num = cmdParser.get<int>("thread");
//     CPUBackend::cpu_threads = cmdParser.get<int>("thread");

//     ParamLoader param_loader(model_path);
//     auto processor = Phi3VProcessor(vocab_path);
//     Phi3VConfig config(tokens_limit, "3.8B");
//     auto model_config = Phi3VConfig(config);
//     auto model = Phi3VModel(model_config);
//     model.load(model_path);

//     vector<string> in_imgs = {
//         "../assets/australia.jpg"};
//     vector<string> in_strs = {
//         "<|image_1|>\nWhat's the content of the image?",
//     };

//     for (int i = 0; i < in_strs.size(); ++i) {
//         auto in_str = in_strs[i];
//         in_str = processor.tokenizer->apply_chat_template(in_str);
//         auto input_tensor = processor.process(in_str, in_imgs[i]);
//         input_tensor[1].saveData<float>();
//         input_tensor[2].saveData<float>();
//         std::cout << "[Q] " << in_strs[i] << std::endl;
//         std::cout << "[A] " << std::flush;
//         for (int step = 0; step < 100; step++) {
//             auto result = model(input_tensor);
//             auto outputs = processor.detokenize(result[0]);
//             auto out_string = outputs.first;
//             auto out_token = outputs.second;
//             auto [not_end, output_string] = processor.tokenizer->postprocess(out_string);
//             if (!not_end) { break; }
//             std::cout << output_string << std::flush;
//             chatPostProcessing(out_token, input_tensor[0], {&input_tensor[1], &input_tensor[2]});
//         }
//         printf("\n");
//     }

//     return 0;
// }

#include <iostream>
#include <fstream>
#include "cmdline.h"
#include "models/phi3v/configuration_phi3v.hpp"
#include "models/phi3v/modeling_phi3v.hpp"
#include "models/phi3v/processing_phi3v.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    // --- 1) Define command-line arguments ---
    cmdline::parser cmdParser;
    cmdParser.add<std::string>("vocab", 'v', 
        "specify mllm tokenizer model path",
        false, 
        "../vocab/phi3v_vocab.mllm"
    );
    cmdParser.add<std::string>("model", 'm', 
        "specify mllm model path",
        false, 
        "../models/phi-3-vision-instruct-q4_k.mllm"
    );
    cmdParser.add<int>("limits", 'l', 
        "max KV cache size",
        false, 
        2500
    );
    cmdParser.add<int>("thread", 't', 
        "num of threads",
        false, 
        4
    );

    // New args: input image and output file
    cmdParser.add<std::string>("input", 'i', 
        "Path to input image",
        false, 
        "../assets/australia.jpg"
    );
    cmdParser.add<std::string>("output", 'o', 
        "Path to output text file",
        false, 
        "./output.txt"
    );

    cmdParser.parse_check(argc, argv);

    // --- 2) Read command-line arguments ---
    std::string vocab_path   = cmdParser.get<std::string>("vocab");
    std::string model_path   = cmdParser.get<std::string>("model");
    int tokens_limit         = cmdParser.get<int>("limits");
    int thread_num           = cmdParser.get<int>("thread");
    std::string input_image  = cmdParser.get<std::string>("input");
    std::string output_file  = cmdParser.get<std::string>("output");

    // If you have a global static that sets # of threads
    CPUBackend::cpu_threads = thread_num;

    // --- 3) Set up model, processor, etc. ---
    ParamLoader param_loader(model_path);
    auto processor = Phi3VProcessor(vocab_path);

    // Adjust model configuration as needed
    Phi3VConfig config(tokens_limit, "3.8B");
    auto model_config = Phi3VConfig(config);
    auto model = Phi3VModel(model_config);
    model.load(model_path);

    // For simplicity, we assume a single image and single question
    // If you want multiple images, you'd gather them here.
    std::vector<std::string> in_imgs = { input_image };
    // In practice, you'd either:
    //   - Hard-code the question,
    //   - Or read from another command line arg or file.
    std::vector<std::string> in_strs = {
        "<|image_1|>\nWhat's the content of the image?",
    };

    // --- 4) Open output stream ---
    // If you only want to print to a file, do that.
    // If you also want to print to console, keep std::cout usage too.
    std::ofstream ofs(output_file);
    if (!ofs.is_open()) {
        std::cerr << "Error: cannot open output file: " << output_file << std::endl;
        return 1;
    }

    for (int i = 0; i < (int)in_strs.size(); ++i) {
        // Get user prompt, apply chat template
        auto in_str = processor.tokenizer->apply_chat_template(in_strs[i]);

        // Preprocess input (text + image)
        auto input_tensor = processor.process(in_str, in_imgs[i]);

        // Debug or diagnostic printing
        std::cout << "[Q] " << in_strs[i] << std::endl;
        std::cout << "[A] " << std::flush;

        ofs << "[Q] " << in_strs[i] << "\n";
        ofs << "[A] ";

        // --- 5) Generate text tokens until done ---
        for (int step = 0; step < 100; step++) {
            auto result = model(input_tensor);
            auto [output_tokens_str, output_tokens_id] = processor.detokenize(result[0]);

            // Post-process
            auto [not_end, output_string] = processor.tokenizer->postprocess(output_tokens_str);
            if (!not_end) {
                // If the model has reached a stopping token, end generation
                ofs << output_string << "\n"; // Final chunk
                std::cout << output_string << std::endl;
                break;
            }

            // Print partial text
            std::cout << output_string << std::flush;
            ofs << output_string << std::flush;

            // Perform "chat style" post-processing (next input state)
            chatPostProcessing(
                output_tokens_id,       // newly generated tokens
                input_tensor[0],        // textual input
                { &input_tensor[1], &input_tensor[2] }  // images
            );
        }
    }

    ofs.close();
    return 0;
}
