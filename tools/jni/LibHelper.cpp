#include "LibHelper.hpp"
#include <Types.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "Generate.hpp"
#include "models/bert/configuration_bert.hpp"
#include "models/bert/modeling_bert.hpp"
#include "models/bert/tokenization_bert.hpp"
#include "models/fuyu/configuration_fuyu.hpp"
#include "models/fuyu/modeling_fuyu.hpp"
#include "models/phonelm/configuration_phonelm.hpp"
#include "models/phonelm/modeling_phonelm.hpp"
#include "models/qwen/configuration_qwen.hpp"
#include "models/qwen/modeling_qwen.hpp"
#include "models/qwen/tokenization_qwen.hpp"
#include "models/smollm/tokenization_smollm.hpp"
#include "tokenizers/Unigram/Unigram.hpp"
#include "models/fuyu/processing_fuyu.hpp"
#include "processor/PostProcess.hpp"
using namespace mllm;

#ifdef USE_QNN
#include "models/qwen/modeling_qwen_npu.hpp"
#include "models/phonelm/modeling_phonelm_npu.hpp"

#include "models/phi3v/configuration_phi3v.hpp"
#include "models/phi3v/modeling_phi3v.hpp"
#include "models/phi3v/processing_phi3v.hpp"

#include "models/smollm/configuration_smollm.hpp"
#include "models/smollm/modeling_smollm.hpp"
#include "models/smollm/tokenization_smollm.hpp"

#include "models/openelm/configuration_openelm.hpp"
#include "models/openelm/modeling_openelm.hpp"
#include "models/llama/tokenization_llama.hpp"

#include <fstream>
#include <string>
#include <iostream> // Optional: For additional debugging
#include <sys/stat.h>

void logToFile(const std::string& message) {
    // Define the path to the debugger.txt file in the Download directory
    static const std::string logPath = "/storage/emulated/0/Download/debugger.txt"; 
    static std::ofstream logFile(logPath, std::ios::out | std::ios::app);

    if (logFile.is_open()) {
        logFile << message << std::endl; // Write the message
    } else {
        // If the file can't be opened, log to standard error for debugging
        std::cerr << "Failed to open log file: " << logPath << std::endl;
    }
}


#endif
inline bool exists_test(const std::string &name) {
    std::ifstream f(name.c_str());
    return f.good();
}

unsigned int LibHelper::postProcessing(shared_ptr<Tensor> result, shared_ptr<Tensor> &out_result) const {
    return 0;
}

ssize_t getFileSize(const std::string& filePath) {
    struct stat stat_buf;
    if (stat(filePath.c_str(), &stat_buf) == 0) {
        return stat_buf.st_size;
    } else {
        LOGE("Failed to get file size for: %s", filePath.c_str());
        logToFile("Failed to get file size for: " + filePath);
        return -1;
    }
}

bool LibHelper::setUp(const std::string &base_path, std::string weights_path, std::string qnn_weights_path, std::string vocab_path, std::string merge_path, PreDefinedModel model, MLLMBackendType backend_type) {
    FuyuConfig fuyuconfig(tokens_limit, "8B");
    QWenConfig qwconfig(tokens_limit, "1.5B");
    SmolLMConfig smolconfig(tokens_limit, "1.7B", RoPEType::HFHUBROPE, 49152);
    OpenELMConfig openelmconfig(tokens_limit, "1.1B", RoPEType::HFHUBROPE);
    BertConfig bertconfig;
    PhoneLMConfig phone_config(tokens_limit, "1.5B");
    Phi3VConfig phi3vconfig(tokens_limit, "3.8B");
    vocab_path = base_path + vocab_path;
    merge_path = base_path + merge_path;
    weights_path = base_path + weights_path;
    qnn_weights_path = base_path + qnn_weights_path;
    model_ = model;
    backend_ = backend_type;
    ParamLoader param_loader(weights_path);

    LOGI("Loading qnn model from %s", qnn_weights_path.c_str());
    LOGI("Loading model from %s", weights_path.c_str());

    // LOG to log files here
    logToFile("Loading qnn model from " + qnn_weights_path);
    logToFile("Loading model from " + weights_path);

    // Log to file the size of the weights and vocab file
    ssize_t vocab_size = getFileSize(vocab_path);
    if (vocab_size != -1) {
        LOGI("Vocab file size: %zd bytes", vocab_size);
        logToFile("Vocab file size: " + std::to_string(vocab_size) + " bytes");
    } else {
        LOGE("Unable to determine vocab file size.");
        logToFile("Unable to determine vocab file size.");
    }

    // Get and log weights file size
    ssize_t weights_size = getFileSize(weights_path);
    if (weights_size != -1) {
        LOGI("Weights file size: %zd bytes", weights_size);
        logToFile("Weights file size: " + std::to_string(weights_size) + " bytes");
    } else {
        LOGE("Unable to determine weights file size.");
        logToFile("Unable to determine weights file size.");
    }


    switch (model) {
    case PhoneLM:
        logToFile("Initializing PhoneLM model.");
        tokenizer_ = make_shared<SmolLMTokenizer>(vocab_path, merge_path);
        // tokenizer_ = std::make_any<SmolLMTokenizer*>(new SmolLMTokenizer(vocab_path, merge_path));
        logToFile("Tokenizer");
        module_ = make_shared<PhoneLMForCausalLM>(phone_config);
        logToFile("Module");
        break;
    case SMOLLM:
        LOGI("Loading model SMOLLM");
        logToFile("Initializing SMOLLM model.");
        tokenizer_ = make_shared<SmolLMTokenizer>(vocab_path, merge_path);
        // tokenizer_ = std::make_any<SmolLMTokenizer*>(new SmolLMTokenizer(vocab_path, merge_path));
        logToFile("Tokenizer");
        module_ = make_shared<SmolLMModel>(smolconfig);
        logToFile("Module");
        break;
    case QWEN25:
        qwconfig = QWenConfig(tokens_limit, "1.5B");
        break;
    case QWEN15:
        qwconfig = QWenConfig(tokens_limit, "1.8B");
        tokenizer_ = make_shared<QWenTokenizer>(vocab_path, merge_path);
        // tokenizer_ = std::make_any<QWenTokenizer*>(new QWenTokenizer(vocab_path, merge_path));
        module_ = make_shared<QWenForCausalLM>(qwconfig);
        break;
    case PHI3V:
        phi3v_processor_ = std::make_any<Phi3VProcessor*>(new Phi3VProcessor(vocab_path));
        module_ = make_shared<Phi3VModel>(phi3vconfig);
        break;
    // case FUYU:
    //     processor_ = new FuyuProcessor(vocab_path, 224, 224);
    //     module_ = make_shared<FuyuModel>(fuyuconfig);
    //     break;
    case Bert:
        tokenizer_ = make_shared<BertTokenizer>(vocab_path, true);
        // tokenizer_ = std::make_any<BertTokenizer*>(new BertTokenizer(vocab_path, true));
        module_ = make_shared<BertModel>(bertconfig);
        break;

    }
    logToFile("Trying to Load Weights");
    module_->load(weights_path);

    logToFile("Weights Loaded");
    is_first_run_cond_ = true;

    return true;
}

void LibHelper::setCallback(callback_t callback) {
    this->callback_ = std::move(callback);
}
// void LibHelper::run(std::string &input_str, uint8_t *image, unsigned max_step, unsigned int image_length, bool chat_template) {
void LibHelper::run(std::string &input_str, std::string &image, unsigned max_step, unsigned int image_length, bool chat_template) {
    std::string output_string_;
    LOGE("Running model %d", model_);
    unsigned max_new_tokens = 500;
    LOGE("Running backend %d", backend_);
    vector<double> profiling_data(3);

    if (model_ == QWEN15 || model_ == QWEN25) {
        auto tokenizer = dynamic_pointer_cast<QWenTokenizer>(tokenizer_);
        // QWenTokenizer* tokenizer = std::any_cast<QWenTokenizer*>(tokenizer_);
        if (chat_template) input_str = tokenizer->apply_chat_template(input_str);
        if (backend_ == MLLMBackendType::QNN) {
            int chunk_size = 64;
            auto res = tokenizer->tokenizePaddingByChunk(input_str, chunk_size, 151936);
            auto input_tensor = res.second;
            max_new_tokens = tokens_limit - input_tensor.sequence();
            auto real_seq_length = res.first;
            const int seq_length_padding = (chunk_size - real_seq_length % chunk_size) + real_seq_length;
            const int chunk_num = seq_length_padding / chunk_size;
            bool isSwitched = false;

            LlmTextGeneratorOpts opt{
                .max_new_tokens = 1,
                .do_sample = false,
                .is_padding = true,
                .seq_before_padding = real_seq_length,
                .chunk_size = chunk_size,
            };
            std::vector<Tensor> chunked_tensors(chunk_num);
            for (int chunk_id = 0; chunk_id < chunk_num; ++chunk_id) {
                chunked_tensors[chunk_id].setBackend(Backend::global_backends[MLLM_CPU]);
                chunked_tensors[chunk_id].setTtype(INPUT_TENSOR);
                chunked_tensors[chunk_id].reshape(1, 1, chunk_size, 1);
                chunked_tensors[chunk_id].setName("input-chunk-" + to_string(chunk_id));
                chunked_tensors[chunk_id].deepCopyFrom(&input_tensor, false, {0, 0, chunk_id * chunk_size, 0});

                prefill_module_->generate(chunked_tensors[chunk_id], opt, [&](unsigned int out_token) -> bool {
                    if (!isSwitched && chunk_id == 0 && static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->isStageSwitching()) {
                        // turn off switching at the first chunk of following inputs
                        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();
                        isSwitched = true;
                    }
                    // switch_flag = true;
                    auto out_string = tokenizer->detokenize({out_token});
                    auto [not_end, output_string] = tokenizer->postprocess(out_string);
                    if (chunk_id == chunk_num - 1) { // print the output of the last chunk
                        output_string_ += output_string;
                        if (!not_end) {
                            auto profile_res = prefill_module_->profiling("Prefilling");
                            if (profile_res.size() == 3) {
                                profiling_data[0] += profile_res[0];
                                profiling_data[1] = profile_res[1];
                            }
                            callback_(output_string_, !not_end, profiling_data);
                        }
                        callback_(output_string_, !not_end, {});
                    }
                    return true;
                });
                Module::isFirstChunk = false;
            }
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setSequenceLength(real_seq_length);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setExecutionType(AUTOREGRESSIVE);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();

            opt = LlmTextGeneratorOpts{
                .max_new_tokens = max_new_tokens - 1,
                .do_sample = false,
                .temperature = 0.3f,
                .top_k = 50,
                .top_p = 0.f,
                .is_padding = false,
            };
            isSwitched = false;
            module_->generate(chunked_tensors.back(), opt, [&](unsigned int out_token) -> bool {
                if (!isSwitched) {
                    static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();
                    isSwitched = true;
                }
                auto out_token_string = tokenizer->detokenize({out_token});
                auto [not_end, output_string] = tokenizer->postprocess(out_token_string);
                output_string_ += output_string;
                if (!not_end) {
                    auto profile_res = module_->profiling("Inference");
                    if (profile_res.size() == 3) {
                        profiling_data[0] += profile_res[0];
                        profiling_data[2] = profile_res[2];
                    }
                    callback_(output_string_, !not_end, profiling_data);
                }
                callback_(output_string_, !not_end, {});
                if (!not_end) { return false; }
                return true;
            });
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setSequenceLength(0);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setExecutionType(PROMPT);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();
        } else { // CPU
            auto input_tensor = tokenizer->tokenize(input_str);
            max_new_tokens = tokens_limit - input_tensor.sequence();
            LlmTextGeneratorOpts opt{
                .max_new_tokens = max_new_tokens,
                .do_sample = false,
            };
            module_->generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
                auto out_token_string = tokenizer->detokenize({out_token});
                auto [not_end, output_string] = tokenizer->postprocess(out_token_string);
                output_string_ += output_string;
                if (!not_end) {
                    auto profile_res = module_->profiling("Inference");
                    if (profile_res.size() == 3) {
                        profiling_data = profile_res;
                    }
                    callback_(output_string_, !not_end, profiling_data);
                }
                callback_(output_string_, !not_end, {});
                if (!not_end) { return false; }
                return true;
            });
            module_->clear_kvcache();
        }

    } 
    else if (model_ == SMOLLM) {
        auto tokenizer = dynamic_pointer_cast<SmolLMTokenizer>(tokenizer_);
        // SmolLMTokenizer* tokenizer = std::any_cast<SmolLMTokenizer*>(tokenizer_);
        if (chat_template) input_str = tokenizer->apply_chat_template(input_str);
            auto input_tensor = tokenizer->tokenize(input_str);
            max_new_tokens = tokens_limit - input_tensor.sequence();
            LlmTextGeneratorOpts opt{
                .max_new_tokens = max_new_tokens,
                .do_sample = false,
            };
            module_->generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
                auto out_token_string = tokenizer->detokenize({out_token});
                auto [not_end, output_string] = tokenizer->postprocess(out_token_string);
                output_string_ += output_string;
                if (!not_end) {
                    auto profile_res = module_->profiling("Inference");
                    if (profile_res.size() == 3) {
                        profiling_data = profile_res;
                    }
                    callback_(output_string_, !not_end, profiling_data);
                }
                callback_(output_string_, !not_end, {});
                if (!not_end) { return false; }
                return true;
            });
            module_->clear_kvcache();
    } 
    else if (model_ == PHI3V) {
        LOGI("LOADING PHI3V");
        Phi3VProcessor* phi3v_processor = std::any_cast<Phi3VProcessor*>(phi3v_processor_);
        LOGI("LOADED PHI3V");
            auto input_str_img = "<|image_1|>\n" + input_str;
            auto in_str = phi3v_processor->tokenizer->apply_chat_template(input_str_img);
            LOGI("Chat template applied");
            LOGI("Image path: %s", image.c_str());
            ssize_t image_size = getFileSize(image);
            if (image_size != -1) {
                LOGI("Image file size: %zd bytes", image_size);
                logToFile("Image file size: " + std::to_string(image_size) + " bytes");
            } else {
                LOGE("Unable to determine image file size.");
                logToFile("Unable to determine image file size.");
            }
            auto input_tensor = phi3v_processor->process(in_str, image);
            LOGI("Input tensor processed");
            input_tensor[1].saveData<float>();
            input_tensor[2].saveData<float>();
            LOGI("String processed");
            for (int step = 0; step < 100; step++) {
                auto result = (*module_)(input_tensor);
                LOGI("Result processed");
                auto outputs = phi3v_processor->detokenize(result[0]);
                auto out_string = outputs.first;
                auto out_token = outputs.second;
                LOGI("Output string: %s", out_string.c_str());

                auto [not_end, output_string] = phi3v_processor->tokenizer->postprocess(out_string);
                output_string_ += output_string;
                callback_(output_string_, !not_end, {});
                if (!not_end) { break; }
                chatPostProcessing(out_token, input_tensor[0], {&input_tensor[1], &input_tensor[2]});
            }
            module_->clear_kvcache();
            
    }

    // else if (model_ == FUYU) {
    //     auto processor = dynamic_cast<FuyuProcessor *>(processor_);
    //     auto input_tensors = processor->process(input_str, {image}, {image_length});
    //     for (int step = 0; step < 100; step++) {
    //         auto result = (*module_)({input_tensors[0], input_tensors[1], input_tensors[2]});
    //         auto outputs = processor->detokenize(result[0]);
    //         auto out_string = outputs.first;
    //         auto out_token = outputs.second;
    //         auto [end, string] = processor->postprocess(out_string);
    //         output_string_ += string;
    //         callback_(output_string_, !end, {});
    //         if (!end) { break; }
    //         chatPostProcessing(out_token, input_tensors[0], {&input_tensors[1], &input_tensors[2]});
    //     }
    //     module_->clear_kvcache();
    // } 
    
    else if (model_ == Bert) {
        LOGE("Bert model is not supported in this version.");
    } else if (model_ == PhoneLM) {
        // static bool switch_flag = false;
        auto tokenizer = dynamic_pointer_cast<SmolLMTokenizer>(tokenizer_);
        // SmolLMTokenizer* tokenizer = std::any_cast<SmolLMTokenizer*>(tokenizer_);
        if (chat_template) input_str = tokenizer->apply_chat_template(input_str);
        if (backend_ == MLLMBackendType::QNN) {
            int chunk_size = 64;
            auto res = tokenizer->tokenizePaddingByChunk(input_str, chunk_size, 49152);
            auto input_tensor = res.second;
            max_new_tokens = tokens_limit - input_tensor.sequence();
            auto real_seq_length = res.first;
            const int seq_length_padding = (chunk_size - real_seq_length % chunk_size) + real_seq_length;
            const int chunk_num = seq_length_padding / chunk_size;
            bool isSwitched = false;

            LlmTextGeneratorOpts opt{
                .max_new_tokens = 1,
                .do_sample = false,
                .is_padding = true,
                .seq_before_padding = real_seq_length,
                .chunk_size = chunk_size,
            };
            std::vector<Tensor> chunked_tensors(chunk_num);
            for (int chunk_id = 0; chunk_id < chunk_num; ++chunk_id) {
                chunked_tensors[chunk_id].setBackend(Backend::global_backends[MLLM_CPU]);
                chunked_tensors[chunk_id].setTtype(INPUT_TENSOR);
                chunked_tensors[chunk_id].reshape(1, 1, chunk_size, 1);
                chunked_tensors[chunk_id].setName("input-chunk-" + to_string(chunk_id));
                chunked_tensors[chunk_id].deepCopyFrom(&input_tensor, false, {0, 0, chunk_id * chunk_size, 0});

                prefill_module_->generate(chunked_tensors[chunk_id], opt, [&](unsigned int out_token) -> bool {
                    // if (switch_flag && !isSwitched && chunk_id == 0) {
                    if (!isSwitched && chunk_id == 0) {
                        // turn off switching at the first chunk of following inputs
                        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();
                        isSwitched = true;
                    }
                    // switch_flag = true;
                    auto out_string = tokenizer->detokenize({out_token});
                    auto [not_end, output_string] = tokenizer->postprocess(out_string);
                    if (chunk_id == chunk_num - 1) { // print the output of the last chunk
                        output_string_ += output_string;
                        if (!not_end) {
                            auto profile_res = prefill_module_->profiling("Prefilling");
                            if (profile_res.size() == 3) {
                                profiling_data[0] += profile_res[0];
                                profiling_data[1] = profile_res[1];
                            }
                            callback_(output_string_, !not_end, profiling_data);
                        }
                        callback_(output_string_, !not_end, {});
                    }
                    if (!not_end) { return false; }
                    return true;
                });
                Module::isFirstChunk = false;
            }
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setSequenceLength(real_seq_length);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setExecutionType(AUTOREGRESSIVE);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();

            opt = LlmTextGeneratorOpts{
                .max_new_tokens = max_new_tokens - 1,
                .do_sample = false,
                .temperature = 0.3f,
                .top_k = 50,
                .top_p = 0.f,
                .is_padding = false,
            };
            isSwitched = false;
            module_->generate(chunked_tensors.back(), opt, [&](unsigned int out_token) -> bool {
                if (!isSwitched) {
                    static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();
                    isSwitched = true;
                }
                auto out_token_string = tokenizer->detokenize({out_token});
                auto [not_end, output_string] = tokenizer->postprocess(out_token_string);
                output_string_ += output_string;
                if (!not_end) {
                    auto profile_res = module_->profiling("Inference");
                    if (profile_res.size() == 3) {
                        profiling_data[0] += profile_res[0];
                        profiling_data[2] = profile_res[2];
                    }
                    callback_(output_string_, !not_end, profiling_data);
                }
                callback_(output_string_, !not_end, {});
                if (!not_end) { return false; }
                return true;
            });
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setSequenceLength(0);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setExecutionType(PROMPT);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();
        } else { // CPU
            auto input_tensor = tokenizer->tokenize(input_str);
            max_new_tokens = tokens_limit - input_tensor.sequence();
            LlmTextGeneratorOpts opt{
                .max_new_tokens = max_new_tokens,
                .do_sample = false,
            };
            module_->generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
                auto out_token_string = tokenizer->detokenize({out_token});
                auto [not_end, output_string] = tokenizer->postprocess(out_token_string);
                output_string_ += output_string;
                if (!not_end) {
                    auto profile_res = module_->profiling("Inference");
                    if (profile_res.size() == 3) {
                        profiling_data = profile_res;
                    }
                    callback_(output_string_, !not_end, profiling_data);
                }
                callback_(output_string_, !not_end, {});
                if (!not_end) { return false; }
                return true;
            });
            module_->clear_kvcache();
        }
    }
}
std::vector<float> LibHelper::runForResult(std::string &input_str) {
    LOGE("Running model %d", model_);
}

LibHelper::~LibHelper() {
    if (phi3v_processor_.has_value()) {
        // Attempt to cast phi3v_processor_ to a pointer to Phi3VProcessor
        Phi3VProcessor* processor_ptr = std::any_cast<Phi3VProcessor*>(phi3v_processor_);
        if (processor_ptr) {
            // Only delete if it was allocated dynamically using new
            delete processor_ptr;
            LOGI("Deleted phi3v_processor_");
        }
    }
}
// #endif
