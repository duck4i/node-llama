#include "npm-llama.h"
#include "llama-cpp.h"

llama_model* loadModel(const std::string& model_path) {
    ggml_backend_load_all();
    
    llama_model_params model_params = llama_model_default_params();
    llama_model* model = llama_load_model_from_file(model_path.c_str(), model_params);
    
    if (model == nullptr) {
        fprintf(stderr, "Error: Unable to load model from %s\n", model_path.c_str());
        return nullptr;
    }
    
    return model;
}

std::string runInference(llama_model* model, const std::string& system_prompt, 
                        const std::string& user_prompt, int max_tokens = 1024) {
    if (!model) {
        fprintf(stderr, "Error: Invalid model handle\n");
        return "";
    }

    // Construct the full prompt
    std::string full_prompt = "<|im_start|>system " + system_prompt + "<|im_end|>" +
                             "<|im_start|>user " + user_prompt + "<|im_end|>" +
                             "<|im_start|>assistant";

    // Tokenize the prompt
    const int n_prompt = -llama_tokenize(model, full_prompt.c_str(), full_prompt.size(), 
                                       nullptr, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    
    if (llama_tokenize(model, full_prompt.c_str(), full_prompt.size(), 
                      prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "Error: Failed to tokenize the prompt\n");
        return "";
    }

    // Initialize context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;  // load from model itself
    ctx_params.no_perf = false;
    ctx_params.flash_attn = true;

    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Error: Failed to create the llama_context\n");
        return "";
    }

    // Initialize sampler
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // Prepare initial batch
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // Generate response
    std::string generated_text;
    int n_decode = 0;
    llama_token new_token_id;
    int max = n_prompt + max_tokens;

    for (int n_pos = 0; n_pos + batch.n_tokens < max;) {
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "Error: Failed to decode\n");
            llama_sampler_free(smpl);
            llama_free(ctx);
            return "";
        }

        n_pos += batch.n_tokens;

        // Sample next token
        new_token_id = llama_sampler_sample(smpl, ctx, -1);

        // Check for end of generation
        if (llama_token_is_eog(model, new_token_id)) {
            break;
        }

        // Convert token to text
        char buf[128];
        int n = llama_token_to_piece(model, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "Error: Failed to convert token to piece\n");
            llama_sampler_free(smpl);
            llama_free(ctx);
            return "";
        }

        // Append to generated text
        generated_text.append(buf, n);

        // Prepare next batch
        batch = llama_batch_get_one(&new_token_id, 1);
        n_decode += 1;
    }

    // Cleanup
    llama_sampler_free(smpl);
    llama_free(ctx);

    return generated_text;
}

void releaseModel(llama_model* model) {
    if (model) {
        llama_free_model(model);
    }
}

Napi::Value RunInference(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();

    if (info.Length() < 3 || !info[0].IsString() || !info[1].IsString() || !info[2].IsString())
    {
        Napi::TypeError::New(env, "Expected three string arguments").ThrowAsJavaScriptException();
        return env.Null();
    }

    std::string model_path = info[0].As<Napi::String>().Utf8Value();
    std::string system_prompt = info[1].As<Napi::String>().Utf8Value();
    std::string prompt = info[2].As<Napi::String>().Utf8Value();

    std::string response;

    llama_model* model = loadModel(model_path);
    if (model) {
        response = runInference(model, system_prompt, prompt);
        releaseModel(model);
    }

    return Napi::String::New(env, response);
}

// Module initialization
Napi::Object Init(Napi::Env env, Napi::Object exports)
{
    exports.Set("RunInference", Napi::Function::New(env, RunInference));
    return exports;
}

// Register your module
NODE_API_MODULE(npm_llama, Init)