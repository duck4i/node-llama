#include <napi.h>
#include "llama-cpp.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// INFERENCE
////////////////////////////////////////////////////////////////////////////////////////////////////

int g_logLevel = GGML_LOG_LEVEL_WARN;

void log(ggml_log_level level, const char *text, void *data)
{
    if ((level >= g_logLevel && level != GGML_LOG_LEVEL_CONT) && text != nullptr)
        printf("%s", text);
}

llama_model *loadModel(const std::string &model_path)
{
    ggml_backend_load_all();
    llama_log_set(log, nullptr);

    llama_model_params model_params = llama_model_default_params();

    llama_model *model = llama_load_model_from_file(model_path.c_str(), model_params);

    if (model == nullptr)
    {
        fprintf(stderr, "Error: Unable to load model from %s\n", model_path.c_str());
        return nullptr;
    }

    return model;
}

llama_context *createContext(llama_model *model, int n_thread = 1, int n_ctx = 0, bool flash_attn = true)
{
    if (!model)
    {
        fprintf(stderr, "Error: Invalid model handle\n");
        return nullptr;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx; // 0 means load from model
    ctx_params.no_perf = true;
    ctx_params.flash_attn = flash_attn;
    ctx_params.n_threads = n_thread;

    llama_context *ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx)
    {
        fprintf(stderr, "Error: Failed to create the llama_context\n");
        return nullptr;
    }

    return ctx;
}

std::string runInference(llama_model *model, llama_context *ctx, const std::string &system_prompt,
                         const std::string &user_prompt, int max_tokens = 1024, size_t seed = LLAMA_DEFAULT_SEED)
{
    if (!model || !ctx)
    {
        fprintf(stderr, "Error: Invalid model or context handle\n");
        return "";
    }

    // Rest of existing runInference code, but remove context creation/cleanup
    bool isFullPrompt = system_prompt.size() > 2 && system_prompt[0] == '!' && system_prompt[1] == '#';

    std::string llama_format_prompt = "<|im_start|>system " + system_prompt + "<|im_end|>" +
                                      "<|im_start|>user " + user_prompt + "<|im_end|>" +
                                      "<|im_start|>assistant";

    std::string full_prompt = isFullPrompt ? user_prompt.substr(2) : llama_format_prompt;

    const int n_prompt = -llama_tokenize(model, full_prompt.c_str(), full_prompt.size(),
                                         nullptr, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);

    if (llama_tokenize(model, full_prompt.c_str(), full_prompt.size(),
                       prompt_tokens.data(), prompt_tokens.size(), true, true) < 0)
    {
        fprintf(stderr, "Error: Failed to tokenize the prompt\n");
        return "";
    }

    // Initialize sampler
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;

    //  Decide on a mode of the sampler - greedy is deterministic and consistent, distributed is more creative
    llama_sampler *smpl = llama_sampler_chain_init(sparams);
    seed == LLAMA_DEFAULT_SEED ? llama_sampler_chain_add(smpl, llama_sampler_init_greedy()) : llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));

    // Prepare initial batch
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // Generate response
    std::string generated_text;
    int n_decode = 0;
    llama_token new_token_id;
    int max = n_prompt + max_tokens;

    for (int n_pos = 0; n_pos + batch.n_tokens < max;)
    {
        if (llama_decode(ctx, batch))
        {
            fprintf(stderr, "Error: Failed to decode\n");
            llama_sampler_free(smpl);
            llama_free(ctx);
            return "";
        }

        n_pos += batch.n_tokens;

        // Sample next token
        new_token_id = llama_sampler_sample(smpl, ctx, -1);

        // Check for end of generation
        if (llama_token_is_eog(model, new_token_id))
        {
            break;
        }

        // Convert token to text
        char buf[128];
        int n = llama_token_to_piece(model, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0)
        {
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

    return generated_text;
}

void releaseContext(llama_context *ctx)
{
    if (ctx)
    {
        llama_free(ctx);
    }
}

void releaseModel(llama_model *model)
{
    if (model)
    {
        llama_free_model(model);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// SYNC
////////////////////////////////////////////////////////////////////////////////////////////////////

Napi::Value SetLogLevel(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsNumber())
    {
        Napi::TypeError::New(env, "Expected a number").ThrowAsJavaScriptException();
        return env.Null();
    }

    int log_level = info[0].As<Napi::Number>().Int32Value();
    g_logLevel = log_level;

    return env.Undefined();
}

struct RunInferenceOptions
{
    std::string modelPath;
    std::string prompt;
    std::string systemPrompt;
    int maxTokens = 1024;
    int threads = 1;
    size_t seed = LLAMA_DEFAULT_SEED;
    int nCtx = 0;
    bool flashAttention = true;
};

RunInferenceOptions ParseRunInferenceOptions(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsObject())
    {
        Napi::TypeError::New(env, "Expected an options object").ThrowAsJavaScriptException();
        return {};
    }

    Napi::Object optionsObj = info[0].As<Napi::Object>();
    RunInferenceOptions options;

    if (optionsObj.Has("modelPath") && optionsObj.Get("modelPath").IsString())
    {
        options.modelPath = optionsObj.Get("modelPath").As<Napi::String>().Utf8Value();
    }
    else
    {
        Napi::TypeError::New(env, "modelPath is required and should be a string").ThrowAsJavaScriptException();
        return {};
    }

    if (optionsObj.Has("prompt") && optionsObj.Get("prompt").IsString())
    {
        options.prompt = optionsObj.Get("prompt").As<Napi::String>().Utf8Value();
    }
    else
    {
        Napi::TypeError::New(env, "prompt is required and should be a string").ThrowAsJavaScriptException();
        return {};
    }

    if (optionsObj.Has("systemPrompt") && optionsObj.Get("systemPrompt").IsString())
    {
        options.systemPrompt = optionsObj.Get("systemPrompt").As<Napi::String>().Utf8Value();
    }

    if (optionsObj.Has("maxTokens") && optionsObj.Get("maxTokens").IsNumber())
    {
        options.maxTokens = optionsObj.Get("maxTokens").As<Napi::Number>().Int32Value();
    }

    if (optionsObj.Has("threads") && optionsObj.Get("threads").IsNumber())
    {
        options.threads = optionsObj.Get("threads").As<Napi::Number>().Int32Value();
    }

    if (optionsObj.Has("seed") && optionsObj.Get("seed").IsNumber())
    {
        options.seed = optionsObj.Get("seed").As<Napi::Number>().Uint32Value();
    }

    if (optionsObj.Has("nCtx") && optionsObj.Get("nCtx").IsNumber())
    {
        options.nCtx = optionsObj.Get("nCtx").As<Napi::Number>().Int32Value();
    }

    if (optionsObj.Has("flashAttention") && optionsObj.Get("flashAttention").IsBoolean())
    {
        options.flashAttention = optionsObj.Get("flashAttention").As<Napi::Boolean>().Value();
    }

    return options;
}

Napi::Value RunInference(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();
    RunInferenceOptions options = ParseRunInferenceOptions(info);

    std::string response;

    llama_model *model = loadModel(options.modelPath);
    if (model != nullptr)
    {
        llama_context *ctx = createContext(model, options.threads, options.nCtx, options.flashAttention);
        if (ctx != nullptr)
        {
            response = runInference(model, ctx, options.systemPrompt, options.prompt, options.maxTokens, options.seed);
            releaseContext(ctx);
        }
        releaseModel(model);
    }

    return Napi::String::New(env, response);
}

Napi::Value GetModelToken(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();

    if (info.Length() < 2 || !info[0].IsExternal() || !info[1].IsString())
    {
        Napi::TypeError::New(env, "Model handle expected").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    llama_model *model = info[0].As<Napi::External<llama_model>>().Data();
    if (!model)
    {
        Napi::TypeError::New(env, "Invalid model handle").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string token_name = info[1].As<Napi::String>().Utf8Value();
    if (token_name.length() < 1)
    {
        Napi::TypeError::New(env, "Token name expected").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    llama_token token = -1;

    if (token_name == "BOS")
    {
        token = llama_token_bos(model);
    }
    else if (token_name == "EOS")
    {
        token = llama_token_eos(model);
    }
    else if (token_name == "PAD")
    {
        token = llama_token_pad(model);
    }
    else if (token_name == "EOT")
    {
        token = llama_token_eot(model);
    }
    else if (token_name == "SEP")
    {
        token = llama_token_sep(model);
    }
    else if (token_name == "CLS")
    {
        token = llama_token_cls(model);
    }
    else if (token_name == "NL")
    {
        token = llama_token_nl(model);
    }
    else
    {
        Napi::TypeError::New(env, "Unknown token type").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    const char *tokenValue = token > 0 ? llama_token_get_text(model, token) : nullptr;
    return tokenValue == nullptr ? env.Undefined() : Napi::String::New(env, tokenValue);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// ASYNC
////////////////////////////////////////////////////////////////////////////////////////////////////

class LoadModelWorker : public Napi::AsyncWorker
{
public:
    LoadModelWorker(Napi::Env &env, const std::string &modelPath)
        : Napi::AsyncWorker(env), _modelPath(modelPath), _deferred(Napi::Promise::Deferred::New(env)) {}

    void Execute() override
    {
        _model = loadModel(_modelPath);

        if (_model == nullptr)
        {
            SetError("Failed to load model");
        }
    }

    void OnOK() override
    {
        Napi::Env env = _deferred.Env();

        // Wrap the model pointer in an external
        Napi::External<llama_model> modelExternal = Napi::External<llama_model>::New(env, _model);
        _deferred.Resolve(modelExternal);
    }

    void OnError(const Napi::Error &error) override
    {
        _deferred.Reject(error.Value());
    }

    Napi::Promise GetPromise() const
    {
        return _deferred.Promise();
    }

private:
    std::string _modelPath;
    llama_model *_model = nullptr;
    Napi::Promise::Deferred _deferred;
};

class CreateContextWorker : public Napi::AsyncWorker
{
public:
    CreateContextWorker(Napi::Env &env, llama_model *model, int n_threads, int n_ctx, bool flash_attn)
        : Napi::AsyncWorker(env), _model(model), _n_threads(n_threads), _n_ctx(n_ctx), _flash_attn(flash_attn),
          _deferred(Napi::Promise::Deferred::New(env)) {}

    void Execute() override
    {
        _context = createContext(_model, _n_threads, _n_ctx, _flash_attn);
        if (_context == nullptr)
        {
            SetError("Failed to create context");
        }
    }

    void OnOK() override
    {
        Napi::Env env = _deferred.Env();
        Napi::External<llama_context> contextExternal = Napi::External<llama_context>::New(env, _context);
        _deferred.Resolve(contextExternal);
    }

    void OnError(const Napi::Error &error) override
    {
        _deferred.Reject(error.Value());
    }

    Napi::Promise GetPromise() const
    {
        return _deferred.Promise();
    }

private:
    llama_model *_model;
    llama_context *_context;
    int _n_threads;
    int _n_ctx;
    bool _flash_attn;
    Napi::Promise::Deferred _deferred;
};

class InferenceWorker : public Napi::AsyncWorker
{
public:
    InferenceWorker(Napi::Env &env, llama_model *model, llama_context *context,
                    const std::string &systemPrompt,
                    const std::string &userPrompt,
                    int maxTokens, size_t seed)
        : Napi::AsyncWorker(env),
          _model(model),
          _context(context),
          _systemPrompt(systemPrompt),
          _userPrompt(userPrompt),
          _maxTokens(maxTokens),
          _seed(seed),
          _deferred(Napi::Promise::Deferred::New(env)) {}

    void Execute() override
    {
        if (!_model || !_context)
        {
            SetError("Invalid model or context handle");
            return;
        }

        _result = runInference(_model, _context, _systemPrompt, _userPrompt, _maxTokens, _seed);
        if (_result.empty())
        {
            SetError("Failed to run inference");
        }
    }

    void OnOK() override
    {
        Napi::Env env = _deferred.Env();
        _deferred.Resolve(Napi::String::New(env, _result));
    }

    void OnError(const Napi::Error &error) override
    {
        _deferred.Reject(error.Value());
    }

    Napi::Promise GetPromise() const
    {
        return _deferred.Promise();
    }

private:
    llama_model *_model;
    llama_context *_context;
    std::string _systemPrompt;
    std::string _userPrompt;
    int _maxTokens;
    size_t _seed;
    std::string _result;
    Napi::Promise::Deferred _deferred;
};

class ReleaseContextWorker : public Napi::AsyncWorker
{
public:
    ReleaseContextWorker(Napi::Env &env, llama_context *context)
        : Napi::AsyncWorker(env), _context(context), _deferred(Napi::Promise::Deferred::New(env)) {}

    void Execute() override
    {
        if (_context)
        {
            llama_free(_context);
        }
    }

    void OnOK() override
    {
        Napi::Env env = _deferred.Env();
        _deferred.Resolve(env.Undefined());
    }

    void OnError(const Napi::Error &error) override
    {
        _deferred.Reject(error.Value());
    }

    Napi::Promise GetPromise() const
    {
        return _deferred.Promise();
    }

private:
    llama_context *_context;
    Napi::Promise::Deferred _deferred;
};

class ReleaseModelWorker : public Napi::AsyncWorker
{
public:
    ReleaseModelWorker(Napi::Env &env, llama_model *model)
        : Napi::AsyncWorker(env), _model(model), _deferred(Napi::Promise::Deferred::New(env)) {}

    void Execute() override
    {
        if (_model)
        {
            llama_free_model(_model);
        }
    }

    void OnOK() override
    {
        Napi::Env env = _deferred.Env();
        _deferred.Resolve(env.Undefined());
    }

    void OnError(const Napi::Error &error) override
    {
        _deferred.Reject(error.Value());
    }

    Napi::Promise GetPromise() const
    {
        return _deferred.Promise();
    }

private:
    llama_model *_model;
    Napi::Promise::Deferred _deferred;
};

Napi::Value LoadModelAsync(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsString())
    {
        Napi::TypeError::New(env, "Model path expected").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string modelPath = info[0].As<Napi::String>().Utf8Value();

    LoadModelWorker *worker = new LoadModelWorker(env, modelPath);
    worker->Queue();

    return worker->GetPromise();
}

struct CreateContextOptions
{
    llama_model *model;
    int threads = 1;
    int nCtx = 0;
    bool flashAttention = true;
};

CreateContextOptions ParseCreateContextOptions(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsObject())
    {
        Napi::TypeError::New(env, "Expected an options object").ThrowAsJavaScriptException();
        return {};
    }

    Napi::Object optionsObj = info[0].As<Napi::Object>();
    CreateContextOptions options;

    if (optionsObj.Has("model") && optionsObj.Get("model").IsExternal())
    {
        options.model = optionsObj.Get("model").As<Napi::External<llama_model>>().Data();
    }
    else
    {
        Napi::TypeError::New(env, "model is required and should be an external").ThrowAsJavaScriptException();
        return {};
    }

    if (optionsObj.Has("threads") && optionsObj.Get("threads").IsNumber())
    {
        options.threads = optionsObj.Get("threads").As<Napi::Number>().Int32Value();
    }

    if (optionsObj.Has("nCtx") && optionsObj.Get("nCtx").IsNumber())
    {
        options.nCtx = optionsObj.Get("nCtx").As<Napi::Number>().Int32Value();
    }

    if (optionsObj.Has("flashAttention") && optionsObj.Get("flashAttention").IsBoolean())
    {
        options.flashAttention = optionsObj.Get("flashAttention").As<Napi::Boolean>().Value();
    }

    return options;
}

Napi::Value CreateContextAsync(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();
    CreateContextOptions options = ParseCreateContextOptions(info);

    if (options.model == nullptr)
    {
        Napi::TypeError::New(env, "Invalid model handle").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    CreateContextWorker *worker = new CreateContextWorker(env, options.model, options.threads, options.nCtx, options.flashAttention);
    worker->Queue();

    return worker->GetPromise();
}

struct RunInferenceAsyncOptions
{
    llama_model *model;
    llama_context *context;
    std::string prompt;
    std::string systemPrompt;
    int maxTokens = 1024;
    size_t seed = LLAMA_DEFAULT_SEED;
};

RunInferenceAsyncOptions ParseRunInferenceAsyncOptions(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsObject())
    {
        Napi::TypeError::New(env, "Expected an options object").ThrowAsJavaScriptException();
        return {};
    }

    Napi::Object optionsObj = info[0].As<Napi::Object>();
    RunInferenceAsyncOptions options;

    if (optionsObj.Has("model") && optionsObj.Get("model").IsExternal())
    {
        options.model = optionsObj.Get("model").As<Napi::External<llama_model>>().Data();
    }
    else
    {
        Napi::TypeError::New(env, "model is required and should be an external").ThrowAsJavaScriptException();
        return {};
    }

    if (optionsObj.Has("context") && optionsObj.Get("context").IsExternal())
    {
        options.context = optionsObj.Get("context").As<Napi::External<llama_context>>().Data();
    }
    else
    {
        Napi::TypeError::New(env, "context is required and should be an external").ThrowAsJavaScriptException();
        return {};
    }

    if (optionsObj.Has("prompt") && optionsObj.Get("prompt").IsString())
    {
        options.prompt = optionsObj.Get("prompt").As<Napi::String>().Utf8Value();
    }
    else
    {
        Napi::TypeError::New(env, "prompt is required and should be a string").ThrowAsJavaScriptException();
        return {};
    }

    if (optionsObj.Has("systemPrompt") && optionsObj.Get("systemPrompt").IsString())
    {
        options.systemPrompt = optionsObj.Get("systemPrompt").As<Napi::String>().Utf8Value();
    }

    if (optionsObj.Has("maxTokens") && optionsObj.Get("maxTokens").IsNumber())
    {
        options.maxTokens = optionsObj.Get("maxTokens").As<Napi::Number>().Int32Value();
    }

    if (optionsObj.Has("seed") && optionsObj.Get("seed").IsNumber())
    {
        options.seed = optionsObj.Get("seed").As<Napi::Number>().Uint32Value();
    }

    return options;
}

Napi::Value RunInferenceAsync(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();
    RunInferenceAsyncOptions options = ParseRunInferenceAsyncOptions(info);

    if (options.model == nullptr || options.context == nullptr || options.prompt.empty())
    {
        Napi::TypeError::New(env, "Invalid options object").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    InferenceWorker *worker = new InferenceWorker(env, options.model, options.context, options.systemPrompt, options.prompt, options.maxTokens, options.seed);
    worker->Queue();

    return worker->GetPromise();
}

Napi::Value ReleaseContextAsync(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsExternal())
    {
        Napi::TypeError::New(env, "Context handle expected").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    llama_context *context = info[0].As<Napi::External<llama_context>>().Data();

    ReleaseContextWorker *worker = new ReleaseContextWorker(env, context);
    worker->Queue();

    return worker->GetPromise();
}

Napi::Value ReleaseModelAsync(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsExternal())
    {
        Napi::TypeError::New(env, "Model handle expected").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    llama_model *model = info[0].As<Napi::External<llama_model>>().Data();

    ReleaseModelWorker *worker = new ReleaseModelWorker(env, model);
    worker->Queue();

    return worker->GetPromise();
}

// Module initialization
Napi::Object Init(Napi::Env env, Napi::Object exports)
{
    exports.Set("SetLogLevel", Napi::Function::New(env, SetLogLevel));
    exports.Set("GetModelToken", Napi::Function::New(env, GetModelToken));

    exports.Set("RunInference", Napi::Function::New(env, RunInference));

    exports.Set("LoadModelAsync", Napi::Function::New(env, LoadModelAsync));
    exports.Set("CreateContextAsync", Napi::Function::New(env, CreateContextAsync));
    exports.Set("RunInferenceAsync", Napi::Function::New(env, RunInferenceAsync));
    exports.Set("ReleaseContextAsync", Napi::Function::New(env, ReleaseContextAsync));
    exports.Set("ReleaseModelAsync", Napi::Function::New(env, ReleaseModelAsync));

    exports.Set("LLAMA_DEFAULT_SEED", static_cast<int>(LLAMA_DEFAULT_SEED));

    return exports;
}

// Register your module
NODE_API_MODULE(npm_llama, Init)