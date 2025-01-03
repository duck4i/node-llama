#include "npm-llama.h"

Napi::Value RunInference(const Napi::CallbackInfo &info)
{
    return Napi::String::New(info.Env(), "Hello, World!");
}

// Module initialization
Napi::Object Init(Napi::Env env, Napi::Object exports)
{
    exports.Set("RunInference", Napi::Function::New(env, RunInference));
    return exports;
}

// Register your module
NODE_API_MODULE(npm_llama, Init) // Make sure this line is present!