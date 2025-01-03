#include "npm-llama.h"

Napi::Value RunInference(const Napi::CallbackInfo &info)
{
    return Napi::String::New(info.Env(), "Hello, World!");
}