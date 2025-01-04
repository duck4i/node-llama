#ifndef NPM_LLAMA_H
#define NPM_LLAMA_H

#include <napi.h>

Napi::Value SetLogLevel(const Napi::CallbackInfo& info);

Napi::Value RunInference(const Napi::CallbackInfo &info);

Napi::Value LoadModelAsync(const Napi::CallbackInfo& info);
Napi::Value RunInferenceAsync(const Napi::CallbackInfo& info);
Napi::Value ReleaseModelAsync(const Napi::CallbackInfo& info);

#endif