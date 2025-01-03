#ifndef NPM_LLAMA_H
#define NPM_LLAMA_H

#include <napi.h>

Napi::Value RunInference(const Napi::CallbackInfo &info);

#endif