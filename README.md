# NODE-LLAMA

Run llama cpp locally inside your Node environment with ease of Typescript.

# Build Status

| OS      | Node 18 | Node 20 | Node 22 |
|---------|---------|---------|---------|
| Ubuntu  | [![Ubuntu Node 18](https://github.com/duck4i/node-llama/actions/workflows/build.yml/badge.svg?branch=main&jobName=build%20(ubuntu-latest%2C%2018.x))](https://github.com/duck4i/node-llama/actions/workflows/build.yml) | [![Ubuntu Node 20](https://github.com/duck4i/node-llama/actions/workflows/build.yml/badge.svg?branch=main&jobName=build%20(ubuntu-latest%2C%2020.x))](https://github.com/duck4i/node-llama/actions/workflows/build.yml) | [![Ubuntu Node 22](https://github.com/duck4i/node-llama/actions/workflows/build.yml/badge.svg?branch=main&jobName=build%20(ubuntu-latest%2C%2022.x))](https://github.com/duck4i/node-llama/actions/workflows/build.yml) |
| macOS   | [![macOS Node 18](https://github.com/duck4i/node-llama/actions/workflows/build.yml/badge.svg?branch=main&jobName=build%20(macos-latest%2C%2018.x))](https://github.com/duck4i/node-llama/actions/workflows/build.yml) | [![macOS Node 20](https://github.com/duck4i/node-llama/actions/workflows/build.yml/badge.svg?branch=main&jobName=build%20(macos-latest%2C%2020.x))](https://github.com/duck4i/node-llama/actions/workflows/build.yml) | [![macOS Node 22](https://github.com/duck4i/node-llama/actions/workflows/build.yml/badge.svg?branch=main&jobName=build%20(macos-latest%2C%2022.x))](https://github.com/duck4i/node-llama/actions/workflows/build.yml) |
| Windows | [![Windows Node 18](https://github.com/duck4i/node-llama/actions/workflows/build.yml/badge.svg?branch=main&jobName=build%20(windows-latest%2C%2018.x))](https://github.com/duck4i/node-llama/actions/workflows/build.yml) | [![Windows Node 20](https://github.com/duck4i/node-llama/actions/workflows/build.yml/badge.svg?branch=main&jobName=build%20(windows-latest%2C%2020.x))](https://github.com/duck4i/node-llama/actions/workflows/build.yml) | [![Windows Node 22](https://github.com/duck4i/node-llama/actions/workflows/build.yml/badge.svg?branch=main&jobName=build%20(windows-latest%2C%2022.x))](https://github.com/duck4i/node-llama/actions/workflows/build.yml) |

# Package Info
[![npm version](https://badge.fury.io/js/@duck4i%2Fllama.svg)](https://badge.fury.io/js/@duck4i%2Fllama)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Node Version](https://img.shields.io/node/v/@duck4i/llama)](https://www.npmjs.com/package/@duck4i/llama)

## Reasoning 

Sometimes you just need a **small model** that can run anywhere and can't be bothered with making REST calls to services like OpenRouter or Ollama. 
This project is super simple, NodeJS native inference based on `llamacpp` project and with no need for external services.

Install NPM, download a model, and run it. Simple as.

## Features
 
- Minimal dependencies (mostly CMake and GCC) and no need for external services
- High performance, full speed of `llamacpp` with a thin layer of Node
- Multithreading support
- Streaming support
- Supports most LLM models
- Easy to use API
- Command line for direct inference and model download

## Installation

```sh
npm install @duck4i/llama
```

Please note that you need CMake and GCC installed if you don't have it already, as the plugin is cpp based.

```sh
sudo apt-get install -y build-essential cmake g++
```

## Bun 

If you are using BUN to install your packages you will have to add llama to list of trusted dependencies to your `package.json` to ensure the CMake compile pass is executed during the install.

Simply add this:
```javascript
  "trustedDependencies": [
    "@duck4i/llama"
  ],
```

## Usage

```javascript

import { RunInference, LLAMA_DEFAULT_SEED } = from "@duck4i/llama";

const system_prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.";
const user_prompt = "What is life expectancy of a duck?";

const inference =  RunInference({
    modelPath: modelPath,
    prompt: user_prompt,
    systemPrompt: system_prompt,
    seed: LLAMA_DEFAULT_SEED,       /*optional*/
    threads: 1,                     /*optional*/
    nCtx: 0,                        /*optional*/
    flashAttention: true,           /*optional*/
    onStream: (text: string, done: boolean) => {}  /*optional*/
});

console.log("Answer", inference);

```

It is likely you will want async functions for better memory management with multiple prompts, which is done like this:

```javascript
import { LoadModelAsync, CreateContextAsync, RunInferenceAsync, ReleaseContextAsync, ReleaseModelAsync } = from "@duck4i/llama";

const system_prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.";
const prompts = [
    "How old can ducks get?",
    "Why are ducks so cool?",
    "Is there a limit on number of ducks I can own?"
]

const model = await LoadModelAsync("model.gguf");
const ctx = await CreateContextAsync({
    model: modelHandle,
    threads: 4,             /*optional*/
    nCtx: 0,                /*optional*/
    flashAttention: true,   /*optional*/
});

console.log("Model loaded", model);

for (const prompt of prompts) {
    const inference = await RunInferenceAsync({
        model: modelHandle,
        context: ctx,
        prompt: "How old can ducks get?",
        systemPrompt: systemPrompt,
        maxTokens: 128,             /*optional*/
        seed: LLAMA_DEFAULT_SEED    /*optional*/
        onStream: (text: string, done: boolean) => {}  /*optional*/
    });
    console.log("Answer:", inference);
}

await ReleaseContextAsync(model);
await ReleaseModelAsync(model);

```

### Model format

The package is designed to handle most of LLaMA models, but its likely you will want more control over the model, so you can push the complete formatted prompt to it with prefix `!#`, like this:

```javascript

const system = "You are ...";
const user = "...";

//  QWEN / LLAMA example (prefix !# will get removed before reaching the llm)
const prompt = `"!#<|im_start|>system ${system}<|im_end|><|im_start|>user ${user}<|im_end|><|im_start|>assistant"`;

const reply = await RunInferenceAsync({
    prompt: prompt,
    ...
})

```

Please note that once you provide full prompt with a `!#` prefix, the system prompt will have no affect after that.

### Token fetch 

Getting tokens from model is done by `GetModelToken` method.

```javascript

const eos = GetModelToken(modelHandle, "EOS");
const bos = GetModelToken(modelHandle, "BOS");
const eot = GetModelToken(modelHandle, "EOT");
const sep = GetModelToken(modelHandle, "SEP");
const cls = GetModelToken(modelHandle, "CLS");
const nl = GetModelToken(modelHandle, "NL");

```

### Logging control

You can control log levels coming from llamacpp like this:

```javascript

import { SetLogLevel } = from '@duck4i/llama';

SetLogLevel(LogLevel.Info);

```

## Command line 

```bash

# Download model
npx llama-download -u https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf?download=true -p model.gguf

# Run inference
npx llama-run -m model.gguf -p "How old can ducks get?"

# Run with system prompt, seed and threads
npx llama-run -m model.gguf -p "How old can ducks get?" -s "[System prompt...]" -d [seed] -t [threads]

```

## Supported Models

All models supported by `llamacpp` natively are supported here too, so do check their [repository](https://github.com/ggerganov/llama.cpp).

Please keep in mind that CUDA is not enabled yet due to complex dependencies so keep the model size in check.

On MacOS, the Metal backend should come included.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
