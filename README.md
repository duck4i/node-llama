# NODE-LLAMA

Run llama cpp locally inside your Node environment. 

## Reasoning 

Sometimes you just need a **small model** that can run anywhere and can't be bothered with making REST calls to services like OpenRouter or Ollama. 
This project is super simple, NodeJS native inference based on `llamacpp` project and with no need for external services.

Install NPM, download a model, and run it. Simple as.

## Features
 
- Minimal dependencies (mostly CMake and GCC) and no need for external services
- High performance, full speed of `llamacpp` with a thin layer of Node
- Supports most LLM models
- Easy to use API
- Command line for direct inference and model download

## Installation

```sh
npm install @duck4i/llama
```

Please note that you need CMake and GCC installed if you don't have it already, as the plugin is cpp based.

```sh
sudo apt install cmake g++
```

## Usage

```javascript

const { RunInference } = require('@duck4i/llama');

const system_prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.";
const user_prompt = "What is life expectancy of a duck?";

const inference = RunInference("model.gguf", system_prompt, user_prompt);

console.log("Answer", inference);

```

It is likely you will want async functions for better memory management with multiple prompts, which is done like this:

```javascript
const { LoadModelAsync, RunInferenceAsync, ReleaseModelAsync } = require('@duck4i/llama');

const system_prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.";
const prompts = [
    "How old can ducks get?",
    "Why are ducks so cool?",
    "Is there a limit on number of ducks I can own?"
]

const model = await LoadModelAsync("model.gguf");
console.log("Model loaded\n", model);

for (const prompt of prompts) {
    const inference = await RunInferenceAsync(model, system_prompt, prompt, /*optional max tokens*/ 1024);
    console.log("Answer:\n", inference);
}

await ReleaseModelAsync(model);

```

You can control log levels coming from llamacpp like this:

```javascript

const { SetLogLevel } = require('@duck4i/llama');

// 0 - none, 1 - debug, 2 - info, 3 - warn, 4 - error
SetLogLevel(1);

```

## Command line 

```bash

# Download model
npx llama-download -u https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf?download=true -p model.gguf

# Run inference
npx llama-run -m model.gguf -p "How old can ducks get?"

# Run with system prompt
npx llama-run -m model.gguf -p "How old can ducks get?" -s "[System prompt...]"

```

## Supported Models

All models supported by `llamacpp` natively are supported here too, so do check their [repository](https://github.com/ggerganov/llama.cpp).

Please keep in mind that CUDA is not enabled yet due to complex dependencies so keep the model size in check.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
