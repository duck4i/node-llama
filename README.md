# NPM-LLAMA

Run llama models locally directly inside your node environment. 

## Reasoning 

Sometimes you just need a **small model** that can run on CPU and can't be bothered with external services like OpenRouter or Ollama. This project is super simple, NodeJS native inference based on `llamacpp` project and with no other dependencies.

Install NPM, download a model, and run it. Simple as.

## Features

- No external dependencies (beside GCC and CMake)
- High performance
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

console.log("Inference", inference);

```

It is likely you will want async functions for better memory management with multiple prompts, which is done like this:

```javascript

const system_prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.";
const prompts = [
    "How old can ducks get?",
    "Why are ducks so cool?",
    "Is there a limit on number of ducks I can own?"
]

const model = await LoadModelAsync("model.gguf");
console.log("Model loaded\n", model);

for (let prompt of prompts) {
    inference = await RunInferenceAsync(model, system_prompt, prompt, /*optional*/ 1024);
    console.log("Inference\n", inference);
}

await ReleaseModelAsync(model);

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

Please keep in mind that node will create a CPU based build so keep the model size in check.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.