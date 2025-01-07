const { RunInference, downloadModel } = require("@duck4i/llama");

downloadModel(
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf?download=true",
    "model.gguf"
).then(() => {

    const res = RunInference("model.gguf", "How many ducks can one own?", "You are a helpful AI assistant.");
    console.log(res);
})