const { RunInference, SetLogLevel, downloadModel } = require('@duck4i/llama');

const download = async () => {
    await downloadModel(
        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf?download=true",
        "model.gguf");
}

download().then(() => {
    console.log("Downloaded model");

    // 0 - none, 1 - debug, 2 - info, 3 - warn, 4 - error
    SetLogLevel(1); // enable debug

    const system_prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.";
    const user_prompt = "What is human life expectancy of a duck?";

    const inference = RunInference("model.gguf", user_prompt, system_prompt);

    console.log("Inference", inference);
});
