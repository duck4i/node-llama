const { RunInferenceAsync, LoadModelAsync, ReleaseModelAsync } = require('@duck4i/llama');

const infer = async () => {

    const system_prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.";
    const prompts = [
        "How old can ducks get?",
        "Why are ducks so cool?",
        "Is there a limit on number of ducks I can own?"
    ]

    const model = await LoadModelAsync("model.gguf");
    console.log("Model loaded\n", model);

    for (const prompt of prompts) {
        const inference = await RunInferenceAsync(model, system_prompt, prompt, /*optional*/ 1024);
        console.log("Inference\n", inference);
    }

    await ReleaseModelAsync(model);

}

infer();