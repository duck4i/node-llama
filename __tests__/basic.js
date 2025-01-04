const { RunInference, LoadModelAsync, RunInferenceAsync, ReleaseModelAsync, SetLogLevel } = require("bindings")("npm-llama");

const system_prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.";

test('log level works', () => {
    SetLogLevel(4);
    expect(true).toBeTruthy();
});

test('direct inference works', () => {
    const inference = RunInference("model.gguf", system_prompt, "How old can ducks get?");
    expect(inference).toBeTruthy();
});

test('inference works with async', async () => {

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

});