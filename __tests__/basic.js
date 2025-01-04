const { execSync } = require('child_process');
const fs = require('fs');

const { RunInference, LoadModelAsync, RunInferenceAsync, ReleaseModelAsync, SetLogLevel } = require("bindings")("npm-llama");

const model = "model.gguf";
const modelUrl = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf?download=true";
const system_prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.";

describe('Node LLaMA Test Suite', () => {

    beforeAll(() => {

        if (!fs.existsSync(model)) {
            execSync(`npx llama-download -p ${model} -u ${modelUrl}`, { stdio: 'inherit' });
        } else {
            console.log("Model already exists");
        }
    });

    test('log level works', () => {
        SetLogLevel(1); // debug logs
        expect(true).toBeTruthy();
    });

    test('direct inference works', () => {
        const inference = RunInference(model, system_prompt, "How old can ducks get?");
        expect(inference).toBeTruthy();
    });

    test('inference works with async', async () => {

        const prompts = [
            "How old can ducks get?",
            "Why are ducks so cool?",
            "Is there a limit on number of ducks I can own?"
        ]

        const modelHandle = await LoadModelAsync(model);
        console.log("Model loaded", model);

        for (const prompt of prompts) {
            const inference = await RunInferenceAsync(modelHandle, system_prompt, prompt, /*optional*/ 1024);
            console.log("Inference", inference);
        }

        await ReleaseModelAsync(modelHandle);
    });
});
