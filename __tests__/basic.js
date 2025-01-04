const { execSync } = require('child_process');
const fs = require('fs');

const { ChatManager, Role } = require('../chatManager');
const {
    RunInference,
    LoadModelAsync,
    RunInferenceAsync,
    ReleaseModelAsync,
    SetLogLevel,
    GetModelToken,
} = require("bindings")("npm-llama");

const model = "model.gguf";
const modelUrl = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf?download=true";
const system_prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.";


describe('Node LLaMA Test Suite', () => {

    beforeAll(() => {

        if (!fs.existsSync(model)) {
            execSync(`npx llama-download -p ${model} -u ${modelUrl}`, { stdio: 'inherit' });
        } else {
            console.log("Model already downloaded");
        }
    });

    test('log level works', () => {
        SetLogLevel(1); // debug logs
        expect(true).toBeTruthy();
    });

    test('direct inference works', () => {
        const inference = RunInference(model, system_prompt, "How old can ducks get?");
        console.log("Result", inference);
        expect(inference).toBeTruthy();
    });

    test('async inference works', async () => {

        const prompts = [
            "How old can ducks get?",
            "Why are ducks so cool?",
            "Is there a limit on number of ducks I can own?"
        ]

        const modelHandle = await LoadModelAsync(model);
        console.log("Model loaded", model);

        for (const prompt of prompts) {
            const inference = await RunInferenceAsync(modelHandle, system_prompt, prompt, /*optional*/ 512);
            console.log("Inference", inference);
        }

        await ReleaseModelAsync(modelHandle);
    });

    test('custom inference works', async () => {

        const user = "How old can ducks live?";
        const prompt = `"!#<|im_start|>system ${system_prompt}<|im_end|><|im_start|>user ${user}<|im_end|><|im_start|>assistant"`;

        const modelHandle = await LoadModelAsync(model);
        const result = await RunInferenceAsync(modelHandle, prompt);
        await ReleaseModelAsync(modelHandle);

        console.log("Result", result);
        expect(true).toBeTruthy();
    });

    test('tokens work', async () => {

        const modelHandle = await LoadModelAsync(model);

        const eos = GetModelToken(modelHandle, "EOS");
        const bos = GetModelToken(modelHandle, "BOS");
        const eot = GetModelToken(modelHandle, "EOT");
        const sep = GetModelToken(modelHandle, "SEP");
        const cls = GetModelToken(modelHandle, "CLS");
        const nl = GetModelToken(modelHandle, "NL");

        console.log("EOS", eos);
        console.log("BOS", bos);
        console.log("EOT", eot);
        console.log("SEP", sep);
        console.log("CLS", cls);
        console.log("NL", nl);

        await ReleaseModelAsync(modelHandle);

        expect(eos.length > 1).toBeTruthy();
        expect(bos.length > 1).toBeTruthy();
        expect(eot.length > 1).toBeTruthy();
        expect(sep.length > 1).toBeTruthy();
    })

    test('chat test works', async () => {
        SetLogLevel(4); // warn

        const modelHandle = await LoadModelAsync(model);

        const chat = new ChatManager(system_prompt);

        let reply = "";
        let prompt = chat.getNextPrompt("Hello, my name is Duck!");

        reply = await RunInferenceAsync(modelHandle, prompt);
        console.log("Reply", reply);

        chat.addMessage(Role.ASSISTANT, reply);

        prompt = chat.getNextPrompt("Do you remember my name?");
        reply = await RunInferenceAsync(modelHandle, prompt);
        console.log("Reply", reply);

        chat.addMessage(Role.ASSISTANT, reply);

        await ReleaseModelAsync(modelHandle);

        expect(reply.includes("Duck")).toBeTruthy();
    });

});
