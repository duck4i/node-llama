import { execSync } from 'child_process';
import { existsSync } from 'fs';
import assert from 'assert';

import { ChatManager, Role } from '../src/chat';
import {
    RunInference,
    LoadModelAsync,
    CreateContextAsync,
    RunInferenceAsync,
    ReleaseContextAsync,
    ReleaseModelAsync,
    SetLogLevel,
    GetModelToken,
} from '../src/index';


const model = "model.gguf";
const modelUrl = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf?download=true";
const systemPrompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.";

describe("Llama tests - basic", () => {

    beforeAll(() => {
        // Setup - Download model if needed
        if (!existsSync(model)) {
            execSync(`npx llama-download -p ${model} -u ${modelUrl}`, { stdio: 'inherit' });
        } else {
            console.log("Model already downloaded");
        }
    })

    test('log level works', async () => {
        SetLogLevel(1); // debug logs
        assert.ok(true);
    });

    test('direct inference works', async () => {
        const inference: string = RunInference(model, "How old can ducks get?", systemPrompt);
        console.log("Result", inference);
        assert.ok(inference.includes('10 years'));
    });

    test('direct inference with seed works', async () => {
        const inference: string = RunInference(model, "How old can ducks get?", systemPrompt, 64, 12345);
        console.log("Result", inference);
        assert.ok(inference.includes('ages of two and three'));
    });

    test('async inference works', async () => {
        const prompts: string[] = [
            "How old can ducks get?",
            "Why are ducks so cool?",
            "Is there a limit on number of ducks I can own?"
        ];

        const modelHandle = await LoadModelAsync(model);
        const ctx = await CreateContextAsync(modelHandle);
        console.log("Model loaded", model);

        for (const prompt of prompts) {
            const inference: string = await RunInferenceAsync(modelHandle, ctx, prompt, systemPrompt, 64);
            console.log("Reply:", inference);
            assert.ok(inference.length > 0);
        }

        await ReleaseContextAsync(ctx);
        await ReleaseModelAsync(modelHandle);
    });

    test('custom inference works', async () => {
        const user = "How old can ducks get?";
        const prompt = `"!#<|im_start|>system ${systemPrompt}<|im_end|><|im_start|>user ${user}<|im_end|><|im_start|>assistant"`;

        const modelHandle = await LoadModelAsync(model);
        const context = await CreateContextAsync(modelHandle);
        const result: string = await RunInferenceAsync(modelHandle, context, prompt);
        await ReleaseContextAsync(context);
        await ReleaseModelAsync(modelHandle);

        console.log("Result", result);
        assert.ok(result.length > 1);
    });


    test('tokens work', async () => {
        const modelHandle = await LoadModelAsync(model);
        const ctx = await CreateContextAsync(modelHandle);

        const eos: string = GetModelToken(modelHandle, "EOS");
        const bos: string = GetModelToken(modelHandle, "BOS");
        const eot: string = GetModelToken(modelHandle, "EOT");
        const sep: string = GetModelToken(modelHandle, "SEP");
        const cls: string = GetModelToken(modelHandle, "CLS");
        const nl: string = GetModelToken(modelHandle, "NL");

        console.log("EOS", eos);
        console.log("BOS", bos);
        console.log("EOT", eot);
        console.log("SEP", sep);
        console.log("CLS", cls);
        console.log("NL", nl);

        await ReleaseContextAsync(ctx);
        await ReleaseModelAsync(modelHandle);

        assert.ok(eos.length > 1);
        assert.ok(bos.length > 1);
    });

    test('chat works', async () => {
        SetLogLevel(4); // warn

        const modelHandle = await LoadModelAsync(model);
        const ctx = await CreateContextAsync(modelHandle);

        const chat = new ChatManager(systemPrompt);

        let reply = "";
        let prompt = chat.getNextPrompt("Hello, my name is Duck!");

        reply = await RunInferenceAsync(modelHandle, ctx, prompt, 128);
        console.log("Reply", reply);

        chat.addMessage(Role.ASSISTANT, reply);

        prompt = chat.getNextPrompt("What was my name?");
        reply = await RunInferenceAsync(modelHandle, ctx, prompt, 128);
        console.log("Reply", reply);

        chat.addMessage(Role.ASSISTANT, reply);

        await ReleaseContextAsync(ctx);
        await ReleaseModelAsync(modelHandle);

        assert.ok(reply.includes("Duck"));
    });
});
