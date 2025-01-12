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
    LLAMA_DEFAULT_SEED,
    type TokenName,
    LogLevel
} from '../src/index';

const modelPath = "model.gguf";
const modelUrl = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf?download=true";
const systemPrompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.";

describe("Llama tests - basic", () => {

    beforeAll(() => {
        // Setup - Download model if needed
        if (!existsSync(modelPath)) {
            execSync(`npx llama-download -p ${modelPath} -u ${modelUrl}`, { stdio: 'inherit' });
        } else {
            console.log("Model already downloaded");
        }
    })

    test('log level works', async () => {
        SetLogLevel(LogLevel.Info); // debug logs
        assert.ok(true);
    });

    test('direct inference works', async () => {

        const inference: string = RunInference({
            modelPath: modelPath,
            prompt: "How old can ducks get?",
            systemPrompt: systemPrompt,
            seed: LLAMA_DEFAULT_SEED,
        });

        console.log("Result", inference);
        assert.ok(inference.includes('10 years'));
    });

    test('direct inference with seed works', async () => {
        const inference: string = RunInference({
            modelPath: modelPath,
            prompt: "How old can ducks get?",
            systemPrompt: systemPrompt,
            seed: 12345,
        });
        console.log("Result", inference);
        assert.ok(inference.includes('ages of two and three'));
    });

    test('direct inference with multithread', async () => {
        const inference: string = RunInference({
            modelPath: modelPath,
            prompt: "How old can ducks get?",
            systemPrompt: systemPrompt,
            threads: 4,
            seed: 12345,
        });
        console.log("Result", inference);
        assert.ok(inference.includes('ages of two and three'));
    });

    test('async inference works', async () => {

        const modelHandle = await LoadModelAsync(modelPath);
        const ctx = await CreateContextAsync({
            model: modelHandle,
        });
        console.log("Model loaded", modelPath);

        const inference = await RunInferenceAsync({
            model: modelHandle,
            context: ctx,
            prompt: "How old can ducks get?",
            systemPrompt: systemPrompt,
            maxTokens: 128,
            seed: LLAMA_DEFAULT_SEED
        });

        console.log("Result", inference);
        assert.ok(inference.includes('10 years old'));
    });

    test('async inference with seed works', async () => {

        const modelHandle = await LoadModelAsync(modelPath);
        const ctx = await CreateContextAsync({
            model: modelHandle,
        });
        console.log("Model loaded", modelPath);

        const inference = await RunInferenceAsync({
            model: modelHandle,
            context: ctx,
            prompt: "How old can ducks get?",
            systemPrompt: systemPrompt,
            seed: 12345
        });

        console.log("Result", inference);
        assert.ok(inference.includes('ages of two and three'));
    });

    test('async inference with multiple requests works', async () => {
        const prompts: string[] = [
            "How old can ducks get?",
            "Why are ducks so cool?",
            "Is there a limit on number of ducks I can own?"
        ];

        const modelHandle = await LoadModelAsync(modelPath);
        const ctx = await CreateContextAsync({
            model: modelHandle,
        });
        console.log("Model loaded", modelPath);

        for (const prompt of prompts) {
            const inference: string = await RunInferenceAsync({
                model: modelHandle,
                context: ctx,
                prompt: prompt,
                systemPrompt: systemPrompt,
                maxTokens: 128,
            });
            console.log("Reply:", inference);
            assert.ok(inference.length > 0);
        }

        await ReleaseContextAsync(ctx);
        await ReleaseModelAsync(modelHandle);
    });

    /*
    test('custom inference works', async () => {
        const user = "How old can ducks get?";
        const prompt = `"!#<|im_start|>system ${systemPrompt}<|im_end|><|im_start|>user ${user}<|im_end|><|im_start|>assistant"`;

        const modelHandle = await LoadModelAsync(model);
        const context = await CreateContextAsync(modelHandle);
        const result: string = await RunInferenceAsync({
            model: modelHandle,
            context: context,
            prompt: prompt,
            systemPrompt: systemPrompt,
        });
        await ReleaseContextAsync(context);
        await ReleaseModelAsync(modelHandle);

        console.log("Result", result);
        assert.ok(result.length > 1);
    });
    */

    test('tokens work', async () => {
        const modelHandle = await LoadModelAsync(modelPath);
        const ctx = await CreateContextAsync({
            model: modelHandle,
        });

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

        const modelHandle = await LoadModelAsync(modelPath);
        const ctx = await CreateContextAsync({
            model: modelHandle,
        });

        const chat = new ChatManager(systemPrompt);

        let reply = "";
        let prompt = chat.getNextPrompt("Hello, my name is Duck!");

        reply = await RunInferenceAsync({
            model: modelHandle,
            context: ctx,
            prompt: prompt,
            systemPrompt: systemPrompt,
            maxTokens: 128,
        });

        console.log("Reply", reply);

        chat.addMessage(Role.ASSISTANT, reply);

        prompt = chat.getNextPrompt("What was my name?");
        reply = await RunInferenceAsync({
            model: modelHandle,
            context: ctx,
            prompt: prompt,
            systemPrompt: systemPrompt,
            maxTokens: 128,
        });
        console.log("Reply", reply);

        chat.addMessage(Role.ASSISTANT, reply);

        await ReleaseContextAsync(ctx);
        await ReleaseModelAsync(modelHandle);

        assert.ok(reply.includes("Duck"));
    });
    
});
