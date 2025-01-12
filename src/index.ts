
import { ChatManager, Role, type Delimiter, type Message } from "./chat";
import { downloadModel } from "./download";

const npmLlama = require("bindings")("npm-llama");

//  Sync functions declarations

export const LLAMA_DEFAULT_SEED : number = npmLlama.LLAMA_DEFAULT_SEED;

export enum LogLevel {
    None = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
    Continue = 5
}

export const SetLogLevel = (level: LogLevel): void => {
    npmLlama.SetLogLevel(level);
}

export interface RunInferenceOptions {
    modelPath: string;
    prompt: string;
    systemPrompt: string;
    maxTokens?: number;
    threads?: number;
    seed?: number;
    nCtx?: number;
    flashAttention?: boolean;
    onStream?: (text: string, done: boolean) => void;
}

export const RunInference = (options: RunInferenceOptions): string => {
    return npmLlama.RunInference(options);
}

export type TokenName = "BOS" | "EOS" | "PAD" | "EOT" | "SEP" | "CLS" | "NL";

export const GetModelToken = (model: any, token: TokenName): string => {
    return npmLlama.GetModelToken(model, token);
}

//  Async functions

export const LoadModelAsync = async (modelPath: string): Promise<any> => {
    return npmLlama.LoadModelAsync(modelPath);
}

export interface CreateContextOptions {
    model: any;
    threads?: number;
    nCtx?: number;
    flashAttention?: boolean;
}

export const CreateContextAsync = async (options: CreateContextOptions): Promise<any> => {
    return npmLlama.CreateContextAsync(options);
}

export interface RunInferenceAsyncOptions {
    model: any;
    context: any;
    prompt: string;
    systemPrompt: string;
    maxTokens?: number;
    seed?: number;
    onStream?: (text: string, done: boolean) => void;
}

export const RunInferenceAsync = async (options: RunInferenceAsyncOptions): Promise<string> => {
    return npmLlama.RunInferenceAsync(options);
}

export const ReleaseContextAsync = async (context: any): Promise<void> => {
    return npmLlama.ReleaseContextAsync(context);
}

export const ReleaseModelAsync = async (model: any): Promise<void> => {
    return npmLlama.ReleaseModelAsync(model);
}

//  Utility functions
export { ChatManager, downloadModel };

export default {
    ChatManager,
    Role,
    downloadModel
};