
import { ChatManager } from "./chat";
import { downloadModel } from "./download";

const npmLlama = require("bindings")("npm-llama");

export { ChatManager, downloadModel };

export const {
    SetLogLevel,
    GetModelToken,
    RunInference,
    LoadModelAsync,
    CreateContextAsync,
    RunInferenceAsync,
    ReleaseContextAsync,
    ReleaseModelAsync
} = npmLlama;

export default {
    ...npmLlama,
    ChatManager,
    downloadModel
};