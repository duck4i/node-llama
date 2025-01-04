const { ChatManager } = require('./chatManager');
const { downloadModel } = require("./downloadModel");

import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const nativeModule = require('./build/Release/npm-llama.node');

export const SetLogLevel = nativeModule.SetLogLevel;
export const RunInference = nativeModule.RunInference;
export const LoadModelAsync = nativeModule.LoadModelAsync;
export const RunInferenceAsync = nativeModule.RunInferenceAsync;
export const ReleaseModelAsync = nativeModule.ReleaseModelAsync;
export const GetModelToken = nativeModule.GetModelToken;

export { ChatManager };
export { downloadModel };