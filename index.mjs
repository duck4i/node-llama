import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const nativeModule = require('./build/Release/npm-llama.node');

export const RunInference = nativeModule.RunInference;
export const LoadModelAsync = nativeModule.LoadModelAsync;
export const RunInferenceAsync = nativeModule.RunInferenceAsync;
export const ReleaseModelAsync = nativeModule.ReleaseModelAsync;