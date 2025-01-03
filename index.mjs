import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const nativeModule = require('./build/Release/npm-llama.node');

export const RunInference = nativeModule.RunInference;