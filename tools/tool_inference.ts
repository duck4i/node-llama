#!/usr/bin/env node

import { Command } from 'commander';
import { RunInference } from "../src";

// Define types for the package.json import
interface PackageInfo {
  version: string;
  [key: string]: any;
}

// Import package.json with type assertion
const packageInfo = require('../package.json') as PackageInfo;

const program = new Command();
const defaultSystem = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.";

program
  .version(packageInfo.version)
  .requiredOption('-m, --model <path>', 'Path to the model')
  .requiredOption('-p, --prompt <prompt>', 'User prompt')
  .option('-s, --system <prompt>', 'System prompt', defaultSystem);

program.parse(process.argv);

interface ProgramOptions {
  model: string;
  prompt: string;
  system: string;
}

const options = program.opts() as ProgramOptions;

console.log(`Model path: ${options.model}\nUser prompt: ${options.prompt}\nSystem prompt: ${options.system}\n`);

const inference = RunInference(options.model, options.prompt, options.system);

console.log(inference.trim());