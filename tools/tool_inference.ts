#!/usr/bin/env node

import { Command } from 'commander';
import { LLAMA_DEFAULT_SEED, RunInference } from "../src";
import { version } from './version';

const program = new Command();
const defaultSystem = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.";

program
  .version(version)
  .requiredOption('-m, --model <path>', 'Path to the model')
  .requiredOption('-p, --prompt <prompt>', 'User prompt')
  .option('-s, --system <prompt>', 'System prompt', defaultSystem)
  .option('-t, --threads <number>', 'Number of threads', "4")
  .option('-d, --seed <number>', 'Seed', `${LLAMA_DEFAULT_SEED}`);

program.parse(process.argv);

interface ProgramOptions {
  model: string;
  prompt: string;
  system: string;
  threads: string;
  seed: string;
}

const options = program.opts() as ProgramOptions;

console.log(`Model path: ${options.model}\nUser prompt: ${options.prompt}\nSystem prompt: ${options.system}\n`);

RunInference({
  modelPath: options.model,
  prompt: options.prompt,
  systemPrompt: options.system,
  threads: parseInt(options.threads),
  seed: parseInt(options.seed),
  onStream: (text: string, done: boolean) => {
    process.stdout.write(text);
    if (done) {
      process.stdout.write("\n");
    }
  }
});
