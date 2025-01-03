const { Command } = require('commander');
const packageInfo = require('./package.json');
const { RunInference } = require("bindings")("npm-llama");

const program = new Command();
const defaultSystem = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.";

program
    .version(packageInfo.version)
    .requiredOption('-m, --model <path>', 'Path to the model')
    .requiredOption('-p, --prompt <prompt>', 'User prompt')
    .option('-s, --system <prompt>', 'System prompt', defaultSystem);

program.parse(process.argv);

const options = program.opts();

console.log(`Model path: ${options.model}\nUser prompt: ${options.prompt}\nSystem prompt: ${options.system}\n`);

const inference = RunInference(`${options.model}`, `${options.system}`, `${options.prompt}`);

console.log(inference);
