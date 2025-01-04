const { RunInference, SetLogLevel } = require('@duck4i/llama');

// 0 - none, 1 - debug, 2 - info, 3 - warn, 4 - error
SetLogLevel(1); // enable debug

const system_prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.";
const user_prompt = "What is human life expectancy of a duck?";

const inference = RunInference("model.gguf", system_prompt, user_prompt);

console.log("Inference", inference);
