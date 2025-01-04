const Role = Object.freeze({
    SYSTEM: 'system',
    USER: 'user',
    ASSISTANT: 'assistant'
});

class ChatManager {
    constructor(systemPrompt = '') {
        this.systemPrompt = systemPrompt;
        this.history = [];
        this.delimiter = {
            start: '<|im_start|>',
            end: '<|im_end|>'
        };
    }

    addMessage(role, content) {
        if (!Object.values(Role).includes(role)) {
            throw new Error(`Invalid role. Must be one of: ${Object.values(Role).join(', ')}`);
        }
        this.history.push({ role, content });
    }

    clear() {
        this.history = [];
    }

    setSystemPrompt(prompt) {
        this.systemPrompt = prompt;
    }

    getNextPrompt(userPrompt) {
        let formatted = `!#${this.delimiter.start}${Role.SYSTEM} ${this.systemPrompt}${this.delimiter.end}`;

        if (userPrompt !== undefined) {
            this.addMessage(Role.USER, userPrompt);
        }

        for (const message of this.history) {
            formatted += `${this.delimiter.start}${message.role} ${message.content}${this.delimiter.end}`;
        }

        // Add the assistant delimiter for the next response
        formatted += `${this.delimiter.start}${Role.ASSISTANT}`;
        return formatted;
    }

    getHistory() {
        return this.history;
    }
}

// Example usage:
/*
const chat = new ChatManager("You are a helpful AI assistant.");

// Add user message
chat.addMessage(Role.USER, "Hello!");

// Get prompt for LLM
const prompt = chat.getNextPrompt();

// After getting LLM response, add it to history
chat.addMessage(Role.ASSISTANT, "Hi there! How can I help you today?");

*/

module.exports = {
    Role,
    ChatManager
}