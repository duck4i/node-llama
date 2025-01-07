enum Role {
    SYSTEM = 'system',
    USER = 'user',
    ASSISTANT = 'assistant'
}

interface Message {
    role: Role;
    content: string;
}

interface Delimiter {
    start: string;
    end: string;
}

class ChatManager {
    private systemPrompt: string;
    private history: Message[];
    private readonly delimiter: Delimiter;

    constructor(systemPrompt: string = '') {
        this.systemPrompt = systemPrompt;
        this.history = [];
        this.delimiter = {
            start: '<|im_start|>',
            end: '<|im_end|>'
        };
    }

    public addMessage(role: Role, content: string): void {
        if (!Object.values(Role).includes(role)) {
            throw new Error(`Invalid role. Must be one of: ${Object.values(Role).join(', ')}`);
        }
        this.history.push({ role, content });
    }

    public clear(): void {
        this.history = [];
    }

    public setSystemPrompt(prompt: string): void {
        this.systemPrompt = prompt;
    }

    public getNextPrompt(userPrompt?: string): string {
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

    public getHistory(): Message[] {
        return this.history;
    }
}

export { ChatManager, Role, type Message, type Delimiter };