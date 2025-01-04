const bindings = require("bindings")("npm-llama");
const { chatManager } = require("./chatManager");
const { downloadModel } = require("./downloadModel");

module.exports = {
    ...bindings,
    chatManager,
    downloadModel
}