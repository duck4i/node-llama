#!/usr/bin/env node

const { Command } = require('commander');
const packageInfo = require('./package.json');
const { downloadModel } = require('./downloadModel');

const program = new Command();
program
    .version(packageInfo.version)
    .requiredOption('-u, --url <url>', 'Download URL')
    .requiredOption('-p, --path <prompt>', 'Output path');

program.parse(process.argv);
const options = program.opts();

const url = `${options.url}`;
const target = `${options.path}`;

console.log(`Downloading from ${url} to ${target}`);

// Run the download
downloadModel(url, target)
    .then(() => {
        console.log('Download completed successfully');
        process.exit(0);
    })
    .catch((error) => {
        console.error('Download failed:', error.message);
        process.exit(1);
    });