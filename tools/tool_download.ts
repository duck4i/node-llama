#!/usr/bin/env node

import { Command } from 'commander';
import { downloadModel } from '../src';
import { version } from './version';

const program = new Command();
program
    .version(version)
    .requiredOption('-u, --url <url>', 'Download URL')
    .requiredOption('-p, --path <prompt>', 'Output path');

program.parse(process.argv);
const options = program.opts() as {
    url: string;
    path: string;
};

const url = `${options.url}`;
const target = `${options.path}`;

console.log(`Downloading from ${url} to ${target}`);

// Run the download
downloadModel(url, target)
    .then(() => {
        console.log('Download completed successfully');
        process.exit(0);
    })
    .catch((error: Error) => {
        console.error('Download failed:', error.message);
        process.exit(1);
    });