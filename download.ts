import axios, { type AxiosProgressEvent } from 'axios';
import { WriteStream, createWriteStream, unlinkSync, existsSync } from 'fs';
import { resolve } from 'path';

interface DownloadProgress {
    loaded: number;
    total: number;
}

async function downloadModel(url: string, outputPath: string): Promise<void> {
    if (!url) {
        throw new Error('No URL provided');
    }

    if (!outputPath) {
        throw new Error('No output path provided');
    }

    const fullPath: string = resolve(process.cwd(), outputPath);

    try {
        const response = await axios({
            method: 'GET',
            url: url,
            responseType: 'stream',
            maxRedirects: 5,
            timeout: 30000,
            headers: {
                'User-Agent': 'Node.js Download Script'
            },
            onDownloadProgress: (progressEvent: AxiosProgressEvent) => {
                const progress = progressEvent as DownloadProgress;
                const percentage: number = Math.round((progress.loaded * 100) / progress.total);
                process.stdout.write(`Downloaded: ${percentage}%\r`);
            }
        });

        const totalSize: number = parseInt(response.headers['content-length'], 10);
        if (totalSize) {
            console.log(`Total file size: ${(totalSize / 1024 / 1024).toFixed(2)} MB`);
        }

        const writer: WriteStream = createWriteStream(fullPath);

        return new Promise<void>((resolve, reject) => {
            response.data.pipe(writer);

            writer.on('finish', () => {
                process.stdout.write('\n');
                console.log('File downloaded successfully!');
                writer.close();
                resolve();
            });

            writer.on('error', (err: Error) => {
                unlinkSync(fullPath);
                reject(err);
            });
        });
    } catch (error) {
        // Clean up the file if it exists
        if (existsSync(fullPath)) {
            unlinkSync(fullPath);
        }
        throw new Error(`Download failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
}

export {
    downloadModel
};