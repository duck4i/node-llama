// Define types for the package.json import
interface PackageInfo {
  version: string;
  [key: string]: any;
}

// Import package.json with type assertion
const packageInfo = require('../package.json') as PackageInfo;

export const version = packageInfo.version;