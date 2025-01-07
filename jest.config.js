/** @type {import('ts-jest').JestConfigWithTsJest} **/
export default {
  testEnvironment: "node",
  transform: {
    "^.+.tsx?$": ["ts-jest",{}],
  },
  extensionsToTreatAsEsm: ['.ts'],
  testTimeout: 480000 // 480 seconds (8m)
};