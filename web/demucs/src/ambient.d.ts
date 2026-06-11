/// <reference types="@webgpu/types" />

// fft.js ships no type declarations; the audio processor uses it as a plain
// constructor. Declared here so the standalone `tsc` build type-checks.
declare module 'fft.js';
