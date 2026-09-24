// The browser every test uses. By default: installed Chrome, headed, on the real GPU, so WebGPU
// and WebGL behave as they do for visitors (a software rasteriser hides real driver behaviour).
// SOFTWARE_GPU=1 falls back to headless SwiftShader, for machines without a GPU.
// CHROME_PATH picks a specific Chrome binary.
import pw from 'playwright';
import { pathToFileURL } from 'node:url';

export const root = pathToFileURL(process.cwd()).href;   // file:///C:/... on Windows, file:///home/... elsewhere

export function launch() {
  if (process.env.SOFTWARE_GPU) {
    return pw.chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-features=Vulkan',
      '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  }
  return pw.chromium.launch({ headless: false, ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : { channel: 'chrome' }) });
}
