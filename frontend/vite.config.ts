import fs from 'fs'
import path from 'path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

/**
 * 自動讀取後端實際使用的 port（由 backend/run.py 寫入 .port 檔）
 * 優先順序：環境變數 VITE_API_PORT > .port 檔 > 預設 8000
 */
function resolveBackendPort(): number {
  // 1. 環境變數優先
  if (process.env.VITE_API_PORT) {
    const p = Number(process.env.VITE_API_PORT)
    if (!isNaN(p)) return p
  }

  // 2. 讀取 backend/.port 檔
  const portFile = path.resolve(__dirname, '../backend/.port')
  try {
    const content = fs.readFileSync(portFile, 'utf-8').trim()
    const p = Number(content)
    if (!isNaN(p)) {
      console.log(`[vite] 從 backend/.port 讀取後端 port: ${p}`)
      return p
    }
  } catch {
    // .port 檔不存在，使用預設值
  }

  // 3. 預設
  return 8000
}

const backendPort = resolveBackendPort()

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: `http://localhost:${backendPort}`,
        changeOrigin: true,
        timeout: 0,        // 禁用代理超時（SSE 長連接需要）
        proxyTimeout: 0,   // 禁用代理回應超時
      },
    },
  },
})
