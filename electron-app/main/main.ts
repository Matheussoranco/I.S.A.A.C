import { app, BrowserWindow, ipcMain, shell } from 'electron'
import { join } from 'path'
import { isDev } from './utils/env'

let mainWindow: Electron.BrowserWindow | null = null

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    webPreferences: {
      preload: join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
    titleBarStyle: 'hidden',
    trafficLightPosition: { x: 16, y: 16 },
    show: false,
  })

  if (isDev) {
    mainWindow.loadURL('http://localhost:5173')
    mainWindow.webContents.openDevTools()
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }

  mainWindow.once('ready-to-show', () => {
    mainWindow?.show()
  })

  mainWindow.on('closed', () => {
    mainWindow = null
  })

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })
}

app.whenReady().then(() => {
  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

// IPC handlers for Python backend communication
ipcMain.handle('isaac:api', async (_event, { method, path, body }) => {
  // Forward to Python backend
  const response = await fetch(`http://localhost:8080${path}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  return response.json()
})

ipcMain.handle('isaac:config:get', async () => {
  const response = await fetch('http://localhost:8080/api/config')
  return response.json()
})

ipcMain.handle('isaac:models:list', async () => {
  const response = await fetch('http://localhost:8080/api/models')
  return response.json()
})

ipcMain.handle('isaac:kanban:boards', async () => {
  const response = await fetch('http://localhost:8080/api/kanban/boards')
  return response.json()
})