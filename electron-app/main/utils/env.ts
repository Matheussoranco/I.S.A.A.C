export const isDev = process.env.NODE_ENV === 'development' || process.env.VITE_DEV === 'true'

export const apiBaseUrl = isDev ? 'http://localhost:8080' : 'http://localhost:8080'