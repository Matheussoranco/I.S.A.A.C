import { create } from 'zustand'

interface CommandPaletteState {
  isOpen: boolean
  query: string
  setQuery: (query: string) => void
  open: () => void
  close: () => void
  toggle: () => void
}

export const useCommandPaletteStore = create<CommandPaletteState>((set) => ({
  isOpen: false,
  query: '',
  setQuery: (query) => set({ query }),
  open: () => set({ isOpen: true, query: '' }),
  close: () => set({ isOpen: false, query: '' }),
  toggle: () => set((state) => ({ isOpen: !state.isOpen, query: '' })),
}))