import { create } from 'zustand'

interface Profile {
  id: string
  name: string
  avatar?: string
  color: string
  config: Record<string, unknown>
}

interface ProfileState {
  profiles: Profile[]
  currentProfile: Profile | null
  setProfiles: (profiles: Profile[]) => void
  setCurrentProfile: (profile: Profile) => void
  switchProfile: (profileId: string) => void
  createProfile: (name: string) => void
  deleteProfile: (profileId: string) => void
}

const defaultProfiles: Profile[] = [
  {
    id: 'default',
    name: 'Default',
    color: '#58a6ff',
    config: { model: 'good', tools: ['web', 'shell', 'fs'] },
  },
  {
    id: 'research',
    name: 'Research',
    color: '#a371f7',
    config: { model: 'better', tools: ['web', 'browser', 'code'] },
  },
  {
    id: 'coding',
    name: 'Coding',
    color: '#3fb950',
    config: { model: 'best', tools: ['code', 'fs', 'shell'] },
  },
]

export const useProfileStore = create<ProfileState>((set, get) => ({
  profiles: defaultProfiles,
  currentProfile: defaultProfiles[0],
  setProfiles: (profiles) => set({ profiles }),
  setCurrentProfile: (profile) => set({ currentProfile: profile }),
  switchProfile: (profileId) => {
    const profile = get().profiles.find((p) => p.id === profileId)
    if (profile) set({ currentProfile: profile })
  },
  createProfile: (name) => {
    const id = name.toLowerCase().replace(/\s+/g, '-')
    const newProfile = {
      id,
      name,
      color: '#58a6ff',
      config: { model: 'good', tools: ['web'] },
    }
    set((state) => ({ profiles: [...state.profiles, newProfile] }))
  },
  deleteProfile: (profileId) => {
    set((state) => ({
      profiles: state.profiles.filter((p) => p.id !== profileId),
      currentProfile: state.currentProfile?.id === profileId ? defaultProfiles[0] : state.currentProfile,
    }))
  },
}))