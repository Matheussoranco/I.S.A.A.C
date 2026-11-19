import React, { useState } from 'react'
import { Layout, Sidebar, Header, MainContent, CommandPalette, ProfileSwitcher } from './components'
import { useProfileStore, useCommandPaletteStore } from './stores'
import { IsaakLogo } from './components/icons'

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const { currentProfile, profiles, switchProfile } = useProfileStore()
  const { isOpen, toggle } = useCommandPaletteStore()

  return (
    <div className="app">
      <aside className={`sidebar ${sidebarOpen ? 'open' : 'collapsed'}`} role="navigation" aria-label="Main navigation">
        <IsaakLogo />
        <nav className="sidebar-nav">
          <ul>
            <li><a href="#chat" data-icon="message-square">Chat</a></li>
            <li><a href="#kanban" data-icon="kanban">Kanban</a></li>
            <li><a href="#models" data-icon="cpu">Models</a></li>
            <li><a href="#memory" data-icon="database">Memory</a></li>
            <li><a href="#settings" data-icon="settings">Settings</a></li>
          </ul>
        </nav>
        <ProfileSwitcher currentProfile={currentProfile} profiles={profiles} onSwitch={switchProfile} />
      </aside>

      <div className="main-wrapper">
        <Header onMenuClick={() => setSidebarOpen(!sidebarOpen)} profile={currentProfile} onCommandPalette={toggle} />
        <MainContent />
      </div>

      <CommandPalette isOpen={isOpen} onClose={toggle} />
    </div>
  )
}

export default App