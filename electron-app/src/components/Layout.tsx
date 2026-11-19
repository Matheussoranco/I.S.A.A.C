import React from 'react'
import { IsaakLogo, MenuIcon, ChevronLeftIcon, SearchIcon, SettingsIcon, UserIcon, ChevronDownIcon, XIcon, CommandIcon } from '../components/icons'
import { useProfileStore } from '../stores/profileStore'
import { useCommandPaletteStore } from '../stores/commandPaletteStore'

export const IsaakLogoIcon = () => <IsaakLogo size={28} />

export const Header: React.FC<{ onMenuClick: () => void; profile: any; onCommandPalette: () => void }> = ({
  onMenuClick,
  profile,
  onCommandPalette,
}) => (
  <header className="header">
    <button className="header-menu-btn" onClick={onMenuClick} aria-label="Toggle sidebar">
      <MenuIcon size={24} />
    </button>
    <div className="header-search">
      <SearchIcon size={18} />
      <input
        type="text"
        placeholder="Search (⌘K)..."
        readOnly
        onClick={() => window.electronAPI?.openCommandPalette?.()}
      />
    </div>
    <div className="header-actions">
      <button className="header-btn" onClick={onCommandPalette} aria-label="Command Palette (⌘K)">
        <CommandIcon size={20} />
      </button>
      <div className="header-profile">
        <div className="profile-avatar" style={{ backgroundColor: profile?.color }}>
          {profile?.name?.[0]?.toUpperCase()}
        </div>
        <ChevronDownIcon size={16} />
      </div>
    </div>
  </header>
)

export const Sidebar: React.FC<{ isOpen: boolean; onClose: () => void }> = ({
  isOpen,
  onClose,
}) => {
  const { profiles, currentProfile, switchProfile } = useProfileStore()

  return (
    <aside className={`sidebar ${isOpen ? 'open' : 'collapsed'}`}>
      <div className="sidebar-header">
        <IsaakLogo size={32} />
        <span className="sidebar-title">I.S.A.A.C.</span>
        {!isOpen && <button className="sidebar-toggle" onClick={onClose}><ChevronLeftIcon size={20} /></button>}
      </div>
      <nav className="sidebar-nav">
        <NavItem icon={<MessageSquareIcon size={20} />} label="Chat" href="#chat" />
        <NavItem icon={<KanbanIcon size={20} />} label="Kanban" href="#kanban" />
        <NavItem icon={<CpuIcon size={20} />} label="Models" href="#models" />
        <NavItem icon={<DatabaseIcon size={20} />} label="Memory" href="#memory" />
        <NavItem icon={<SettingsIcon size={20} />} label="Settings" href="#settings" />
      </nav>
      <ProfileSwitcher />
    </aside>
  )
}

const NavItem: React.FC<{ icon: React.ReactNode; label: string; href: string }> = ({
  icon,
  label,
  href,
}) => (
  <a href={href} className="nav-item" title={label}>
    <span className="nav-icon">{icon}</span>
    <span className="nav-label">{label}</span>
  </a>
)

const ProfileSwitcher: React.FC = () => {
  const { profiles, currentProfile, switchProfile } = useProfileStore()

  return (
    <div className="profile-switcher">
      <div className="current-profile" onClick={() => {}}>
        <div className="profile-avatar-small" style={{ backgroundColor: currentProfile?.color }}>
          {currentProfile?.name?.[0]?.toUpperCase()}
        </div>
        <span className="profile-name">{currentProfile?.name}</span>
        <ChevronDownIcon size={14} />
      </div>
    </div>
  )
}

import { useProfileStore } from '../stores/profileStore'
import { IsaakLogo, MessageSquareIcon, KanbanIcon, CpuIcon, DatabaseIcon, SettingsIcon, ChevronLeftIcon, ChevronDownIcon, SearchIcon, CommandIcon, UserIcon, ChevronLeftIcon as ChevronLeftIcon2 } from '../components/icons'