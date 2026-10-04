import { createContext, useContext } from 'react'
import type { User } from './types'

export const Workspace = createContext<{
  user: User; openUpload: () => void; openAnalysis: (datasetId?: string) => void; signOut: () => void
} | null>(null)

export function useWorkspace() {
  const value = useContext(Workspace)
  if (!value) throw new Error('Workspace unavailable')
  return value
}
