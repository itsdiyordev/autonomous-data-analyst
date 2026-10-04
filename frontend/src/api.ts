const base = import.meta.env.VITE_API_URL || ''

export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = localStorage.getItem('analytiq-token')
  const headers = new Headers(options.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  if (options.body && !(options.body instanceof FormData)) headers.set('Content-Type', 'application/json')
  const response = await fetch(`${base}/api${path}`, { ...options, headers })
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: 'Server could not complete this request.' }))
    if (response.status === 401 && !path.startsWith('/auth/')) window.dispatchEvent(new Event('session-expired'))
    const message = typeof body.detail === 'string' ? body.detail : body.detail?.map((e: { msg: string }) => e.msg).join('; ')
    throw new Error(message || `Request failed (${response.status})`)
  }
  if (response.status === 204) return undefined as T
  return response.json()
}

export async function download(path: string, filename: string) {
  const response = await fetch(`${base}/api${path}`, { headers: { Authorization: `Bearer ${localStorage.getItem('analytiq-token')}` } })
  if (!response.ok) throw new Error('Download failed. Please try again.')
  const url = URL.createObjectURL(await response.blob())
  const anchor = document.createElement('a')
  anchor.href = url; anchor.download = filename; anchor.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export const number = (value: number) => new Intl.NumberFormat('en', { notation: value >= 1000000 ? 'compact' : 'standard', maximumFractionDigits: 1 }).format(value)
export const date = (value: string) => new Date(value.endsWith('Z') || value.includes('+') ? value : value + 'Z').toLocaleDateString('en', { month: 'short', day: 'numeric' })
export const metric = (name: string, value: number) => name === 'clusters' ? String(Math.round(value)) : ['accuracy', 'f1', 'precision', 'recall', 'f1_macro', 'roc_auc', 'pr_auc'].includes(name) ? `${(value * 100).toFixed(1)}%` : value.toFixed(3)
export const label = (value: string) => value.replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase())
