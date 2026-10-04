import { useCallback, useState } from 'react'
import { Link } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowRight, ArrowUpRight, Database, FileSpreadsheet, Grid2X2, List, Plus, Search, Sparkles, Trash2, UploadCloud } from '../icons'
import { toast } from 'sonner'
import { api, date, number } from '../api'
import type { Dataset } from '../types'
import { useWorkspace } from '../workspace'
import { Empty, ErrorState, Loading, Modal, PageHeading } from '../components/ui'

export default function Datasets() {
  const { openUpload, openAnalysis } = useWorkspace()
  const [search, setSearch] = useState('')
  const [grid, setGrid] = useState(true)
  const [remove, setRemove] = useState<Dataset | null>(null)
  const queryClient = useQueryClient()
  const query = useQuery({ queryKey: ['datasets'], queryFn: () => api<Dataset[]>('/datasets') })
  const closeDelete = useCallback(() => setRemove(null), [])
  function refresh() {
    queryClient.invalidateQueries({ queryKey: ['datasets'] })
    queryClient.invalidateQueries({ queryKey: ['dashboard'] })
  }
  const demo = useMutation({
    mutationFn: (kind: string) => api(`/datasets/demo?kind=${kind}`, { method: 'POST' }),
    onSuccess: () => { refresh(); toast.success('Sample customer dataset added') }, onError: (error: Error) => toast.error(error.message),
  })
  const deletion = useMutation({
    mutationFn: () => api(`/datasets/${remove!.id}`, { method: 'DELETE' }),
    onSuccess: () => { refresh(); setRemove(null); toast.success('Dataset deleted') }, onError: (error: Error) => toast.error(error.message),
  })
  const data = query.data?.filter(dataset => `${dataset.name} ${dataset.filename}`.toLowerCase().includes(search.toLowerCase())) || []

  return <>
    <PageHeading eyebrow="STEP 1 · YOUR DATA" title="All your data, in one place." text="Upload a spreadsheet, explore its charts, or use it to train a model." actions={<button className="button primary" onClick={openUpload}><Plus size={18} />Upload dataset</button>} />
    <div className="page-guide"><span className="guide-icon"><Database size={20} /></span><div><strong>Start with a CSV or Excel file.</strong><p>Each dataset gets an explorer with distributions, relationships, and a data-quality summary.</p></div><button className="button secondary" disabled={demo.isPending} onClick={() => demo.mutate('churn')}><Sparkles size={16} />{demo.isPending ? 'Adding sample…' : 'Try sample data'}</button><button className="button secondary" disabled={demo.isPending} onClick={() => demo.mutate('temporal')}>Try time-series sample</button></div>
    <div className="collection-toolbar">
      <div className="search-input"><Search size={18} /><input aria-label="Search datasets" placeholder="Search datasets or filenames…" value={search} onChange={event => setSearch(event.target.value)} /></div>
      <div className="toolbar-right"><span className="collection-count">{data.length} {data.length === 1 ? 'dataset' : 'datasets'}</span><div className="view-toggle"><button className={grid ? 'active' : ''} aria-label="Grid view" aria-pressed={grid} onClick={() => setGrid(true)}><Grid2X2 size={18} /></button><button className={!grid ? 'active' : ''} aria-label="List view" aria-pressed={!grid} onClick={() => setGrid(false)}><List size={19} /></button></div></div>
    </div>
    {query.isLoading ? <Loading /> : query.error ? <ErrorState error={query.error} retry={() => query.refetch()} /> : !data.length ? <div className="panel"><Empty title={search ? 'No matching datasets' : 'Your first dataset starts here'} text={search ? 'Search for a different name or clear the search field.' : 'Add your own spreadsheet, or try a sample to see how everything works.'} action={<button className="button primary" onClick={openUpload}><UploadCloud size={17} />Upload dataset</button>} /></div> :
      <div className={grid ? 'dataset-grid' : 'dataset-grid list-view'}>
        {data.map(dataset => <article key={dataset.id} className="dataset-card">
          <div className="dataset-card-top"><span className="dataset-file-icon"><FileSpreadsheet size={26} /></span><span className="small-tag">{dataset.filename.split('.').pop()?.toUpperCase()}</span><button className="icon-button danger-hover" aria-label={`Delete ${dataset.name}`} onClick={() => setRemove(dataset)}><Trash2 size={17} /></button></div>
          <Link className="dataset-title" to={`/datasets/${dataset.id}`}><h3>{dataset.name}</h3><ArrowUpRight size={18} /></Link>
          <p className="dataset-filename">{dataset.filename}</p>
          <div className="dataset-card-stats"><div><strong>{number(dataset.row_count)}</strong><small>Rows</small></div><div><strong>{dataset.column_count}</strong><small>Columns</small></div><div><strong>{(dataset.size_bytes / 1024).toFixed(0)} KB</strong><small>File size</small></div></div>
          <div className="quality-track"><div><span>Data quality</span><strong>{dataset.quality_score}%</strong></div><div className="progress-track"><i style={{ width: `${dataset.quality_score}%` }} /></div></div>
          <div className="dataset-card-actions"><Link className="button secondary" to={`/datasets/${dataset.id}`}>Explore data<ArrowRight size={15} /></Link><button className="button subtle" onClick={() => openAnalysis(dataset.id)}><Sparkles size={15} />Train model</button></div>
          <div className="dataset-card-footer"><span><Database size={13} />Added {date(dataset.created_at)}</span><span>Ready to explore</span></div>
        </article>)}
      </div>}
    {remove && <Modal title="Delete this dataset?" subtitle={`${remove.name} will be removed from your workspace.`} close={closeDelete}><div className="info-note">A dataset used by a model build is kept so the results remain reproducible.</div>{deletion.error && <div className="inline-error" role="alert">{deletion.error.message}</div>}<div className="modal-footer"><button className="button secondary" onClick={closeDelete}>Keep dataset</button><button className="button danger" disabled={deletion.isPending} onClick={() => deletion.mutate()}>{deletion.isPending ? 'Deleting…' : 'Delete dataset'}</button></div></Modal>}
  </>
}
