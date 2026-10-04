import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import type { Profile } from '../types'

const grid = 'var(--border)'
const tick = { fontSize: 12, fill: 'var(--muted)' }
const tooltipStyle = { borderRadius: 12, border: '1px solid var(--border)', background: 'var(--panel)', color: 'var(--text)', boxShadow: '0 8px 25px #171b3510', fontSize: 13 }

export function DistributionChart({ data, color = '#8270df' }: { data: { label: string; value: number }[]; color?: string }) {
  return <div className="chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={data} margin={{ top: 12, right: 8, left: -24, bottom: 4 }}><CartesianGrid stroke={grid} strokeDasharray="3 5" vertical={false} /><XAxis dataKey="label" tick={tick} axisLine={false} tickLine={false} interval="preserveStartEnd" /><YAxis tick={tick} axisLine={false} tickLine={false} /><Tooltip cursor={{ fill: '#8fd4ad18' }} contentStyle={tooltipStyle} /><Bar dataKey="value" name="Records" fill={color} radius={[5, 5, 0, 0]} maxBarSize={42} /></BarChart></ResponsiveContainer></div>
}

export function ImportanceChart({ data }: { data: { feature: string; importance: number }[] }) {
  return <div className="chart importance"><ResponsiveContainer width="100%" height="100%"><BarChart data={data.slice(0, 10)} layout="vertical" margin={{ left: 5, right: 22, bottom: 4 }}><CartesianGrid stroke={grid} strokeDasharray="3 5" horizontal={false} /><XAxis type="number" tick={tick} axisLine={false} tickLine={false} /><YAxis dataKey="feature" type="category" width={130} tick={tick} axisLine={false} tickLine={false} /><Tooltip contentStyle={tooltipStyle} /><Bar dataKey="importance" name="Permutation importance" radius={[0, 5, 5, 0]} barSize={18}>{data.slice(0, 10).map((item, i) => <Cell key={item.feature} fill={i === 0 ? '#155940' : i === 1 ? '#268d68' : '#81c9a2'} />)}</Bar></BarChart></ResponsiveContainer></div>
}

export function ScatterPlot({ data, xLabel, yLabel }: { data: Record<string, number>[]; xLabel: string; yLabel: string }) {
  return <div className="chart"><ResponsiveContainer width="100%" height="100%"><ScatterChart margin={{ top: 12, right: 12, left: 0, bottom: 18 }}><CartesianGrid stroke={grid} strokeDasharray="3 5" /><XAxis dataKey="x" name={xLabel} type="number" tick={tick} axisLine={false} tickLine={false} label={{ value: xLabel, position: 'bottom', fontSize: 11, fill: '#8b9992' }} /><YAxis dataKey="y" name={yLabel} type="number" tick={tick} axisLine={false} tickLine={false} /><Tooltip contentStyle={tooltipStyle} cursor={{ strokeDasharray: '3 3' }} /><Scatter data={data} fill="#268d68" fillOpacity={0.5} /></ScatterChart></ResponsiveContainer></div>
}

export function CurveChart({ data, xKey, yKey, xLabel, yLabel }: { data: Record<string, number>[]; xKey: string; yKey: string; xLabel: string; yLabel: string }) {
  return <div className="chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={data} margin={{ left: -18, bottom: 18, right: 10 }}><CartesianGrid stroke={grid} strokeDasharray="3 5" /><XAxis dataKey={xKey} type="number" domain={[0, 1]} tick={tick} axisLine={false} tickLine={false} label={{ value: xLabel, position: 'bottom', fontSize: 11, fill: '#8b9992' }} /><YAxis domain={[0, 1]} tick={tick} axisLine={false} tickLine={false} /><Tooltip contentStyle={tooltipStyle} /><Line type="monotone" dataKey={yKey} name={yLabel} stroke="#268d68" strokeWidth={2.5} dot={false} /></LineChart></ResponsiveContainer></div>
}

export function CorrelationHeatmap({ profile }: { profile: Profile }) {
  const names = profile.correlation_columns.slice(0, 8)
  if (names.length < 2) return <div className="chart-empty">At least two numeric columns are needed.</div>
  const lookup = new Map(profile.correlations.map(p => [`${p.x}:${p.y}`, p.value]))
  return <div className="heatmap-wrap"><div className="heatmap" style={{ gridTemplateColumns: `112px repeat(${names.length}, minmax(36px, 1fr))` }}><div />{names.map((n, i) => <div key={n} className="heatmap-col" title={n}>{i + 1}</div>)}{names.map((y, i) => <div className="heatmap-row" key={y}><div className="heatmap-label" title={y}>{i + 1}. {y}</div>{names.map(x => { const value = lookup.get(`${x}:${y}`); return <div className="heatmap-cell" key={x} title={`${x} ↔ ${y}: ${value ?? 'N/A'}`} style={{ background: value == null ? '#e9edeb' : value >= 0 ? `rgba(32, 132, 91, ${0.08 + Math.abs(value) * 0.85})` : `rgba(211, 126, 100, ${0.1 + Math.abs(value) * 0.75})`, color: value != null && Math.abs(value) > 0.65 ? '#fff' : '#365348' }}>{value?.toFixed(1) ?? '—'}</div> })}</div>)}</div><div className="heatmap-legend"><span>−1 · Negative</span><div /><span>Positive · +1</span></div></div>
}

export function ClusterPlot({ data }: { data: { x: number; y: number; cluster: number }[] }) {
  const colors = ['#ae8bd7', '#86a9d4', '#8ab9a3', '#d8b681', '#d49cac', '#9fb8c9', '#c5ba8b']
  const groups = [...new Set(data.map(point => point.cluster))].sort((a, b) => a - b)
  return <><div className="chart"><ResponsiveContainer width="100%" height="100%"><ScatterChart margin={{ top: 12, right: 12, left: 0, bottom: 18 }}><CartesianGrid stroke={grid} strokeDasharray="3 5" /><XAxis dataKey="x" name="Projection 1" type="number" tick={tick} axisLine={false} tickLine={false} /><YAxis dataKey="y" name="Projection 2" type="number" tick={tick} axisLine={false} tickLine={false} /><Tooltip contentStyle={tooltipStyle} />{groups.map(group => <Scatter key={group} name={`Group ${group + 1}`} data={data.filter(point => point.cluster === group)} fill={colors[group % colors.length]} fillOpacity={0.7} />)}</ScatterChart></ResponsiveContainer></div><div className="cluster-legend">{groups.map(group => <span key={group}><i style={{ background: colors[group % colors.length] }} />Group {group + 1}</span>)}</div></>
}

export function TemporalChart({ data }: { data: { date: string; value: number | null; rolling_average: number | null }[] }) {
  return <div className="chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={data} margin={{ left: 4, right: 14, bottom: 8 }}><CartesianGrid stroke={grid} strokeDasharray="3 5" /><XAxis dataKey="date" tick={tick} tickFormatter={value => String(value).slice(0, 10)} minTickGap={50} /><YAxis tick={tick} /><Tooltip contentStyle={tooltipStyle} labelFormatter={value => String(value).slice(0, 10)} /><Line dataKey="value" name="Observed period value" stroke="#8270df" dot={false} connectNulls={false} strokeWidth={2} /><Line dataKey="rolling_average" name="Rolling average" stroke="#268d68" dot={false} connectNulls={false} strokeDasharray="4 4" /></LineChart></ResponsiveContainer></div>
}
