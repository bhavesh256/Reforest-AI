import { useState, useEffect, useRef, useMemo } from 'react'
import Plot from 'react-plotly.js'
import { motion, useInView, AnimatePresence } from 'framer-motion'
import {
  Trees, Activity, Info, BarChart3, TrendingUp, TrendingDown,
  AlertTriangle, ShieldCheck, Brain, Layers, Target, Leaf,
  ChevronDown, Zap, Globe, Flame, ArrowRight, Sparkles,
  ChevronUp, Menu, X
} from 'lucide-react'

// ============================================================================
// CONSTANTS
// ============================================================================
const COLORS = {
  green: '#00e676', greenDark: '#00c853', teal: '#00bfa5',
  red: '#ff5252', redLight: '#ff8a65', warning: '#ffc107',
  blue: '#448aff', purple: '#b388ff', cyan: '#40c4ff',
  white: '#e8edf5', muted: '#5a6682',
}

const PLOTLY_LAYOUT = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { family: 'Inter, sans-serif', color: '#8b97b0', size: 12 },
  margin: { t: 30, r: 20, b: 50, l: 60 },
  xaxis: { gridcolor: 'rgba(255,255,255,0.04)', linecolor: 'rgba(255,255,255,0.08)', zerolinecolor: 'rgba(255,255,255,0.06)', tickfont: { size: 11 } },
  yaxis: { gridcolor: 'rgba(255,255,255,0.04)', linecolor: 'rgba(255,255,255,0.08)', zerolinecolor: 'rgba(255,255,255,0.06)', tickfont: { size: 11 } },
  hoverlabel: { bgcolor: '#1a2332', bordercolor: 'rgba(255,255,255,0.1)', font: { color: '#e8edf5', family: 'Inter', size: 13 } },
  legend: { bgcolor: 'rgba(0,0,0,0)', font: { color: '#8b97b0', size: 11 } },
  modebar: { bgcolor: 'rgba(0,0,0,0)', color: '#5a6682', activecolor: '#00e676' },
}

const PLOTLY_CONFIG = { displayModeBar: false, responsive: true, displaylogo: false }

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================
const fmt = (n, d = 0) => {
  if (n == null || isNaN(n)) return '—'
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(1) + 'B'
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M'
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K'
  return n.toFixed(d)
}
const fmtHa = (n) => {
  if (n == null || isNaN(n)) return '—'
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + 'M ha'
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K ha'
  return n.toFixed(0) + ' ha'
}

const badgeClass = (l) => ({ Critical: 'badge-critical', High: 'badge-high', Moderate: 'badge-moderate', Low: 'badge-low' }[l] || 'badge-moderate')

// ============================================================================
// REUSABLE COMPONENTS
// ============================================================================

// SpotlightCard — Mouse-following glow
const SpotlightCard = ({ children, className = "", glowColor = "rgba(0, 230, 118, 0.06)" }) => {
  const ref = useRef(null)
  const handleMouseMove = (e) => {
    if (!ref.current) return
    const rect = ref.current.getBoundingClientRect()
    ref.current.style.setProperty("--mouse-x", `${e.clientX - rect.left}px`)
    ref.current.style.setProperty("--mouse-y", `${e.clientY - rect.top}px`)
    ref.current.style.setProperty("--glow-color", glowColor)
  }
  return (
    <div ref={ref} onMouseMove={handleMouseMove} className={`spotlight-card p-6 ${className}`}>
      {children}
    </div>
  )
}

// AnimatedNumber — Smooth count-up
const AnimatedNumber = ({ value, duration = 1500, suffix = "", prefix = "" }) => {
  const [count, setCount] = useState(0)
  useEffect(() => {
    const end = parseFloat(value) || 0
    let startTime = null
    const update = (ts) => {
      if (!startTime) startTime = ts
      const p = Math.min((ts - startTime) / duration, 1)
      const eased = 1 - Math.pow(1 - p, 3)
      setCount(end * eased)
      if (p < 1) requestAnimationFrame(update)
    }
    requestAnimationFrame(update)
  }, [value, duration])
  return <span>{prefix}{fmt(count)}{suffix}</span>
}

// FadeInSection — Animate on scroll
const FadeInSection = ({ children, className = "", delay = 0 }) => {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: "-100px" })
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 40 }}
      animate={isInView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.7, delay, ease: [0.25, 0.46, 0.45, 0.94] }}
      className={className}
    >
      {children}
    </motion.div>
  )
}

// SectionHeader
const SectionHeader = ({ icon: Icon, label, title, subtitle, color = "text-primary" }) => (
  <FadeInSection className="mb-12">
    <div className={`flex items-center gap-2 text-sm font-semibold ${color} mb-3`}>
      <Icon size={16} />
      <span className="uppercase tracking-widest">{label}</span>
    </div>
    <h2 className="text-3xl md:text-4xl font-bold text-white mb-3">{title}</h2>
    {subtitle && <p className="text-text-muted max-w-2xl text-lg">{subtitle}</p>}
  </FadeInSection>
)

// Tooltip
const Tooltip = ({ children, text }) => (
  <div className="group relative inline-flex">
    {children}
    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-3 bg-card border border-white/10 rounded-lg shadow-xl text-xs text-text-muted hidden group-hover:block z-50 pointer-events-none">
      {text}
    </div>
  </div>
)

// Priority bar
const PriorityBar = ({ score, label }) => {
  const colorMap = { Critical: '#ff5252', High: '#ffc107', Moderate: '#448aff', Low: '#00e676' }
  return (
    <div className="flex items-center gap-2">
      <div className="w-20 h-2 rounded-full bg-white/5 overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: colorMap[label] || COLORS.blue }}
          initial={{ width: 0 }}
          animate={{ width: `${score}%` }}
          transition={{ duration: 1.2, ease: "easeOut" }}
        />
      </div>
      <span className="text-xs font-mono text-text-muted">{score.toFixed(0)}</span>
    </div>
  )
}

// ============================================================================
// NAVIGATION
// ============================================================================
const sections = [
  { id: 'dashboard', label: 'Dashboard', icon: Globe },
  { id: 'past', label: 'Past', icon: BarChart3 },
  { id: 'present', label: 'Present', icon: Activity },
  { id: 'future', label: 'Future', icon: TrendingUp },
  { id: 'ai', label: 'AI Insights', icon: Brain },
  { id: 'plan', label: 'Action Plan', icon: Target },
]

const Navbar = () => {
  const [scrolled, setScrolled] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const [active, setActive] = useState('dashboard')

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50)
      const sectionEls = sections.map(s => document.getElementById(s.id)).filter(Boolean)
      for (let i = sectionEls.length - 1; i >= 0; i--) {
        if (sectionEls[i].getBoundingClientRect().top <= 120) {
          setActive(sections[i].id)
          break
        }
      }
    }
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <nav className={`fixed top-0 w-full z-50 transition-all duration-300 ${scrolled ? 'glass-panel border-x-0 border-t-0 rounded-none shadow-2xl' : 'bg-transparent border-b border-transparent'}`}>
      <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
        <a href="#dashboard" className="flex items-center gap-2.5 text-xl font-display font-bold text-white group">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-primary to-secondary flex items-center justify-center group-hover:scale-110 transition-transform">
            <Trees size={18} className="text-background" />
          </div>
          <span>ReForest <span className="text-primary">AI</span></span>
        </a>
        
        <button onClick={() => setMobileOpen(!mobileOpen)} className="md:hidden text-text-muted p-2">
          {mobileOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
        
        <div className={`${mobileOpen ? 'flex' : 'hidden'} md:flex flex-col md:flex-row absolute md:static top-full left-0 w-full md:w-auto bg-background md:bg-transparent p-6 md:p-0 gap-1 md:gap-1 border-b md:border-0 border-white/5`}>
          {sections.map(s => (
            <a
              key={s.id}
              href={`#${s.id}`}
              onClick={() => setMobileOpen(false)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                active === s.id 
                  ? 'bg-primary/10 text-primary' 
                  : 'text-text-muted hover:text-white hover:bg-white/5'
              }`}
            >
              {s.label}
            </a>
          ))}
        </div>
      </div>
    </nav>
  )
}

// ============================================================================
// SECTION: HERO DASHBOARD
// ============================================================================
const HeroDashboard = ({ data }) => {
  const o = data.overview
  return (
    <section id="dashboard" className="pt-32 pb-8">
      {/* Gradient orb background */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[600px] bg-gradient-radial from-primary/5 via-transparent to-transparent rounded-full blur-3xl pointer-events-none" />
      
      <FadeInSection className="text-center max-w-4xl mx-auto mb-20 relative">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-primary/10 border border-primary/20 text-primary text-sm font-medium mb-8">
          <Sparkles size={14} />
          Powered by Ensemble ML + Fuzzy Logic AI
        </div>
        <h1 className="text-5xl md:text-7xl font-bold leading-[1.1] mb-6">
          Reversing India's
          <br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary via-secondary to-cyan-400">
            Deforestation Crisis
          </span>
        </h1>
        <p className="text-lg md:text-xl text-text-muted max-w-2xl mx-auto leading-relaxed">
          Analyzing 20+ years of satellite data with advanced ML to predict optimal reforestation zones and model future scenarios for India's forests.
        </p>
      </FadeInSection>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        <FadeInSection delay={0.1}>
          <SpotlightCard glowColor="rgba(255, 82, 82, 0.08)">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-8 h-8 rounded-lg bg-danger/10 flex items-center justify-center"><AlertTriangle size={16} className="text-danger" /></div>
              <span className="text-sm font-medium text-text-muted">Total Forest Loss</span>
            </div>
            <div className="text-3xl md:text-4xl font-display font-bold text-danger"><AnimatedNumber value={o.total_loss_ha} suffix=" ha" /></div>
            <div className="text-xs text-text-muted mt-2">Since 2001 · GFW Data (&gt;30% canopy)</div>
          </SpotlightCard>
        </FadeInSection>

        <FadeInSection delay={0.2}>
          <SpotlightCard glowColor="rgba(255, 193, 7, 0.08)">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-8 h-8 rounded-lg bg-warning/10 flex items-center justify-center"><Flame size={16} className="text-warning" /></div>
              <span className="text-sm font-medium text-text-muted">CO₂ Released</span>
            </div>
            <div className="text-3xl md:text-4xl font-display font-bold text-warning"><AnimatedNumber value={o.total_emissions_Mg} suffix=" Mg" /></div>
            <div className="text-xs text-text-muted mt-2">Greenhouse gas emissions from loss</div>
          </SpotlightCard>
        </FadeInSection>

        <FadeInSection delay={0.3}>
          <SpotlightCard glowColor="rgba(0, 230, 118, 0.08)">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center"><Trees size={16} className="text-primary" /></div>
              <span className="text-sm font-medium text-text-muted">Reforestation Target</span>
            </div>
            <div className="text-3xl md:text-4xl font-display font-bold text-primary"><AnimatedNumber value={o.total_reforestation_needed_ha} suffix=" ha" /></div>
            <div className="text-xs text-text-muted mt-2">AI-recommended intervention area</div>
          </SpotlightCard>
        </FadeInSection>

        <FadeInSection delay={0.4}>
          <SpotlightCard className="border-primary/20" glowColor="rgba(0, 230, 118, 0.12)">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center"><ShieldCheck size={16} className="text-primary" /></div>
                <span className="text-sm font-medium text-text-muted">Official Cover</span>
              </div>
              <Tooltip text={`ISFR ${o.isfr_report_year}: Forest cover ${o.isfr_forest_cover_pct}% (${(o.isfr_forest_area_sqkm/1000).toFixed(0)}K km²) + Tree cover ${o.isfr_tree_cover_pct}% = ${o.isfr_total_cover_pct}% total. GFW measures only dense canopy (>30%).`}>
                <Info size={14} className="text-text-muted cursor-help hover:text-primary transition-colors" />
              </Tooltip>
            </div>
            <div className="text-3xl md:text-4xl font-display font-bold text-white"><AnimatedNumber value={o.isfr_total_cover_pct} suffix="%" /></div>
            <div className="text-xs text-primary mt-2">ISFR {o.isfr_report_year} · Forest + Tree Cover</div>
          </SpotlightCard>
        </FadeInSection>
      </div>
    </section>
  )
}

// ============================================================================
// SECTION: PAST ANALYSIS
// ============================================================================
const PastAnalysis = ({ data }) => {
  const t = data.timeline
  const states = data.states
  if (!t) return null

  return (
    <section id="past" className="py-24">
      <SectionHeader icon={BarChart3} label="Historical Analysis" title="Two Decades of Forest Loss" subtitle="India's deforestation trajectory from 2001 to 2020 — patterns, peaks, and the compounding crisis." color="text-danger" />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <FadeInSection>
          <SpotlightCard glowColor="rgba(255, 82, 82, 0.06)">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <BarChart3 size={18} className="text-danger" /> Annual Tree Cover Loss
            </h3>
            <Plot
              data={[{
                x: t.years, y: t.annual_loss_ha, type: 'bar',
                marker: {
                  color: t.annual_loss_ha.map(v => {
                    const max = Math.max(...t.annual_loss_ha)
                    const r = v / max
                    return r > 0.8 ? COLORS.red : r > 0.5 ? COLORS.warning : COLORS.teal
                  }),
                  opacity: 0.85,
                },
                hovertemplate: '<b>%{x}</b><br>Loss: %{y:,.0f} ha<extra></extra>',
              }]}
              layout={{ ...PLOTLY_LAYOUT, xaxis: { ...PLOTLY_LAYOUT.xaxis, dtick: 2 }, yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Loss (ha)' }, height: 350 }}
              config={PLOTLY_CONFIG}
              className="w-full"
            />
          </SpotlightCard>
        </FadeInSection>

        <FadeInSection delay={0.15}>
          <SpotlightCard glowColor="rgba(255, 82, 82, 0.06)">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <TrendingUp size={18} className="text-danger" /> Cumulative Loss
            </h3>
            <Plot
              data={[{
                x: t.years, y: t.cumulative_loss_ha, type: 'scatter', mode: 'lines+markers',
                fill: 'tozeroy', fillcolor: 'rgba(255, 82, 82, 0.06)',
                line: { color: COLORS.red, width: 3, shape: 'spline' },
                marker: { size: 4, color: COLORS.red },
                hovertemplate: '<b>%{x}</b><br>Cumulative: %{y:,.0f} ha<extra></extra>',
              }]}
              layout={{ ...PLOTLY_LAYOUT, xaxis: { ...PLOTLY_LAYOUT.xaxis, dtick: 2 }, yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Cumulative (ha)' }, height: 350 }}
              config={PLOTLY_CONFIG}
              className="w-full"
            />
          </SpotlightCard>
        </FadeInSection>
      </div>

      <FadeInSection delay={0.2}>
        <SpotlightCard glowColor="rgba(255, 193, 7, 0.06)">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Flame size={18} className="text-warning" /> CO₂ Emissions from Deforestation
          </h3>
          <Plot
            data={[{
              x: t.years, y: t.annual_emissions_Mg, type: 'scatter', mode: 'lines+markers',
              fill: 'tozeroy', fillcolor: 'rgba(255, 193, 7, 0.04)',
              line: { color: COLORS.warning, width: 3, shape: 'spline' },
              marker: { size: 4, color: COLORS.warning },
              hovertemplate: '<b>%{x}</b><br>Emissions: %{y:,.0f} Mg CO₂<extra></extra>',
            }]}
            layout={{ ...PLOTLY_LAYOUT, xaxis: { ...PLOTLY_LAYOUT.xaxis, dtick: 2 }, yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'CO₂ Emissions (Mg)' }, height: 300 }}
            config={PLOTLY_CONFIG}
            className="w-full"
          />
        </SpotlightCard>
      </FadeInSection>

      {/* Heatmap */}
      {states && states.length > 0 && (
        <FadeInSection delay={0.3} className="mt-8">
          <SpotlightCard>
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Layers size={18} className="text-purple-400" /> State-wise Loss Heatmap (Top 15)
            </h3>
            <Plot
              data={[{
                z: states.slice(0, 15).map(s => s.annual_losses),
                x: Array.from({ length: 20 }, (_, i) => 2001 + i),
                y: states.slice(0, 15).map(s => s.name),
                type: 'heatmap',
                colorscale: [[0, '#0a0e17'], [0.2, '#1a4731'], [0.4, '#2d7a4f'], [0.6, '#ffc107'], [0.8, '#ff8a65'], [1, '#ff5252']],
                hovertemplate: '<b>%{y}</b> (%{x})<br>Loss: %{z:,.0f} ha<extra></extra>',
                colorbar: { title: { text: 'Loss (ha)', font: { color: '#8b97b0' } }, tickfont: { color: '#8b97b0' } },
              }]}
              layout={{ ...PLOTLY_LAYOUT, margin: { t: 20, r: 100, b: 50, l: 160 }, xaxis: { ...PLOTLY_LAYOUT.xaxis, dtick: 2 }, yaxis: { ...PLOTLY_LAYOUT.yaxis, autorange: 'reversed' }, height: 450 }}
              config={PLOTLY_CONFIG}
              className="w-full"
            />
          </SpotlightCard>
        </FadeInSection>
      )}
    </section>
  )
}

// ============================================================================
// SECTION: PRESENT STATE
// ============================================================================
const PresentState = ({ data }) => {
  const o = data.overview
  const states = data.states
  if (!states) return null

  const top10 = states.slice(0, 10)

  return (
    <section id="present" className="py-24">
      <SectionHeader icon={Activity} label="Current Status" title="Where India Stands Today" subtitle="Vulnerability analysis of 36 states and union territories using AI-driven fuzzy logic scoring." color="text-cyan-400" />

      <div className="grid grid-cols-2 md:grid-cols-4 gap-5 mb-10">
        {[
          { label: 'Dense Forest Cover', value: `${o.forest_cover_pct_current}%`, detail: 'GFW >30% canopy threshold', color: 'text-primary', icon: Trees },
          { label: 'Net Forest Loss', value: fmtHa(o.net_loss_ha), detail: 'Total loss minus total gain', color: 'text-danger', icon: TrendingDown },
          { label: 'Critical States', value: o.num_critical_states, detail: 'Identified by fuzzy logic AI', color: 'text-danger', icon: AlertTriangle },
          { label: 'Avg. Annual Loss', value: fmtHa(o.avg_annual_loss_ha), detail: 'Per year over 20 years', color: 'text-warning', icon: Activity },
        ].map((item, i) => (
          <FadeInSection key={i} delay={i * 0.1}>
            <SpotlightCard>
              <div className="flex items-center gap-2 mb-2">
                <item.icon size={16} className={item.color} />
                <span className="text-xs font-medium text-text-muted">{item.label}</span>
              </div>
              <div className={`text-2xl md:text-3xl font-display font-bold ${item.color}`}>{item.value}</div>
              <div className="text-xs text-text-muted mt-1">{item.detail}</div>
            </SpotlightCard>
          </FadeInSection>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <FadeInSection>
          <SpotlightCard>
            <h3 className="text-lg font-semibold text-white mb-4">Top 10 Most Affected States</h3>
            <Plot
              data={[{
                y: top10.map(s => s.name).reverse(),
                x: top10.map(s => s.total_loss_ha).reverse(),
                type: 'bar', orientation: 'h',
                marker: { color: top10.map((_, i) => i < 3 ? COLORS.red : i < 6 ? COLORS.warning : COLORS.teal).reverse(), opacity: 0.85 },
                hovertemplate: '<b>%{y}</b><br>Total Loss: %{x:,.0f} ha<extra></extra>',
              }]}
              layout={{ ...PLOTLY_LAYOUT, margin: { t: 10, r: 20, b: 50, l: 160 }, xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'Total Loss (ha)' }, height: 400 }}
              config={PLOTLY_CONFIG}
              className="w-full"
            />
          </SpotlightCard>
        </FadeInSection>

        <FadeInSection delay={0.15}>
          <SpotlightCard className="max-h-[480px] overflow-y-auto">
            <h3 className="text-lg font-semibold text-white mb-4">State Rankings</h3>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-text-muted border-b border-white/5">
                  <th className="py-2 pr-2">#</th>
                  <th className="py-2">State</th>
                  <th className="py-2 text-right">Loss</th>
                  <th className="py-2 text-right">Priority</th>
                  <th className="py-2 text-right">Score</th>
                </tr>
              </thead>
              <tbody>
                {states.map((s, i) => (
                  <tr key={s.name} className="border-b border-white/3 hover:bg-white/3 transition-colors">
                    <td className="py-2.5 pr-2"><span className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold ${i < 3 ? 'bg-danger/20 text-danger' : 'bg-white/5 text-text-muted'}`}>{i + 1}</span></td>
                    <td className="py-2.5 font-medium text-white text-xs">{s.name}</td>
                    <td className="py-2.5 text-right text-xs font-mono" style={{ color: s.total_loss_ha > 50000 ? COLORS.red : COLORS.white }}>{fmtHa(s.total_loss_ha)}</td>
                    <td className="py-2.5 text-right"><span className={`${badgeClass(s.fuzzy_priority_label)} text-[10px] px-2 py-0.5 rounded-full font-semibold`}>{s.fuzzy_priority_label}</span></td>
                    <td className="py-2.5 text-right"><PriorityBar score={s.vulnerability_score * 100} label={s.fuzzy_priority_label} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </SpotlightCard>
        </FadeInSection>
      </div>
    </section>
  )
}

// ============================================================================
// SECTION: FUTURE SCENARIOS
// ============================================================================
const FutureScenarios = ({ data }) => {
  const pred = data.predictions
  if (!pred || !pred.country_forecast) return null
  const cf = pred.country_forecast

  const bauTotal = cf.bau_scenario.annual_loss_ha.reduce((a, b) => a + b, 0)
  const refTotal = cf.reforestation_scenario.annual_loss_ha.reduce((a, b) => a + b, 0)
  const bauEmTotal = cf.bau_scenario.annual_emissions_Mg.reduce((a, b) => a + b, 0)
  const refEmTotal = cf.reforestation_scenario.annual_emissions_Mg.reduce((a, b) => a + b, 0)

  return (
    <section id="future" className="py-24">
      <SectionHeader icon={TrendingUp} label="Predictive Modeling" title="Two Possible Futures" subtitle="What happens if we do nothing vs. what's possible with strategic reforestation." color="text-purple-400" />

      {/* Scenario comparison cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-10">
        <FadeInSection>
          <SpotlightCard className="border-danger/20" glowColor="rgba(255, 82, 82, 0.08)">
            <div className="flex items-center gap-3 mb-5">
              <div className="w-10 h-10 rounded-xl bg-danger/10 flex items-center justify-center"><TrendingUp size={20} className="text-danger" /></div>
              <div>
                <h3 className="text-lg font-bold text-white">Business as Usual</h3>
                <p className="text-xs text-text-muted">If deforestation continues unchecked</p>
              </div>
            </div>
            <div className="space-y-4">
              <div className="flex justify-between items-center py-2 border-b border-white/5">
                <span className="text-sm text-text-muted">Additional Loss (2021-2030)</span>
                <span className="text-lg font-bold text-danger">{fmtHa(bauTotal)}</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-white/5">
                <span className="text-sm text-text-muted">Projected CO₂ Emissions</span>
                <span className="text-lg font-bold text-warning">{fmt(bauEmTotal)} Mg</span>
              </div>
              <div className="flex justify-between items-center py-2">
                <span className="text-sm text-text-muted">Cumulative Loss by 2030</span>
                <span className="text-lg font-bold text-danger">{fmtHa(cf.bau_scenario.cumulative_loss_ha[cf.bau_scenario.cumulative_loss_ha.length - 1])}</span>
              </div>
            </div>
          </SpotlightCard>
        </FadeInSection>

        <FadeInSection delay={0.15}>
          <SpotlightCard className="border-primary/20" glowColor="rgba(0, 230, 118, 0.08)">
            <div className="flex items-center gap-3 mb-5">
              <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center"><Leaf size={20} className="text-primary" /></div>
              <div>
                <h3 className="text-lg font-bold text-white">With Reforestation</h3>
                <p className="text-xs text-text-muted">Active intervention + strategic planting</p>
              </div>
            </div>
            <div className="space-y-4">
              <div className="flex justify-between items-center py-2 border-b border-white/5">
                <span className="text-sm text-text-muted">Reduced Loss (2021-2030)</span>
                <span className="text-lg font-bold text-primary">{fmtHa(refTotal)}</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-white/5">
                <span className="text-sm text-text-muted">Carbon Saved</span>
                <span className="text-lg font-bold text-primary">{fmt(Math.max(0, bauEmTotal - refEmTotal))} Mg</span>
              </div>
              <div className="flex justify-between items-center py-2">
                <span className="text-sm text-text-muted">Cumulative Loss by 2030</span>
                <span className="text-lg font-bold text-primary">{fmtHa(cf.reforestation_scenario.cumulative_loss_ha[cf.reforestation_scenario.cumulative_loss_ha.length - 1])}</span>
              </div>
            </div>
          </SpotlightCard>
        </FadeInSection>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <FadeInSection>
          <SpotlightCard>
            <h3 className="text-lg font-semibold text-white mb-4">Annual Loss Projection</h3>
            <Plot
              data={[
                { x: cf.past.years, y: cf.past.annual_loss_ha, type: 'scatter', mode: 'lines+markers', name: 'Historical', line: { color: COLORS.white, width: 2 }, marker: { size: 3 } },
                { x: cf.bau_scenario.years, y: cf.bau_scenario.annual_loss_ha, type: 'scatter', mode: 'lines+markers', name: 'BAU', line: { color: COLORS.red, width: 3, dash: 'dot' }, marker: { size: 4, symbol: 'diamond' }, fill: 'tozeroy', fillcolor: 'rgba(255,82,82,0.04)' },
                { x: cf.reforestation_scenario.years, y: cf.reforestation_scenario.annual_loss_ha, type: 'scatter', mode: 'lines+markers', name: 'Reforestation', line: { color: COLORS.green, width: 3, dash: 'dot' }, marker: { size: 4, symbol: 'star' }, fill: 'tozeroy', fillcolor: 'rgba(0,230,118,0.04)' },
              ]}
              layout={{
                ...PLOTLY_LAYOUT,
                xaxis: { ...PLOTLY_LAYOUT.xaxis, dtick: 2 }, yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Annual Loss (ha)' },
                legend: { ...PLOTLY_LAYOUT.legend, x: 0, y: 1.15, orientation: 'h' },
                shapes: [{ type: 'line', x0: 2020.5, x1: 2020.5, y0: 0, y1: 1, yref: 'paper', line: { color: COLORS.muted, width: 2, dash: 'dash' } }],
                annotations: [{ x: 2020.5, y: 1.05, yref: 'paper', text: 'Forecast →', showarrow: false, font: { color: COLORS.muted, size: 11 } }],
                height: 380,
              }}
              config={PLOTLY_CONFIG}
              className="w-full"
            />
          </SpotlightCard>
        </FadeInSection>

        <FadeInSection delay={0.15}>
          <SpotlightCard>
            <h3 className="text-lg font-semibold text-white mb-4">Cumulative Loss Trajectory</h3>
            <Plot
              data={[
                { x: cf.past.years, y: cf.past.cumulative_loss_ha, type: 'scatter', mode: 'lines', name: 'Historical', line: { color: COLORS.white, width: 2 } },
                { x: cf.bau_scenario.years, y: cf.bau_scenario.cumulative_loss_ha, type: 'scatter', mode: 'lines', name: 'BAU', line: { color: COLORS.red, width: 3, dash: 'dot' }, fill: 'tonexty', fillcolor: 'rgba(255,82,82,0.04)' },
                { x: cf.reforestation_scenario.years, y: cf.reforestation_scenario.cumulative_loss_ha, type: 'scatter', mode: 'lines', name: 'Reforestation', line: { color: COLORS.green, width: 3, dash: 'dot' } },
              ]}
              layout={{ ...PLOTLY_LAYOUT, xaxis: { ...PLOTLY_LAYOUT.xaxis, dtick: 2 }, yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Cumulative (ha)' }, legend: { ...PLOTLY_LAYOUT.legend, x: 0, y: 1.15, orientation: 'h' }, height: 380 }}
              config={PLOTLY_CONFIG}
              className="w-full"
            />
          </SpotlightCard>
        </FadeInSection>
      </div>

      {/* Carbon savings */}
      <FadeInSection delay={0.2} className="mt-6">
        <SpotlightCard>
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Leaf size={18} className="text-primary" /> Cumulative Carbon Savings from Reforestation
          </h3>
          <Plot
            data={[{
              x: cf.reforestation_scenario.years,
              y: cf.reforestation_scenario.carbon_saved_Mg || cf.reforestation_scenario.years.map(() => 0),
              type: 'bar',
              marker: { color: 'rgba(0, 230, 118, 0.6)', line: { width: 0 } },
              hovertemplate: '<b>%{x}</b><br>Carbon Saved: %{y:,.0f} Mg CO₂<extra></extra>',
            }]}
            layout={{ ...PLOTLY_LAYOUT, xaxis: { ...PLOTLY_LAYOUT.xaxis, dtick: 1 }, yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Cumulative Saved (Mg CO₂)' }, height: 280 }}
            config={PLOTLY_CONFIG}
            className="w-full"
          />
        </SpotlightCard>
      </FadeInSection>
    </section>
  )
}

// ============================================================================
// SECTION: ML INSIGHTS
// ============================================================================
const MLInsights = ({ data }) => {
  const mp = data.modelPerformance
  if (!mp) return null
  const models = mp.models || mp
  const featureImp = data.modelPerformance?.feature_importance || {}
  const smote = data.modelPerformance?.smote_analysis || {}

  // Reconstruct models if the API returns flat structure
  const modelEntries = Object.entries(models).filter(([k]) => !['feature_importance', 'smote_analysis'].includes(k))

  let bestName = ''
  let bestR2 = -Infinity
  modelEntries.forEach(([n, m]) => { if (m.test_r2 > bestR2) { bestR2 = m.test_r2; bestName = n } })

  const icons = { 'Random Forest': '🌲', 'XGBoost': '🚀', 'LightGBM': '💡', 'CatBoost': '🐱', 'AdaBoost': '🔥', SVR: '📐', Ridge: '📏', 'Stacking Ensemble': '🏗️' }

  return (
    <section id="ai" className="py-24">
      <SectionHeader icon={Brain} label="Machine Learning" title="AI Model Performance" subtitle="8 ensemble models trained, cross-validated, and compared. SMOTE applied to balance priority classification." color="text-blue-400" />

      {/* Model Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 mb-10">
        {modelEntries.map(([name, m], i) => (
          <FadeInSection key={name} delay={i * 0.08}>
            <SpotlightCard className={name === bestName ? 'border-primary/30 bg-primary/3' : ''} glowColor={name === bestName ? 'rgba(0,230,118,0.1)' : undefined}>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <span className="text-xl">{icons[name] || '🤖'}</span>
                  <span className="font-semibold text-white text-sm">{name}</span>
                </div>
                {name === bestName && <span className="text-xs px-2 py-0.5 bg-primary/15 text-primary rounded-full font-bold">🏆 Best</span>}
              </div>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-text-muted">R² Score</span>
                    <span className="font-mono font-bold" style={{ color: m.test_r2 > 0.8 ? COLORS.green : m.test_r2 > 0.5 ? COLORS.warning : COLORS.red }}>{m.test_r2.toFixed(4)}</span>
                  </div>
                  <div className="w-full h-1.5 rounded-full bg-white/5 overflow-hidden">
                    <motion.div className="h-full rounded-full bg-gradient-to-r from-secondary to-primary" initial={{ width: 0 }} animate={{ width: `${Math.max(0, m.test_r2) * 100}%` }} transition={{ duration: 1.5, delay: i * 0.1 }} />
                  </div>
                </div>
                <div className="flex justify-between text-xs"><span className="text-text-muted">MAE</span><span className="font-mono text-white">{fmt(m.test_mae)}</span></div>
                <div className="flex justify-between text-xs"><span className="text-text-muted">RMSE</span><span className="font-mono text-white">{fmt(m.test_rmse)}</span></div>
                {m.cv_r2_mean !== undefined && <div className="flex justify-between text-xs"><span className="text-text-muted">CV R² Mean</span><span className="font-mono text-white">{m.cv_r2_mean.toFixed(4)}</span></div>}
              </div>
            </SpotlightCard>
          </FadeInSection>
        ))}
      </div>

      {/* Model Comparison Chart */}
      <FadeInSection>
        <SpotlightCard className="mb-8">
          <h3 className="text-lg font-semibold text-white mb-4">Train vs Test R² Comparison</h3>
          <Plot
            data={[
              { x: modelEntries.map(([n]) => n), y: modelEntries.map(([, m]) => m.train_r2), type: 'bar', name: 'Train R²', marker: { color: 'rgba(0, 191, 165, 0.6)' } },
              { x: modelEntries.map(([n]) => n), y: modelEntries.map(([, m]) => m.test_r2), type: 'bar', name: 'Test R²', marker: { color: 'rgba(0, 230, 118, 0.8)' } },
            ]}
            layout={{ ...PLOTLY_LAYOUT, barmode: 'group', yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'R² Score', range: [Math.min(0, ...modelEntries.map(([, m]) => m.test_r2)) - 0.1, 1.05] }, legend: { ...PLOTLY_LAYOUT.legend, x: 0, y: 1.15, orientation: 'h' }, height: 350 }}
            config={PLOTLY_CONFIG}
            className="w-full"
          />
        </SpotlightCard>
      </FadeInSection>

      {/* SMOTE Before/After */}
      {smote.before_smote && (
        <FadeInSection className="mb-8">
          <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2"><Zap size={20} className="text-warning" /> SMOTE: Before vs After</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            <SpotlightCard className="border-warning/20" glowColor="rgba(255,193,7,0.08)">
              <div className="text-center">
                <div className="text-xs uppercase tracking-widest text-warning font-semibold mb-3">Before SMOTE</div>
                <div className="text-4xl font-display font-bold text-warning mb-1">{(smote.before_smote.accuracy * 100).toFixed(1)}%</div>
                <div className="text-sm text-text-muted">Accuracy</div>
                <div className="mt-3 text-xs text-text-muted">F1: {(smote.before_smote.f1_macro * 100).toFixed(1)}%</div>
              </div>
            </SpotlightCard>

            <div className="flex items-center justify-center">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ duration: 0.5, delay: 0.3 }}
                className="w-16 h-16 rounded-full bg-gradient-to-r from-warning to-primary flex items-center justify-center"
              >
                <ArrowRight size={24} className="text-background" />
              </motion.div>
            </div>

            <SpotlightCard className="border-primary/20" glowColor="rgba(0,230,118,0.08)">
              <div className="text-center">
                <div className="text-xs uppercase tracking-widest text-primary font-semibold mb-3">After SMOTE</div>
                <div className="text-4xl font-display font-bold text-primary mb-1">{(smote.after_smote.accuracy * 100).toFixed(1)}%</div>
                <div className="text-sm text-text-muted">Accuracy</div>
                <div className="mt-3 text-xs text-text-muted">F1: {(smote.after_smote.f1_macro * 100).toFixed(1)}%</div>
              </div>
            </SpotlightCard>
          </div>
        </FadeInSection>
      )}

      {/* Feature Importance */}
      {Object.keys(featureImp).length > 0 && (
        <FadeInSection>
          <SpotlightCard>
            <h3 className="text-lg font-semibold text-white mb-4">Feature Importance</h3>
            <Plot
              data={Object.entries(featureImp).map(([model, features]) => {
                const top = (features || []).slice(0, 8)
                const colorMap = { 'Random Forest': COLORS.teal, XGBoost: COLORS.blue, LightGBM: COLORS.purple, CatBoost: COLORS.warning, AdaBoost: COLORS.red }
                return {
                  y: top.map(f => f.feature).reverse(),
                  x: top.map(f => f.importance).reverse(),
                  type: 'bar', orientation: 'h', name: model,
                  marker: { color: colorMap[model] || COLORS.white, opacity: 0.8 },
                }
              })}
              layout={{ ...PLOTLY_LAYOUT, barmode: 'group', margin: { t: 20, r: 20, b: 50, l: 200 }, xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'Importance' }, legend: { ...PLOTLY_LAYOUT.legend, x: 0.5, y: 1.15, orientation: 'h' }, height: 380 }}
              config={PLOTLY_CONFIG}
              className="w-full"
            />
          </SpotlightCard>
        </FadeInSection>
      )}
    </section>
  )
}

// ============================================================================
// SECTION: REFORESTATION PLAN
// ============================================================================
const ReforestationPlan = ({ data }) => {
  const plan = data.reforestationPlan
  const o = data.overview
  const forecasts = data.predictions?.state_forecasts || {}
  if (!plan || !o) return null

  const sorted = [...plan].filter(s => s.reforestation_needed_ha > 0).sort((a, b) => b.priority_score - a.priority_score)
  const priorityColors = { Critical: COLORS.red, High: COLORS.warning, Moderate: COLORS.blue, Low: COLORS.green }

  return (
    <section id="plan" className="py-24">
      <SectionHeader icon={Target} label="Action Plan" title="Strategic Reforestation Blueprint" subtitle="AI-prioritized regions requiring immediate intervention to reverse deforestation damage." color="text-primary" />

      {/* Summary cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-10">
        {[
          { label: 'Total Reforestation Needed', value: fmtHa(o.total_reforestation_needed_ha), detail: `Across ${o.num_states} states by 2030`, color: 'text-primary', icon: Trees },
          { label: 'Carbon Savings Potential', value: `${fmt(o.total_carbon_savings_Mg)} Mg`, detail: 'CO₂ prevented from atmosphere', color: 'text-secondary', icon: Leaf },
          { label: 'Urgent States', value: `${o.num_critical_states + o.num_high_states}`, detail: 'Critical + High priority zones', color: 'text-danger', icon: AlertTriangle },
        ].map((item, i) => (
          <FadeInSection key={i} delay={i * 0.1}>
            <SpotlightCard>
              <div className="flex items-center gap-2 mb-2">
                <item.icon size={16} className={item.color} />
                <span className="text-xs font-medium text-text-muted">{item.label}</span>
              </div>
              <div className={`text-3xl font-display font-bold ${item.color}`}>{item.value}</div>
              <div className="text-xs text-text-muted mt-1">{item.detail}</div>
            </SpotlightCard>
          </FadeInSection>
        ))}
      </div>

      {/* Priority Chart */}
      <FadeInSection className="mb-8">
        <SpotlightCard>
          <h3 className="text-lg font-semibold text-white mb-4">Fuzzy Logic Priority Scores by State</h3>
          <Plot
            data={[{
              y: sorted.map(s => s.name).reverse(),
              x: sorted.map(s => s.priority_score).reverse(),
              type: 'bar', orientation: 'h',
              marker: { color: sorted.map(s => priorityColors[s.priority_label] || COLORS.blue).reverse(), opacity: 0.85 },
              hovertemplate: '<b>%{y}</b><br>Score: %{x:.1f}/100<extra></extra>',
            }]}
            layout={{ ...PLOTLY_LAYOUT, margin: { t: 10, r: 20, b: 50, l: 160 }, xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'Priority Score (0-100)', range: [0, 100] }, height: Math.max(400, sorted.length * 28) }}
            config={PLOTLY_CONFIG}
            className="w-full"
          />
        </SpotlightCard>
      </FadeInSection>

      {/* Reforestation Table */}
      <FadeInSection>
        <SpotlightCard>
          <h3 className="text-lg font-semibold text-white mb-4">Detailed Reforestation Requirements</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-text-muted border-b border-white/10">
                  <th className="py-3 pr-3">#</th>
                  <th className="py-3">State</th>
                  <th className="py-3">Priority</th>
                  <th className="py-3 text-right">Score</th>
                  <th className="py-3 text-right">Reforestation Needed</th>
                  <th className="py-3 text-right">Carbon Savings</th>
                  <th className="py-3 text-right">Current Loss Rate</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((s, i) => (
                  <tr key={s.name} className="border-b border-white/3 hover:bg-white/3 transition-colors">
                    <td className="py-3 pr-3"><span className={`inline-flex items-center justify-center w-7 h-7 rounded-full text-xs font-bold ${i < 3 ? 'bg-danger/20 text-danger' : 'bg-white/5 text-text-muted'}`}>{i + 1}</span></td>
                    <td className="py-3 font-semibold text-white">{s.name}</td>
                    <td className="py-3"><span className={`${badgeClass(s.priority_label)} text-xs px-2.5 py-1 rounded-full font-semibold`}>{s.priority_label}</span></td>
                    <td className="py-3 text-right"><PriorityBar score={s.priority_score} label={s.priority_label} /></td>
                    <td className="py-3 text-right font-semibold text-primary">{fmtHa(Math.max(0, s.reforestation_needed_ha))}</td>
                    <td className="py-3 text-right text-text-muted">{fmt(Math.max(0, s.carbon_savings_Mg))} Mg</td>
                    <td className="py-3 text-right text-text-muted">{fmtHa(s.current_loss_rate)}/yr</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SpotlightCard>
      </FadeInSection>
    </section>
  )
}

// ============================================================================
// FOOTER
// ============================================================================
const Footer = () => (
  <footer className="border-t border-white/5 py-12 mt-16">
    <div className="max-w-7xl mx-auto px-6 text-center">
      <div className="flex items-center justify-center gap-2 mb-4">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-secondary flex items-center justify-center">
          <Trees size={16} className="text-background" />
        </div>
        <span className="font-display font-bold text-white">ReForest AI</span>
      </div>
      <p className="text-sm text-text-muted mb-2">
        ML-powered reforestation analysis for India · Data: Global Forest Watch · ISFR 2023
      </p>
      <p className="text-xs text-text-muted/50">
        Built with Ensemble Learning, Fuzzy Logic, SMOTE, and Time-Series Forecasting
      </p>
    </div>
  </footer>
)

// ============================================================================
// MAIN APP
// ============================================================================
function App() {
  const [appData, setAppData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [overview, timeline, states, predictions, modelPerf, reforestPlan] = await Promise.all([
          fetch('/api/overview').then(r => r.json()),
          fetch('/api/timeline').then(r => r.json()),
          fetch('/api/states').then(r => r.json()),
          fetch('/api/predictions').then(r => r.json()),
          fetch('/api/model-performance').then(r => r.json()),
          fetch('/api/reforestation-plan').then(r => r.json()),
        ])
        setAppData({ overview, timeline, states, predictions, modelPerformance: modelPerf, reforestationPlan: reforestPlan })
      } catch (err) {
        console.error("Failed to fetch data:", err)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  if (loading || !appData) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex flex-col items-center gap-6"
        >
          <div className="relative">
            <div className="w-16 h-16 border-4 border-primary/20 border-t-primary rounded-full animate-spin" />
            <Trees size={20} className="text-primary absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
          </div>
          <div className="text-xl font-display font-bold text-primary">Loading Reforestation Intelligence</div>
          <div className="text-sm text-text-muted">Analyzing satellite data & ML predictions...</div>
        </motion.div>
      </div>
    )
  }

  return (
    <div className="min-h-screen relative overflow-x-hidden">
      {/* Background gradient orbs */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
        <div className="absolute top-[10%] left-[10%] w-[500px] h-[500px] bg-primary/3 rounded-full blur-[150px]" />
        <div className="absolute top-[60%] right-[5%] w-[400px] h-[400px] bg-purple-500/3 rounded-full blur-[120px]" />
        <div className="absolute bottom-[10%] left-[30%] w-[300px] h-[300px] bg-cyan-400/3 rounded-full blur-[100px]" />
      </div>

      <Navbar />

      <main className="relative z-10 px-6 md:px-12 lg:px-24 max-w-7xl mx-auto">
        <HeroDashboard data={appData} />
        <PastAnalysis data={appData} />
        <PresentState data={appData} />
        <FutureScenarios data={appData} />
        <MLInsights data={appData} />
        <ReforestationPlan data={appData} />
      </main>

      <Footer />
    </div>
  )
}

export default App
