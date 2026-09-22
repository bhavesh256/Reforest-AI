import { useState, useEffect, useRef, useMemo } from 'react'
import Plot from 'react-plotly.js'
import { motion, useInView, AnimatePresence } from 'framer-motion'
import {
  Trees, Activity, Info, BarChart3, TrendingUp, TrendingDown,
  AlertTriangle, ShieldCheck, Brain, Layers, Target, Leaf,
  ChevronDown, Zap, Globe, Flame, ArrowRight,
  ChevronUp, Menu, X, Filter
} from 'lucide-react'

// ============================================================================
// CONSTANTS
// ============================================================================
const COLORS = {
  green: '#00e676', greenDark: '#00c853', teal: '#00bfa5',
  red: '#ff5252', redLight: '#ff8a65', warning: '#ffc107',
  blue: '#448aff', purple: '#b388ff', cyan: '#40c4ff',
  white: '#f0f4fc', muted: '#94a3b8', background: '#060910'
}

const PLOTLY_LAYOUT = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { family: 'Inter, sans-serif', color: '#94a3b8', size: 12 },
  margin: { t: 30, r: 20, b: 50, l: 60 },
  xaxis: { gridcolor: 'rgba(255,255,255,0.05)', linecolor: 'rgba(255,255,255,0.1)', zerolinecolor: 'rgba(255,255,255,0.1)', tickfont: { size: 11 } },
  yaxis: { gridcolor: 'rgba(255,255,255,0.05)', linecolor: 'rgba(255,255,255,0.1)', zerolinecolor: 'rgba(255,255,255,0.1)', tickfont: { size: 11 } },
  hoverlabel: { bgcolor: '#0f1524', bordercolor: 'rgba(255,255,255,0.15)', font: { color: '#f0f4fc', family: 'Inter', size: 13 } },
  legend: { bgcolor: 'rgba(0,0,0,0)', font: { color: '#94a3b8', size: 11 } },
  modebar: { bgcolor: 'rgba(0,0,0,0)', color: '#94a3b8', activecolor: '#00e676' },
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
  return n.toLocaleString(undefined, { maximumFractionDigits: d })
}
const fmtHa = (n) => {
  if (n == null || isNaN(n)) return '—'
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + 'M ha'
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K ha'
  return n.toLocaleString(undefined, { maximumFractionDigits: 0 }) + ' ha'
}

const badgeClass = (l) => ({ Critical: 'badge-critical', High: 'badge-high', Moderate: 'badge-moderate', Low: 'badge-low' }[l] || 'badge-moderate')

// ============================================================================
// REUSABLE COMPONENTS
// ============================================================================

// SpotlightCard — Mouse-following glow
const SpotlightCard = ({ children, className = "", glowColor = "rgba(0, 230, 118, 0.08)", style = {} }) => {
  const ref = useRef(null)
  const handleMouseMove = (e) => {
    if (!ref.current) return
    const rect = ref.current.getBoundingClientRect()
    ref.current.style.setProperty("--mouse-x", `${e.clientX - rect.left}px`)
    ref.current.style.setProperty("--mouse-y", `${e.clientY - rect.top}px`)
    ref.current.style.setProperty("--glow-color", glowColor)
  }
  return (
    <div ref={ref} onMouseMove={handleMouseMove} className={`spotlight-card p-6 ${className}`} style={style}>
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
    <h2 className="text-3xl md:text-5xl font-bold text-white mb-4">{title}</h2>
    {subtitle && <p className="text-text-muted max-w-2xl text-lg">{subtitle}</p>}
    <div className={`h-1 w-20 rounded-full mt-6 bg-gradient-to-r from-transparent via-${color.replace('text-', '')} to-transparent opacity-50`} />
  </FadeInSection>
)

// Tooltip
const Tooltip = ({ children, text }) => (
  <div className="group relative inline-flex">
    {children}
    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-3 bg-card border border-white/10 rounded-xl shadow-2xl text-xs text-text-muted hidden group-hover:block z-50 pointer-events-none backdrop-blur-md">
      {text}
    </div>
  </div>
)

// Priority bar
const PriorityBar = ({ score, label }) => {
  const colorMap = { Critical: '#ff5252', High: '#ffc107', Moderate: '#448aff', Low: '#00e676' }
  return (
    <div className="flex items-center gap-2">
      <div className="w-24 h-2 rounded-full bg-white/5 overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: colorMap[label] || COLORS.blue, boxShadow: `0 0 10px ${colorMap[label]}80` }}
          initial={{ width: 0 }}
          animate={{ width: `${score}%` }}
          transition={{ duration: 1.2, ease: "easeOut" }}
        />
      </div>
      <span className="text-xs font-mono text-text-muted w-6 text-right">{score.toFixed(0)}</span>
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
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center group-hover:scale-110 transition-transform shadow-[0_0_15px_rgba(0,230,118,0.5)]">
            <Trees size={20} className="text-background" />
          </div>
          <span className="tracking-tight">ReForest <span className="text-primary">AI</span></span>
        </a>
        
        <button onClick={() => setMobileOpen(!mobileOpen)} className="md:hidden text-text-muted p-2 hover:bg-white/5 rounded-lg transition-colors">
          {mobileOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
        
        <div className={`${mobileOpen ? 'flex' : 'hidden'} md:flex flex-col md:flex-row absolute md:static top-full left-0 w-full md:w-auto bg-[#0a0e17] md:bg-transparent p-6 md:p-0 gap-2 md:gap-1 border-b md:border-0 border-white/10 shadow-2xl md:shadow-none`}>
          {sections.map(s => (
            <a
              key={s.id}
              href={`#${s.id}`}
              onClick={() => setMobileOpen(false)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 flex items-center gap-2 ${
                active === s.id 
                  ? 'bg-primary/10 text-primary border border-primary/20' 
                  : 'text-text-muted hover:text-white hover:bg-white/5 border border-transparent'
              }`}
            >
              <s.icon size={16} className={active === s.id ? 'text-primary' : 'text-text-muted'} />
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
    <section id="dashboard" className="pt-32 pb-16 relative">
      <FadeInSection className="text-center max-w-5xl mx-auto mb-20 relative z-10">

        <h1 className="text-5xl md:text-7xl lg:text-8xl font-bold leading-[1.1] mb-6 tracking-tight">
          Reversing India's
          <br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary via-secondary to-cyan-400 glow-text">
            Deforestation Crisis
          </span>
        </h1>
        <p className="text-lg md:text-xl text-text-muted max-w-2xl mx-auto leading-relaxed">
          Analyzing 20+ years of satellite data with advanced ML to predict optimal reforestation zones and model future scenarios for India's forests.
        </p>
      </FadeInSection>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        <FadeInSection delay={0.1}>
          <SpotlightCard glowColor="rgba(255, 82, 82, 0.15)" className="h-full">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-danger/10 flex items-center justify-center border border-danger/20"><AlertTriangle size={20} className="text-danger" /></div>
              <span className="text-sm font-semibold text-text-muted uppercase tracking-wider">Total Forest Loss</span>
            </div>
            <div className="text-4xl md:text-5xl font-display font-bold text-danger glow-danger mb-2"><AnimatedNumber value={o.total_loss_ha} suffix=" ha" /></div>
            <div className="text-sm text-text-muted">Since 2001 · GFW Data (&gt;30% canopy)</div>
          </SpotlightCard>
        </FadeInSection>

        <FadeInSection delay={0.2}>
          <SpotlightCard glowColor="rgba(255, 193, 7, 0.15)" className="h-full">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-warning/10 flex items-center justify-center border border-warning/20"><Flame size={20} className="text-warning" /></div>
              <span className="text-sm font-semibold text-text-muted uppercase tracking-wider">Total Annual Emissions</span>
            </div>
            <div className="text-4xl md:text-5xl font-display font-bold text-warning mb-2">~3.19B t</div>
            <div className="text-sm text-text-muted">CO₂ only (2024) · Up to 4.37B t total GHG</div>
          </SpotlightCard>
        </FadeInSection>

        <FadeInSection delay={0.3}>
          <SpotlightCard glowColor="rgba(0, 230, 118, 0.15)" className="h-full border-primary/30 bg-primary/5">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-primary/20 flex items-center justify-center border border-primary/30"><Trees size={20} className="text-primary" /></div>
              <span className="text-sm font-semibold text-primary uppercase tracking-wider">Reforestation Target</span>
            </div>
            <div className="text-4xl md:text-5xl font-display font-bold text-primary glow-text mb-2"><AnimatedNumber value={o.total_projected_loss_avoidance_ha} suffix=" ha" /></div>
            <div className="text-sm text-primary/80">AI-recommended intervention area</div>
          </SpotlightCard>
        </FadeInSection>

        <FadeInSection delay={0.4}>
          <SpotlightCard className="h-full">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center border border-white/10"><ShieldCheck size={20} className="text-white" /></div>
                <span className="text-sm font-semibold text-text-muted uppercase tracking-wider">Official Cover</span>
              </div>
              <Tooltip text={`ISFR ${o.isfr_report_year}: Forest cover ${o.isfr_forest_cover_pct}% (${(o.isfr_forest_area_sqkm/1000).toFixed(0)}K km²) + Tree cover ${o.isfr_tree_cover_pct}% = ${o.isfr_total_cover_pct}% total. GFW measures only dense canopy (>30%).`}>
                <Info size={16} className="text-text-muted cursor-help hover:text-white transition-colors" />
              </Tooltip>
            </div>
            <div className="text-4xl md:text-5xl font-display font-bold text-white mb-2"><AnimatedNumber value={o.isfr_total_cover_pct} suffix="%" /></div>
            <div className="text-sm text-text-muted">ISFR {o.isfr_report_year} · Forest + Tree Cover</div>
          </SpotlightCard>
        </FadeInSection>
      </div>

      {/* Primary Forest Loss Context */}
      <FadeInSection delay={0.5} className="mt-8">
        <div className="p-5 rounded-2xl bg-white/[0.03] border border-white/10 text-center">
          <p className="text-sm text-text-muted leading-relaxed">
            Between 2002 and 2025, India lost approximately <span className="text-white font-semibold">370 kha</span> of humid primary forest using the &gt;30% canopy parameter, accounting for <span className="text-white font-semibold">15%</span> of its total tree cover loss in that timeframe.
            <a href="https://www.globalnaturewatch.org/dashboards/country/IND/?category=forest-change" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline ml-1">[Source]</a>
          </p>
        </div>
      </FadeInSection>
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
    <section id="past" className="py-24 relative">
      <SectionHeader icon={BarChart3} label="Historical Analysis" title="Two Decades of Forest Loss" subtitle="India's deforestation trajectory from 2001 to 2020 — patterns, peaks, and the compounding crisis." color="text-danger" />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <FadeInSection>
          <SpotlightCard glowColor="rgba(255, 82, 82, 0.1)">
            <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-3">
              <div className="p-2 bg-danger/10 rounded-lg"><BarChart3 size={20} className="text-danger" /></div>
              Annual Tree Cover Loss
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
                  opacity: 0.9,
                  line: { width: 0 }
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
          <SpotlightCard glowColor="rgba(255, 82, 82, 0.1)">
            <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-3">
              <div className="p-2 bg-danger/10 rounded-lg"><TrendingUp size={20} className="text-danger" /></div>
              Cumulative Loss
            </h3>
            <Plot
              data={[{
                x: t.years, y: t.cumulative_loss_ha, type: 'scatter', mode: 'lines+markers',
                fill: 'tozeroy', fillcolor: 'rgba(255, 82, 82, 0.1)',
                line: { color: COLORS.red, width: 3, shape: 'spline' },
                marker: { size: 6, color: COLORS.red, symbol: 'circle' },
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
        <SpotlightCard glowColor="rgba(255, 193, 7, 0.1)">
          <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-3">
            <div className="p-2 bg-warning/10 rounded-lg"><Flame size={20} className="text-warning" /></div>
            CO₂ Emissions from Deforestation
          </h3>
          <Plot
            data={[{
              x: t.years, y: t.annual_emissions_Mg, type: 'scatter', mode: 'lines+markers',
              fill: 'tozeroy', fillcolor: 'rgba(255, 193, 7, 0.08)',
              line: { color: COLORS.warning, width: 3, shape: 'spline' },
              marker: { size: 6, color: COLORS.warning },
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
            <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-3">
              <div className="p-2 bg-purple-500/10 rounded-lg"><Layers size={20} className="text-purple-400" /></div>
              State-wise Loss Heatmap (Top 15)
            </h3>
            <Plot
              data={[{
                z: states.slice(0, 15).map(s => s.annual_losses),
                x: Array.from({ length: 20 }, (_, i) => 2001 + i),
                y: states.slice(0, 15).map(s => s.name),
                type: 'heatmap',
                colorscale: [[0, '#060910'], [0.2, '#0f291e'], [0.4, '#1b5e3a'], [0.6, '#d97706'], [0.8, '#dc2626'], [1, '#991b1b']],
                hovertemplate: '<b>%{y}</b> (%{x})<br>Loss: %{z:,.0f} ha<extra></extra>',
                colorbar: { title: { text: 'Loss (ha)', font: { color: '#94a3b8' } }, tickfont: { color: '#94a3b8' }, thickness: 15 },
              }]}
              layout={{ ...PLOTLY_LAYOUT, margin: { t: 20, r: 80, b: 50, l: 160 }, xaxis: { ...PLOTLY_LAYOUT.xaxis, dtick: 2 }, yaxis: { ...PLOTLY_LAYOUT.yaxis, autorange: 'reversed' }, height: 500 }}
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
    <section id="present" className="py-24 relative">
      <div className="absolute top-1/2 -right-64 w-[600px] h-[600px] bg-cyan-400/5 rounded-full blur-[120px] pointer-events-none" />
      <SectionHeader icon={Activity} label="Current Status" title="Where India Stands Today" subtitle="Vulnerability analysis of 36 states and union territories using multi-factor scoring." color="text-cyan-400" />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-10">
        {[
          { label: 'Dense Forest Cover', value: `${o.forest_cover_pct_current}%`, detail: 'GFW >30% canopy threshold', color: 'text-primary', icon: Trees },
          { label: 'Net Forest Loss', value: fmtHa(o.net_loss_ha), detail: 'Total loss minus total gain', color: 'text-danger', icon: TrendingDown },
          { label: 'Avg. Annual Loss', value: fmtHa(o.avg_annual_loss_ha), detail: 'Per year over study period', color: 'text-warning', icon: Activity },
        ].map((item, i) => (
          <FadeInSection key={i} delay={i * 0.1}>
            <SpotlightCard className="h-full border-t-4" style={{ borderTopColor: COLORS[item.color.replace('text-', '')] || COLORS.white }}>
              <div className="flex items-center gap-2 mb-3">
                <div className={`p-1.5 rounded-md bg-white/5`}><item.icon size={16} className={item.color} /></div>
                <span className="text-xs font-semibold uppercase tracking-wider text-text-muted">{item.label}</span>
              </div>
              <div className={`text-3xl font-display font-bold ${item.color} mb-1`}>{item.value}</div>
              <div className="text-xs text-text-muted">{item.detail}</div>
            </SpotlightCard>
          </FadeInSection>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <FadeInSection>
          <SpotlightCard className="h-full">
            <h3 className="text-xl font-bold text-white mb-6">Top 10 Most Affected States</h3>
            <Plot
              data={[{
                y: top10.map(s => s.name).reverse(),
                x: top10.map(s => s.total_loss_ha).reverse(),
                type: 'bar', orientation: 'h',
                marker: { color: top10.map((_, i) => i < 3 ? COLORS.red : i < 6 ? COLORS.warning : COLORS.teal).reverse(), opacity: 0.9, line: {width: 0} },
                hovertemplate: '<b>%{y}</b><br>Total Loss: %{x:,.0f} ha<extra></extra>',
              }]}
              layout={{ ...PLOTLY_LAYOUT, margin: { t: 10, r: 20, b: 50, l: 160 }, xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'Total Loss (ha)' }, height: 450 }}
              config={PLOTLY_CONFIG}
              className="w-full"
            />
          </SpotlightCard>
        </FadeInSection>

        <FadeInSection delay={0.15}>
          <SpotlightCard className="h-full max-h-[550px] flex flex-col p-0 overflow-hidden">
            <div className="p-6 pb-4 border-b border-white/10 shrink-0">
              <h3 className="text-xl font-bold text-white">State Rankings</h3>
            </div>
            <div className="overflow-y-auto flex-grow p-6 pt-0 custom-scrollbar">
              <table className="w-full text-sm mt-4">
                <thead>
                  <tr className="text-left text-text-muted border-b border-white/10 uppercase text-xs tracking-wider">
                    <th className="pb-3 pr-2 font-semibold">#</th>
                    <th className="pb-3 font-semibold">State</th>
                    <th className="pb-3 text-right font-semibold">Loss</th>
                    <th className="pb-3 text-center font-semibold">Priority</th>
                    <th className="pb-3 text-right font-semibold">Score</th>
                  </tr>
                </thead>
                <tbody>
                  {states.map((s, i) => {
                    const score = Math.round(s.vulnerability_score * 100)
                    const priorityLabel = score >= 80 ? 'Critical' : score >= 60 ? 'High' : score >= 40 ? 'Moderate' : 'Low'
                    return (
                    <tr key={s.name} className="border-b border-white/5 hover:bg-white/5">
                      <td className="py-3 pr-2">
                        <span className={`inline-flex items-center justify-center w-6 h-6 rounded-md text-xs font-bold ${i < 3 ? 'bg-danger/20 text-danger' : 'bg-white/5 text-text-muted'}`}>{i + 1}</span>
                      </td>
                      <td className="py-3 font-medium text-white">{s.name}</td>
                      <td className="py-3 text-right text-xs font-mono" style={{ color: s.total_loss_ha > 50000 ? COLORS.red : COLORS.white }}>{fmtHa(s.total_loss_ha)}</td>
                      <td className="py-3 text-center">
                        <span className={`${badgeClass(priorityLabel)} text-[10px] px-2 py-1 rounded-full font-bold uppercase tracking-wide`}>{priorityLabel}</span>
                      </td>
                      <td className="py-3 text-right flex justify-end"><PriorityBar score={score} label={priorityLabel} /></td>
                    </tr>
                  )})}
                </tbody>
              </table>
            </div>
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
    <section id="future" className="py-24 relative">
      <SectionHeader icon={TrendingUp} label="Predictive Modeling" title="Two Possible Futures" subtitle="What happens if we do nothing vs. what's possible with strategic reforestation." color="text-purple-400" />

      {/* Scenario comparison cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-10">
        <FadeInSection>
          <SpotlightCard className="border-danger/30 bg-gradient-to-br from-card to-danger/5" glowColor="rgba(255, 82, 82, 0.1)">
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 rounded-xl bg-danger/15 flex items-center justify-center border border-danger/30 shadow-[0_0_15px_rgba(255,82,82,0.3)]"><TrendingUp size={24} className="text-danger" /></div>
              <div>
                <h3 className="text-2xl font-bold text-white">Business as Usual</h3>
                <p className="text-sm text-text-muted">If deforestation continues unchecked</p>
              </div>
            </div>
            <div className="space-y-5 bg-black/20 p-5 rounded-xl border border-white/5">
              <div className="flex justify-between items-center pb-4 border-b border-white/5">
                <span className="text-sm font-medium text-text-muted uppercase tracking-wider">Additional Loss (2021-2030)</span>
                <span className="text-xl font-bold text-danger glow-danger">{fmtHa(bauTotal)}</span>
              </div>
              <div className="flex justify-between items-center pb-4 border-b border-white/5">
                <span className="text-sm font-medium text-text-muted uppercase tracking-wider">Projected CO₂ Emissions</span>
                <span className="text-xl font-bold text-warning">{fmt(bauEmTotal)} Mg</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-text-muted uppercase tracking-wider">Cumulative Loss by 2030</span>
                <span className="text-xl font-bold text-danger">{fmtHa(cf.bau_scenario.cumulative_loss_ha[cf.bau_scenario.cumulative_loss_ha.length - 1])}</span>
              </div>
            </div>
          </SpotlightCard>
        </FadeInSection>

        <FadeInSection delay={0.15}>
          <SpotlightCard className="border-primary/30 bg-gradient-to-br from-card to-primary/5" glowColor="rgba(0, 230, 118, 0.1)">
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 rounded-xl bg-primary/15 flex items-center justify-center border border-primary/30 shadow-[0_0_15px_rgba(0,230,118,0.3)]"><Leaf size={24} className="text-primary" /></div>
              <div>
                <h3 className="text-2xl font-bold text-white">With Reforestation</h3>
                <p className="text-sm text-text-muted">Active intervention + strategic planting</p>
              </div>
            </div>
            <div className="space-y-5 bg-black/20 p-5 rounded-xl border border-white/5">
              <div className="flex justify-between items-center pb-4 border-b border-white/5">
                <span className="text-sm font-medium text-text-muted uppercase tracking-wider">Reduced Loss (2021-2030)</span>
                <span className="text-xl font-bold text-primary glow-text">{fmtHa(refTotal)}</span>
              </div>
              <div className="flex justify-between items-center pb-4 border-b border-white/5">
                <span className="text-sm font-medium text-text-muted uppercase tracking-wider">Carbon Saved</span>
                <span className="text-xl font-bold text-primary">{fmt(Math.max(0, bauEmTotal - refEmTotal))} Mg</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-text-muted uppercase tracking-wider">Cumulative Loss by 2030</span>
                <span className="text-xl font-bold text-primary">{fmtHa(cf.reforestation_scenario.cumulative_loss_ha[cf.reforestation_scenario.cumulative_loss_ha.length - 1])}</span>
              </div>
            </div>
          </SpotlightCard>
        </FadeInSection>
      </div>


      {/* Carbon savings */}
      <FadeInSection delay={0.2} className="mt-6">
        <SpotlightCard className="bg-gradient-to-r from-card via-primary/5 to-card border-primary/20">
          <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-3">
            <div className="p-2 bg-primary/10 rounded-lg"><Leaf size={20} className="text-primary" /></div>
            Cumulative Carbon Savings from Reforestation
          </h3>
          <Plot
            data={[{
              x: cf.reforestation_scenario.years,
              y: cf.reforestation_scenario.carbon_saved_Mg || cf.reforestation_scenario.years.map(() => 0),
              type: 'bar',
              marker: { color: 'rgba(0, 230, 118, 0.8)', line: { width: 0 }, opacity: 0.9 },
              hovertemplate: '<b>%{x}</b><br>Carbon Saved: %{y:,.0f} Mg CO₂<extra></extra>',
            }]}
            layout={{ ...PLOTLY_LAYOUT, xaxis: { ...PLOTLY_LAYOUT.xaxis, dtick: 1 }, yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Cumulative Saved (Mg CO₂)' }, height: 320 }}
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
const MODEL_CATEGORIES = {
  'Linear Regression': {
    icon: '📏',
    color: 'rgba(68, 138, 255, 0.15)',
    models: ['Ridge', 'Bayesian Ridge'],
    description: 'Linear models with regularization'
  },
  'Boosting Algorithms': {
    icon: '🚀',
    color: 'rgba(255, 193, 7, 0.15)',
    models: ['Gradient Boosting', 'XGBoost', 'AdaBoost', 'AdaBoost (Deep Trees)', 'AdaBoost (Linear Loss)', 'HistGradientBoosting', 'LightGBM'],
    description: 'Sequential learners that correct previous errors'
  },
  'Ensemble Learning': {
    icon: '🌲',
    color: 'rgba(0, 230, 118, 0.15)',
    models: ['Random Forest', 'Extra Trees', 'Bagging Trees'],
    description: 'Parallel learners combined for robust predictions'
  },
}

const MLInsights = ({ data }) => {
  const mp = data.modelPerformance
  if (!mp) return null
  const models = mp.models || mp
  const featureImp = data.modelPerformance?.feature_importance || {}
  const smote = data.modelPerformance?.smote_analysis || {}

  const [selectedModel, setSelectedModel] = useState('Random Forest')
  const [activeCategory, setActiveCategory] = useState('all')

  // Build categorized model entries — only keep models in our 3 categories
  const allowedModels = new Set(Object.values(MODEL_CATEGORIES).flatMap(c => c.models))
  const modelEntries = Object.entries(models)
    .filter(([k, v]) => !['feature_importance', 'smote_analysis', '__selected_model__'].includes(k) && v.test_r2 > 0 && allowedModels.has(k))
    .sort((a, b) => b[1].test_r2 - a[1].test_r2)

  const filteredEntries = activeCategory === 'all'
    ? modelEntries
    : modelEntries.filter(([name]) => MODEL_CATEGORIES[activeCategory]?.models.includes(name))

  let bestName = modelEntries.length > 0 ? modelEntries[0][0] : ''

  useEffect(() => {
    if (Object.keys(featureImp).length > 0 && !featureImp[selectedModel] && bestName) {
      if (featureImp[bestName]) setSelectedModel(bestName)
      else setSelectedModel(Object.keys(featureImp)[0])
    }
  }, [featureImp, bestName])

  const getModelCategory = (name) => {
    for (const [cat, info] of Object.entries(MODEL_CATEGORIES)) {
      if (info.models.includes(name)) return cat
    }
    return null
  }

  const catColors = { 'Linear Regression': '#448aff', 'Boosting Algorithms': '#ffc107', 'Ensemble Learning': '#00e676' }
  const icons = { 'Random Forest': '🌲', 'XGBoost': '🚀', 'AdaBoost': '🔥', 'AdaBoost (Deep Trees)': '🔥', 'AdaBoost (Linear Loss)': '🔥', Ridge: '📏', 'Extra Trees': '🌳', 'Gradient Boosting': '📈', 'HistGradientBoosting': '📊', 'Bagging Trees': '🎒', 'Bayesian Ridge': '📐', 'LightGBM': '💡' }

  return (
    <section id="ai" className="py-24 relative">
      <div className="absolute top-0 -left-64 w-[500px] h-[500px] bg-blue-500/5 rounded-full blur-[120px] pointer-events-none" />
      <SectionHeader icon={Brain} label="Machine Learning" title="AI Model Performance" subtitle="Models trained with regularization and cross-validation across 3 algorithm families. SMOTE applied to balance priority classification." color="text-blue-400" />

      {/* Category Filter Tabs */}
      <div className="flex flex-wrap gap-2 mb-8">
        <button
          onClick={() => setActiveCategory('all')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeCategory === 'all' ? 'bg-white/10 text-white border border-white/20' : 'text-text-muted hover:text-white hover:bg-white/5 border border-transparent'}`}
        >All Models ({modelEntries.length})</button>
        {Object.entries(MODEL_CATEGORIES).map(([cat, info]) => {
          const count = modelEntries.filter(([n]) => info.models.includes(n)).length
          return (
            <button
              key={cat}
              onClick={() => setActiveCategory(cat)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2 ${activeCategory === cat ? 'bg-white/10 text-white border border-white/20' : 'text-text-muted hover:text-white hover:bg-white/5 border border-transparent'}`}
            >
              <span>{info.icon}</span> {cat} ({count})
            </button>
          )
        })}
      </div>

      {/* Model Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 mb-12">
        {filteredEntries.map(([name, m], i) => {
          const cat = getModelCategory(name)
          return (
          <FadeInSection key={name} delay={i * 0.05}>
            <SpotlightCard className={`h-full ${name === bestName ? 'border-primary/40 bg-primary/5 shadow-[0_0_20px_rgba(0,230,118,0.15)]' : ''}`} glowColor={name === bestName ? 'rgba(0,230,118,0.15)' : undefined}>
              <div className="flex items-center justify-between mb-5">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center text-xl border border-white/10">{icons[name] || '🤖'}</div>
                  <div>
                    <span className="font-bold text-white text-sm block">{name}</span>
                    {cat && <span className="text-[10px] uppercase tracking-wider" style={{ color: catColors[cat] }}>{cat}</span>}
                  </div>
                </div>
                {name === bestName && <span className="text-[10px] px-2 py-1 bg-primary/20 text-primary rounded-full font-bold uppercase tracking-wider border border-primary/30 animate-pulse">Best</span>}
              </div>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-xs mb-1.5 font-medium uppercase tracking-wider">
                    <span className="text-text-muted">Test R²</span>
                    <span className="font-mono font-bold" style={{ color: m.test_r2 > 0.9 ? COLORS.green : m.test_r2 > 0.7 ? COLORS.warning : COLORS.red }}>{m.test_r2.toFixed(4)}</span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-white/5 overflow-hidden shadow-inner">
                    <motion.div className="h-full rounded-full bg-gradient-to-r from-blue-500 to-primary" initial={{ width: 0 }} animate={{ width: `${Math.max(0, m.test_r2) * 100}%` }} transition={{ duration: 1.5, delay: i * 0.1 }} />
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-2 pt-2 border-t border-white/5">
                  <div>
                    <div className="text-[10px] text-text-muted uppercase tracking-wider mb-1">MAE</div>
                    <div className="font-mono text-white text-sm">{fmt(m.test_mae)}</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-text-muted uppercase tracking-wider mb-1">Overfit Gap</div>
                    <div className="font-mono text-sm" style={{ color: Math.abs(m.overfit_gap) > 0.1 ? COLORS.red : COLORS.white }}>{m.overfit_gap !== undefined ? m.overfit_gap.toFixed(4) : '—'}</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-text-muted uppercase tracking-wider mb-1">Train R²</div>
                    <div className="font-mono text-white text-sm">{m.train_r2.toFixed(4)}</div>
                  </div>
                </div>

                {/* Per-model SMOTE comparison */}
                {smote.before_smote && (
                  <div className="pt-3 border-t border-white/5">
                    <div className="text-[10px] text-purple-400 uppercase tracking-wider mb-2 font-bold">SMOTE Impact</div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="p-2 bg-black/20 rounded-lg text-center">
                        <div className="text-[9px] text-text-muted uppercase mb-1">Before</div>
                        <div className="text-sm font-bold text-white">{(smote.before_smote.accuracy * 100).toFixed(1)}%</div>
                      </div>
                      <div className="p-2 bg-purple-900/20 rounded-lg text-center border border-purple-500/20">
                        <div className="text-[9px] text-purple-400 uppercase mb-1">After</div>
                        <div className="text-sm font-bold text-purple-400">{(smote.after_smote.accuracy * 100).toFixed(1)}%</div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </SpotlightCard>
          </FadeInSection>
        )})}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Model Comparison Chart */}
        <FadeInSection>
          <SpotlightCard className="h-full">
            <h3 className="text-xl font-bold text-white mb-6">Train vs Test R² Comparison</h3>
            <Plot
              data={[
                { x: filteredEntries.map(([n]) => n), y: filteredEntries.map(([, m]) => Math.max(0, m.train_r2)), type: 'bar', name: 'Train R²', marker: { color: 'rgba(68, 138, 255, 0.7)' } },
                { x: filteredEntries.map(([n]) => n), y: filteredEntries.map(([, m]) => Math.max(0, m.test_r2)), type: 'bar', name: 'Test R²', marker: { color: 'rgba(0, 230, 118, 0.9)' } },
              ]}
              layout={{ ...PLOTLY_LAYOUT, barmode: 'group', yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'R² Score', range: [0, 1.05] }, legend: { ...PLOTLY_LAYOUT.legend, x: 0, y: 1.15, orientation: 'h' }, height: 400 }}
              config={PLOTLY_CONFIG}
              className="w-full"
            />
          </SpotlightCard>
        </FadeInSection>

        {/* Feature Importance */}
        {Object.keys(featureImp).length > 0 && (
          <FadeInSection delay={0.15}>
            <SpotlightCard className="h-full">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-6 gap-4">
                <h3 className="text-xl font-bold text-white">Feature Importance</h3>
                <div className="relative">
                  <select 
                    value={selectedModel} 
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="appearance-none bg-white/5 border border-white/10 text-white text-sm rounded-lg px-4 py-2 pr-10 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all cursor-pointer hover:bg-white/10"
                  >
                    {Object.keys(featureImp).filter(m => allowedModels.has(m)).map(m => (
                      <option key={m} value={m} className="bg-card text-white">{m}</option>
                    ))}
                  </select>
                  <ChevronDown size={16} className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted pointer-events-none" />
                </div>
              </div>
              
              {featureImp[selectedModel] ? (
                <Plot
                  data={[{
                    y: featureImp[selectedModel].map(f => f.feature.replace(/_/g, ' ')).reverse(),
                    x: featureImp[selectedModel].map(f => f.importance).reverse(),
                    type: 'bar', orientation: 'h',
                    marker: { color: 'rgba(179, 136, 255, 0.8)', line: { width: 0 } },
                    hovertemplate: '<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
                  }]}
                  layout={{ ...PLOTLY_LAYOUT, margin: { t: 10, r: 20, b: 50, l: 180 }, xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'Relative Importance' }, height: 350 }}
                  config={PLOTLY_CONFIG}
                  className="w-full"
                />
              ) : (
                <div className="h-[350px] flex items-center justify-center text-text-muted">No feature importance data for this model</div>
              )}
            </SpotlightCard>
          </FadeInSection>
        )}
      </div>

      {/* SMOTE Before/After Summary */}
      {smote.before_smote && (
        <FadeInSection>
          <div className="p-8 rounded-2xl bg-gradient-to-r from-card via-[#1e1a3b] to-card border border-purple-500/20 shadow-2xl relative overflow-hidden">
            <div className="absolute top-0 right-0 w-64 h-64 bg-purple-500/10 rounded-full blur-[80px]" />
            <h3 className="text-2xl font-bold text-white mb-8 flex items-center gap-3">
              <div className="p-2 bg-purple-500/20 rounded-xl"><Zap size={24} className="text-purple-400" /></div>
              SMOTE — Class Balance Overview
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] gap-8 items-center">
              <div className="text-center p-6 bg-black/20 rounded-xl border border-white/5">
                <div className="text-sm uppercase tracking-widest text-text-muted font-bold mb-4">Before SMOTE</div>
                <div className="text-5xl font-display font-bold text-white mb-2">{(smote.before_smote.accuracy * 100).toFixed(1)}<span className="text-2xl text-text-muted">%</span></div>
                <div className="text-sm text-text-muted font-medium mb-6">Classification Accuracy</div>
                <div className="grid grid-cols-3 gap-2 text-xs border-t border-white/10 pt-4">
                  <div><div className="text-text-muted mb-1">Low</div><div className="font-bold text-white">{smote.before_smote.class_distribution.Low || 0}</div></div>
                  <div><div className="text-text-muted mb-1">Med</div><div className="font-bold text-white">{smote.before_smote.class_distribution.Medium || 0}</div></div>
                  <div><div className="text-text-muted mb-1">High</div><div className="font-bold text-white">{smote.before_smote.class_distribution.High || 0}</div></div>
                </div>
              </div>

              <div className="flex justify-center hidden md:flex">
                <div className="w-12 h-12 rounded-full bg-gradient-to-r from-white/10 to-purple-500/30 flex items-center justify-center border border-purple-500/30 animate-pulse">
                  <ArrowRight size={20} className="text-purple-300" />
                </div>
              </div>

              <div className="text-center p-6 bg-purple-900/20 rounded-xl border border-purple-500/30 shadow-[0_0_30px_rgba(168,85,247,0.15)] relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-b from-purple-500/5 to-transparent pointer-events-none" />
                <div className="text-sm uppercase tracking-widest text-purple-400 font-bold mb-4">After SMOTE</div>
                <div className="text-5xl font-display font-bold text-purple-400 mb-2 glow-text" style={{ textShadow: '0 0 20px rgba(168,85,247,0.4)' }}>{(smote.after_smote.accuracy * 100).toFixed(1)}<span className="text-2xl text-purple-400/50">%</span></div>
                <div className="text-sm text-purple-300/80 font-medium mb-6">Classification Accuracy</div>
                <div className="grid grid-cols-3 gap-2 text-xs border-t border-purple-500/20 pt-4 text-purple-100">
                  <div><div className="text-purple-300/60 mb-1">Low</div><div className="font-bold">{smote.after_smote.class_distribution.Low || 0}</div></div>
                  <div><div className="text-purple-300/60 mb-1">Med</div><div className="font-bold">{smote.after_smote.class_distribution.Medium || 0}</div></div>
                  <div><div className="text-purple-300/60 mb-1">High</div><div className="font-bold">{smote.after_smote.class_distribution.High || 0}</div></div>
                </div>
              </div>
            </div>
          </div>
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
  if (!plan || !o) return null

  // Filter out states that don't need reforestation and sort
  const sorted = [...plan].filter(s => s.projected_loss_avoidance_ha > 0).sort((a, b) => b.priority_score - a.priority_score)
  const priorityColors = { Critical: COLORS.red, High: COLORS.warning, Moderate: COLORS.blue, Low: COLORS.green }

  return (
    <section id="plan" className="py-24 relative">
      <div className="absolute bottom-0 right-0 w-[800px] h-[400px] bg-primary/5 rounded-full blur-[150px] pointer-events-none" />
      <SectionHeader icon={Target} label="Action Plan" title="Strategic Reforestation Blueprint" subtitle="AI-prioritized regions requiring immediate intervention to maximize carbon sequestration and reverse damage." color="text-primary" />

      {/* Summary cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
        {[
          { label: 'Total Avoidance Target', value: fmtHa(o.total_projected_loss_avoidance_ha), detail: `Across ${o.num_states} states by 2030`, color: 'text-primary', icon: Trees },
          { label: 'Carbon Savings Potential', value: `${fmt(o.total_carbon_savings_Mg)} Mg`, detail: 'CO₂ prevented from atmosphere', color: 'text-secondary', icon: Leaf },
          { label: 'Urgent Priority States', value: `${o.num_critical_states + o.num_high_states}`, detail: 'Critical + High priority zones', color: 'text-danger', icon: AlertTriangle },
        ].map((item, i) => (
          <FadeInSection key={i} delay={i * 0.1}>
            <SpotlightCard className="h-full bg-gradient-to-br from-card to-white/[0.02]">
              <div className="flex items-center gap-3 mb-4">
                <div className={`p-2 rounded-xl bg-white/5 border border-white/10`}><item.icon size={20} className={item.color} /></div>
                <span className="text-sm font-bold uppercase tracking-wider text-text-muted">{item.label}</span>
              </div>
              <div className={`text-4xl font-display font-bold ${item.color} mb-2`}>{item.value}</div>
              <div className="text-sm text-text-muted">{item.detail}</div>
            </SpotlightCard>
          </FadeInSection>
        ))}
      </div>

      {/* Priority Chart */}
      <FadeInSection className="mb-10">
        <SpotlightCard>
          <h3 className="text-xl font-bold text-white mb-6">Priority Scores by State</h3>
          <Plot
            data={[{
              y: sorted.map(s => s.name).reverse(),
              x: sorted.map(s => s.priority_score).reverse(),
              type: 'bar', orientation: 'h',
              marker: { color: sorted.map(s => priorityColors[s.priority_label] || COLORS.blue).reverse(), opacity: 0.9, line: {width: 0} },
              hovertemplate: '<b>%{y}</b><br>Score: %{x:.1f}/100<extra></extra>',
            }]}
            layout={{ ...PLOTLY_LAYOUT, margin: { t: 10, r: 20, b: 50, l: 160 }, xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'Priority Score (0-100)', range: [0, 100] }, height: Math.max(400, sorted.length * 30) }}
            config={PLOTLY_CONFIG}
            className="w-full"
          />
        </SpotlightCard>
      </FadeInSection>

      {/* Reforestation Table */}
      <FadeInSection>
        <SpotlightCard className="p-0 overflow-hidden">
          <div className="p-6 border-b border-white/10 bg-black/20 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <h3 className="text-xl font-bold text-white">Detailed Intervention Requirements</h3>
            <div className="flex items-center gap-2 text-sm text-text-muted bg-white/5 px-3 py-1.5 rounded-lg border border-white/10">
              <Filter size={14} /> Showing {sorted.length} states requiring action
            </div>
          </div>
          <div className="overflow-x-auto custom-scrollbar">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-text-muted border-b border-white/10 bg-black/40 uppercase tracking-wider text-[11px] font-bold">
                  <th className="py-4 pl-6 pr-3">#</th>
                  <th className="py-4">State</th>
                  <th className="py-4 text-center">Priority</th>
                  <th className="py-4 text-right">Score</th>
                  <th className="py-4 text-right">Loss Avoidance Target</th>
                  <th className="py-4 text-right">Carbon Savings</th>
                  <th className="py-4 text-right pr-6">Current Loss Rate</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((s, i) => (
                  <tr key={s.name} className="border-b border-white/5 hover:bg-white/5 group">
                    <td className="py-4 pl-6 pr-3">
                      <span className={`inline-flex items-center justify-center w-7 h-7 rounded-lg text-xs font-bold ${i < 3 ? 'bg-danger/20 text-danger border border-danger/30' : 'bg-white/5 text-text-muted'}`}>{i + 1}</span>
                    </td>
                    <td className="py-4 font-bold text-white text-base">{s.name}</td>
                    <td className="py-4 text-center">
                      <span className={`${badgeClass(s.priority_label)} text-[10px] px-3 py-1 rounded-full font-bold uppercase tracking-wider shadow-sm`}>{s.priority_label}</span>
                    </td>
                    <td className="py-4 text-right flex justify-end items-center h-full pt-6"><PriorityBar score={s.priority_score} label={s.priority_label} /></td>
                    <td className="py-4 text-right font-bold text-primary text-base glow-text group-hover:scale-105 transition-transform origin-right">{fmtHa(Math.max(0, s.projected_loss_avoidance_ha))}</td>
                    <td className="py-4 text-right text-text-muted font-medium">{fmt(Math.max(0, s.carbon_savings_Mg))} Mg</td>
                    <td className="py-4 text-right text-text-muted pr-6">{fmtHa(s.current_loss_rate)}/yr</td>
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
  <footer className="border-t border-white/10 py-16 mt-16 bg-black/40 relative overflow-hidden">
    <div className="absolute top-0 left-1/2 -translate-x-1/2 w-1/2 h-px bg-gradient-to-r from-transparent via-primary/50 to-transparent" />
    <div className="max-w-7xl mx-auto px-6 text-center relative z-10">
      <div className="flex items-center justify-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center shadow-[0_0_15px_rgba(0,230,118,0.3)]">
          <Trees size={20} className="text-background" />
        </div>
        <span className="font-display font-bold text-2xl text-white tracking-tight">ReForest AI</span>
      </div>
      <p className="text-base text-text-muted mb-4 max-w-xl mx-auto">
        Advanced machine learning models predicting deforestation patterns and optimizing reforestation strategies for India.
      </p>
      <div className="flex items-center justify-center gap-4 text-sm text-text-muted/60 mb-8">
        <span>Data: Global Forest Watch</span>
        <span>&bull;</span>
        <span>ISFR 2023</span>
      </div>
      <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 text-xs text-text-muted/80">
        <Brain size={14} className="text-primary" />
        Built with Ensemble Learning, SMOTE, and Time-Series Forecasting
      </div>
    </div>
  </footer>
)

// ============================================================================
// MAIN APP
// ============================================================================
function App() {
  const [appData, setAppData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [overview, timeline, states, predictions, modelPerf, reforestPlan] = await Promise.all([
          fetch('/api/overview').then(r => { if (!r.ok) throw new Error('Overview failed'); return r.json() }),
          fetch('/api/timeline').then(r => { if (!r.ok) throw new Error('Timeline failed'); return r.json() }),
          fetch('/api/states').then(r => { if (!r.ok) throw new Error('States failed'); return r.json() }),
          fetch('/api/predictions').then(r => { if (!r.ok) throw new Error('Predictions failed'); return r.json() }),
          fetch('/api/model-performance').then(r => { if (!r.ok) throw new Error('Model perf failed'); return r.json() }),
          fetch('/api/reforestation-plan').then(r => { if (!r.ok) throw new Error('Plan failed'); return r.json() }),
        ])
        setAppData({ overview, timeline, states, predictions, modelPerformance: modelPerf, reforestationPlan: reforestPlan })
      } catch (err) {
        console.error("Failed to fetch data:", err)
        setError(err.message || String(err))
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background text-danger flex-col gap-4">
        <AlertTriangle size={48} />
        <h2 className="text-2xl font-bold">Failed to load dashboard data</h2>
        <p className="text-white/70">{error}</p>
        <button onClick={() => window.location.reload()} className="px-4 py-2 bg-primary/20 text-primary rounded-lg mt-4">Retry</button>
      </div>
    )
  }

  if (loading || !appData) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="flex flex-col items-center gap-8"
        >
          <div className="relative">
            <div className="absolute inset-0 bg-primary/20 rounded-full blur-xl animate-pulse" />
            <div className="w-24 h-24 border-4 border-white/10 border-t-primary rounded-full animate-spin relative z-10" />
            <Trees size={32} className="text-primary absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-20" />
          </div>
          <div className="text-center">
            <div className="text-2xl font-display font-bold text-white mb-2">Loading Reforestation Intelligence</div>
            <div className="text-sm text-text-muted flex items-center justify-center gap-2">
              <span className="w-2 h-2 bg-primary rounded-full animate-ping" />
              Analyzing satellite data & ML predictions...
            </div>
          </div>
        </motion.div>
      </div>
    )
  }

  return (
    <div className="min-h-screen relative overflow-x-hidden selection:bg-primary/30 selection:text-white">
      {/* Background gradient orbs */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
        <div className="absolute top-[-10%] left-[-10%] w-[600px] h-[600px] bg-primary/5 rounded-full blur-[150px]" />
        <div className="absolute top-[40%] right-[-5%] w-[500px] h-[500px] bg-purple-500/5 rounded-full blur-[150px]" />
        <div className="absolute bottom-[-10%] left-[20%] w-[600px] h-[600px] bg-cyan-400/5 rounded-full blur-[150px]" />
      </div>

      <Navbar />

      <main className="relative z-10 px-6 md:px-12 lg:px-24 max-w-[1400px] mx-auto">
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
