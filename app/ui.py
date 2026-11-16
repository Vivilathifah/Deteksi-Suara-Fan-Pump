"""Komponen tampilan bersama — gaya dashboard "big-screen" biru gelap."""
import itertools
import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import data as D

# ------------------------------------------------------------------ warna
BLUE, ORANGE, GREEN, YELLOW, PURPLE, PINK, CYAN, LIME = (
    "#3BA4FF", "#FF7A59", "#2EE6B6", "#FFD452", "#8B7CFF", "#FF5C8A", "#5CE1FF", "#A3E635")
PALETTE = [BLUE, ORANGE, GREEN, YELLOW, PURPLE, PINK, CYAN, LIME]
SEQ = ["#0B2A6B", "#1F5FBF", "#3BA4FF", "#2EE6B6", "#FFD452"]       # skala warna heatmap

# ------------------------------------------------------------------ grafik
# Grafik memakai tema Streamlit (theme="streamlit"), jadi warna teks & garis otomatis menyesuaikan tema terang/gelap.
# Di sini hanya latar dibuat transparan dan garis grid dibuat lembut.
GRIDC = "rgba(128,150,200,0.18)"

CSS = """
<style>
/* ===== palet: gelap (bawaan) ===== */
:root{--tx:#DDE7FF;--hd:#FFFFFF;--mu:#8FB3FF;--acc:#3BA4FF;--bd:rgba(64,140,255,.30);
  --p1:rgba(22,56,128,.55);--p2:rgba(12,34,84,.55);--glow:0 0 14px rgba(59,164,255,.6);--ins:inset 0 0 22px rgba(59,164,255,.07);
  --img:radial-gradient(1400px 700px at 50% -15%,rgba(40,100,220,.55) 0%,rgba(11,30,74,0) 60%);
  --side:linear-gradient(180deg,#0A1B44,#081638);--track:rgba(64,140,255,.12);--step:rgba(18,48,107,.45)}
/* ===== palet: terang ===== */
html[data-mt="light"]{--tx:#1E2A44;--hd:#0B1B3F;--mu:#5B6B8C;--acc:#2563EB;--bd:rgba(37,99,235,.18);
  --p1:#FFFFFF;--p2:#F7FAFF;--glow:none;--ins:0 2px 10px rgba(15,40,100,.06);
  --img:radial-gradient(1400px 600px at 50% -20%,rgba(59,130,246,.14) 0%,rgba(243,246,252,0) 60%);
  --side:linear-gradient(180deg,#FFFFFF,#EEF3FC);--track:rgba(37,99,235,.10);--step:#FFFFFF}
.stApp{background-image:var(--img)}
header[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:var(--side);border-right:1px solid var(--bd)}
.block-container{padding-top:1.6rem;max-width:1500px}
.topbar{display:flex;justify-content:space-between;align-items:center;gap:1rem;padding:.75rem 1.1rem;margin-bottom:.8rem;
  border:1px solid var(--bd);border-radius:8px;box-shadow:var(--ins);
  background:linear-gradient(90deg,color-mix(in srgb,var(--acc) 20%,transparent),transparent 65%),var(--p2)}
.tb-t{font-size:1.55rem;font-weight:700;color:var(--hd);letter-spacing:.3px;text-shadow:var(--glow)}
.tb-s{font-size:.9rem;color:var(--mu);margin-top:.15rem}
.tb-r{font-size:.82rem;color:var(--mu);text-align:right;white-space:nowrap}
.steps{display:flex;flex-wrap:wrap;gap:.4rem;align-items:center;margin:.1rem 0 1rem}
.step{padding:.25rem .7rem;border-radius:4px;font-size:.78rem;font-weight:600;color:var(--mu);border:1px solid var(--bd);background:var(--step)}
.step.on{background:var(--acc);color:#fff;border-color:var(--acc);box-shadow:0 0 12px color-mix(in srgb,var(--acc) 55%,transparent)}
.arrow{color:var(--acc);opacity:.6}
[class*="st-key-pnl"]{background:linear-gradient(180deg,var(--p1),var(--p2))!important;
  border:1px solid var(--bd)!important;border-radius:8px!important;box-shadow:var(--ins)}
.pt{font-size:.95rem;font-weight:600;color:var(--hd);border-left:3px solid var(--acc);padding-left:.55rem;margin:.1rem 0 .4rem}
[data-testid="stMetric"]{background:linear-gradient(180deg,var(--p1),var(--p2));border:1px solid var(--bd)!important;
  border-radius:8px;padding:.7rem .9rem;box-shadow:var(--ins)}
[data-testid="stMetricLabel"]{color:var(--mu)}
[data-testid="stMetricValue"]{font-weight:700;color:var(--hd);text-shadow:var(--glow)}
.kpi{padding:.7rem .95rem;border-radius:8px;border:1px solid var(--bd);height:100%;box-shadow:var(--ins);
  background:linear-gradient(180deg,var(--p1),var(--p2))}
.kpi .row{display:flex;align-items:center;justify-content:space-between;gap:.5rem}
.kpi .v{font-size:1.9rem;font-weight:600;color:var(--hd);line-height:1.15;font-variant-numeric:tabular-nums}
.kpi .l{font-size:.85rem;color:var(--mu);margin-top:.2rem}
.kpi.sm .v{font-size:1.25rem;white-space:nowrap}.kpi.sm .row{flex-wrap:wrap}
.kpi .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:.4rem}
.badge{font-size:.78rem;font-weight:700;padding:.15rem .6rem;border-radius:999px;color:#04122E;white-space:nowrap}
.rank{display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));gap:.7rem .9rem}
.rank>div{min-width:0;border-left:2px solid var(--bd);padding-left:.55rem}
.rank .n{font-size:.78rem;color:var(--mu)}.rank .v{font-size:1.5rem;color:var(--hd);font-weight:500;font-variant-numeric:tabular-nums}
.rank .s{font-size:.75rem;color:var(--mu);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bars .it{margin:.35rem 0 .7rem}.bars .h{display:flex;justify-content:space-between;font-size:.82rem;color:var(--tx)}
.bars .tr{height:10px;background:var(--track);border-radius:2px;margin-top:.25rem}
.bars .fl{height:10px;border-radius:2px;background:linear-gradient(90deg,#1F5FBF,var(--acc));box-shadow:0 0 8px color-mix(in srgb,var(--acc) 45%,transparent)}
.note{border-left:3px solid var(--acc);padding:.55rem .9rem;border-radius:4px;margin:.5rem 0 1rem;
  background:color-mix(in srgb,var(--acc) 10%,transparent);font-size:.9rem;color:var(--tx)}
.legend{font-size:.8rem;color:var(--mu);line-height:1.7}
.pd{font-size:.82rem;color:var(--mu);margin:-.25rem 0 .45rem;line-height:1.45}
.intro{font-size:.93rem;color:var(--tx);margin:-.1rem 0 .9rem;line-height:1.6;max-width:1100px}
.ins{border:1px solid color-mix(in srgb,#2EE6B6 45%,transparent);background:color-mix(in srgb,#2EE6B6 9%,transparent);
  border-radius:8px;padding:.65rem .95rem;margin:.3rem 0 1rem;font-size:.9rem;color:var(--tx);line-height:1.6}
.ins .it{font-weight:700;color:#14B88E}

/* kartu jawaban prediksi */
.vd{border:1px solid color-mix(in srgb,var(--vc) 55%,transparent);border-radius:12px;padding:1rem 1.2rem;margin:.6rem 0 1rem;
  background:linear-gradient(120deg,color-mix(in srgb,var(--vc) 20%,transparent),color-mix(in srgb,var(--vc) 4%,transparent) 70%);
  box-shadow:0 8px 30px color-mix(in srgb,var(--vc) 18%,transparent);animation:popIn .6s cubic-bezier(.2,.9,.3,1.2) both}
.vd-l{font-size:.75rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--mu)}
.vd-main{display:flex;align-items:center;gap:1rem;margin:.4rem 0 .7rem}
.vd-ic{font-size:3rem;line-height:1;animation:bob 2.6s ease-in-out infinite}
.vd-k{font-size:.9rem;color:var(--mu)}
.vd-v{font-size:2.3rem;font-weight:800;color:var(--vc);letter-spacing:.02em;line-height:1.1;text-shadow:0 0 18px color-mix(in srgb,var(--vc) 45%,transparent)}
.vd-m{font-size:.95rem;color:var(--tx)}
.vd-conf{font-size:.88rem;color:var(--tx)}
.vd-bar{height:9px;border-radius:99px;background:var(--track);margin-top:.3rem;overflow:hidden;max-width:420px}
.vd-bar>div{height:100%;border-radius:99px;animation:grow 1.2s cubic-bezier(.2,.8,.2,1) both}
.vd-true{margin-top:.7rem;font-size:.95rem;color:var(--tx);padding-top:.6rem;border-top:1px dashed var(--bd)}
@keyframes popIn{from{opacity:0;transform:scale(.94) translateY(10px)}to{opacity:1;transform:none}}
@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}

/* ===================== ANIMASI ===================== */
@keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes zoomIn{from{opacity:0;transform:scale(.97)}to{opacity:1;transform:none}}
@keyframes shine{0%{background-position:-120% 0}100%{background-position:220% 0}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 color-mix(in srgb,currentColor 55%,transparent)}50%{box-shadow:0 0 0 6px transparent}}
@keyframes grow{from{width:0}}
@keyframes glowline{0%,100%{opacity:.35}50%{opacity:1}}
@keyframes spinIn{from{opacity:0;transform:rotate(-90deg) scale(.6)}to{opacity:1;transform:none}}
.topbar{position:relative;overflow:hidden;animation:fadeUp .55s ease both}
.topbar::after{content:"";position:absolute;inset:0;pointer-events:none;
  background:linear-gradient(105deg,transparent 35%,rgba(255,255,255,.16) 50%,transparent 65%);
  background-size:250% 100%;animation:shine 6s ease-in-out infinite}
.topbar::before{content:"";position:absolute;left:0;right:0;bottom:0;height:2px;
  background:linear-gradient(90deg,transparent,var(--acc),#2EE6B6,transparent);animation:glowline 3s ease-in-out infinite}
.steps,.intro{animation:fadeUp .6s .08s ease both}
.step.on{animation:pulse 2.2s ease-in-out infinite;color:#fff}
[class*="st-key-pnl"]{animation:fadeUp .6s ease both;transition:transform .25s ease,box-shadow .25s ease,border-color .25s ease}
[class*="st-key-pnl"]:hover{transform:translateY(-3px);border-color:color-mix(in srgb,var(--acc) 70%,transparent)!important;
  box-shadow:0 10px 28px color-mix(in srgb,var(--acc) 22%,transparent)}
.kpi{animation:fadeUp .6s ease both;transition:transform .25s ease,box-shadow .25s ease,border-color .25s ease}
.kpi:hover{transform:translateY(-3px);border-color:var(--acc);box-shadow:0 10px 26px color-mix(in srgb,var(--acc) 25%,transparent)}
.kpi .dot{animation:pulse 2.4s ease-in-out infinite}
[data-testid="stMetric"]{animation:fadeUp .6s ease both;transition:transform .25s ease}
[data-testid="stMetric"]:hover{transform:translateY(-3px)}
.ins{animation:fadeUp .7s .15s ease both}
.bars .fl{animation:grow 1.2s cubic-bezier(.2,.8,.2,1) both}
.rank>div{animation:fadeUp .5s ease both}
.rank>div:nth-child(2){animation-delay:.08s}.rank>div:nth-child(3){animation-delay:.16s}
.rank>div:nth-child(4){animation-delay:.24s}.rank>div:nth-child(5){animation-delay:.32s}
.js-plotly-plot{animation:zoomIn .7s ease both}
.badge{transition:transform .2s ease}.kpi:hover .badge{transform:scale(1.08)}

/* ===================== SIDEBAR ===================== */
[data-testid="stSidebarNav"] a{border-radius:6px;transition:background .2s ease,transform .2s ease,padding .2s ease}
[data-testid="stSidebarNav"] a:hover{transform:translateX(4px);background:color-mix(in srgb,var(--acc) 14%,transparent)}
[data-testid="stSidebarNav"] a[aria-current="page"]{background:color-mix(in srgb,var(--acc) 22%,transparent)!important;
  box-shadow:inset 3px 0 0 var(--acc)}
.sb-h{font-size:.72rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--mu);margin:.9rem 0 .4rem}
.sb-card{border:1px solid var(--bd);border-radius:10px;padding:.65rem .75rem;background:linear-gradient(180deg,var(--p1),var(--p2));
  animation:fadeUp .6s ease both}
.sb-prog{height:7px;border-radius:99px;background:var(--track);overflow:hidden;margin:.15rem 0 .55rem}
.sb-prog>div{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--acc),#2EE6B6);animation:grow 1.4s cubic-bezier(.2,.8,.2,1) both}
.sb-st{display:flex;align-items:center;gap:.5rem;font-size:.8rem;color:var(--tx);padding:.16rem 0}
.sb-st .ic{width:18px;height:18px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;
  font-size:.66rem;font-weight:800;flex:none}
.sb-st.ok .ic{background:#2EE6B6;color:#04122E;animation:spinIn .5s ease both}
.sb-st.no .ic{border:1.5px dashed var(--mu);color:var(--mu)}
.sb-st.no{color:var(--mu)}
.sb-kv{display:flex;justify-content:space-between;font-size:.8rem;color:var(--mu);padding:.18rem 0}
.sb-kv b{color:var(--hd);font-variant-numeric:tabular-nums}
.sb-ft{font-size:.72rem;color:var(--mu);margin-top:1rem;line-height:1.5}
@media (prefers-reduced-motion: reduce){*{animation:none!important;transition:none!important}}
.hl{font-size:1.25rem;color:var(--acc)}
</style>
"""

# Menandai halaman dengan data-mt="light|dark" sesuai warna latar aplikasi, supaya CSS di atas ikut berganti
# saat pengguna mengganti tema lewat menu ⋮ -> Settings. Dipasang sekali per tab browser.
_THEME_JS = r"""
(function(){var W=window.parent||window;if(W.__mtOn)return;W.__mtOn=1;var D=W.document;
function u(){var a=D.querySelector('.stApp');if(!a)return;var m=W.getComputedStyle(a).backgroundColor.match(/\d+/g);if(!m)return;
var t=(.299*m[0]+.587*m[1]+.114*m[2])>140?'light':'dark';
if(D.documentElement.getAttribute('data-mt')!==t)D.documentElement.setAttribute('data-mt',t);}
u();W.setInterval(u,300);
var RM=W.matchMedia&&W.matchMedia('(prefers-reduced-motion: reduce)').matches;
function cnt(){D.querySelectorAll('.countup:not([data-done])').forEach(function(el){el.setAttribute('data-done','1');
  var to=parseFloat(el.dataset.to),dec=+el.dataset.dec||0,pre=el.dataset.pre||'',suf=el.dataset.suf||'',t0=null,dur=1200;
  if(isNaN(to)||RM)return;
  function f(v){return pre+v.toLocaleString('en-US',{minimumFractionDigits:dec,maximumFractionDigits:dec})+suf;}
  function st(ts){if(!t0)t0=ts;var p=Math.min(1,(ts-t0)/dur);p=1-Math.pow(1-p,3);el.textContent=f(to*p);if(1>p)W.requestAnimationFrame(st);}
  el.textContent=f(0);W.requestAnimationFrame(st);});}
cnt();new W.MutationObserver(cnt).observe(D.body,{childList:true,subtree:true});})();
"""

FLOW = ["Dataset", "Preprocessing", "Fitur", "Jarak DTW/FastDTW", "Fuzzy clustering", "Prediksi 1-NN", "Evaluasi"]
_ids = itertools.count()


def _theme_probe():
    try:
        st.html(f"<script>{_THEME_JS}</script>", unsafe_allow_javascript=True)
    except TypeError:  # Streamlit lama
        import streamlit.components.v1 as components
        components.html(f"<script>{_THEME_JS}</script>", height=0)


def setup(title, subtitle="", icon=""):
    global _ids
    _ids = itertools.count()
    st.markdown(CSS, unsafe_allow_html=True)
    _theme_probe()
    st.markdown(f'<div class="topbar"><div><div class="tb-t">{icon} {title}</div><div class="tb-s">{subtitle}</div></div>'
                f'<div class="tb-r">MIMII DUE<br>DTW vs FastDTW</div></div>', unsafe_allow_html=True)


def flow(active=None):
    parts = [f'<span class="step{" on" if s == active else ""}">{i + 1}. {s}</span>' for i, s in enumerate(FLOW)]
    st.markdown('<div class="steps">' + '<span class="arrow">›</span>'.join(parts) + "</div>", unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def panel(title=None, desc=None):
    """Kotak panel berbingkai. Pakai: `with ui.panel("Judul", "keterangan singkat"): ...`"""
    c = st.container(border=True, key=f"pnl{next(_ids)}")
    if title:
        c.markdown(f'<div class="pt">{title}</div>', unsafe_allow_html=True)
    if desc:
        c.markdown(f'<div class="pd">{desc}</div>', unsafe_allow_html=True)
    return c


def intro(text):
    """Paragraf pembuka halaman: apa isi halaman ini & untuk apa."""
    st.markdown(f'<div class="intro">{text}</div>', unsafe_allow_html=True)


def insight(text, title="Kesimpulan"):
    """Kotak kesimpulan (dihitung dari data)."""
    st.markdown(f'<div class="ins"><span class="it">💡 {title}</span><br>{text}</div>', unsafe_allow_html=True)


def how(text, label="📖 Cara membaca"):
    """Penjelasan lebih rinci yang bisa dibuka-tutup."""
    with st.expander(label):
        st.markdown(text)


# ---------- kalimat interpretasi otomatis
def says_corr(r):
    if r >= .99:
        return "hampir identik"
    if r >= .95:
        return "sangat mirip"
    if r >= .85:
        return "mirip"
    if r >= .7:
        return "cukup mirip"
    return "kurang mirip"


def says_err(mape):
    if mape < 1:
        return "sangat kecil"
    if mape < 5:
        return "kecil"
    if mape < 15:
        return "sedang"
    return "besar"


def says_ari(a):
    if a >= .9:
        return "hampir sama persis"
    if a >= .7:
        return "sebagian besar sama"
    if a >= .4:
        return "cukup berbeda"
    return "sangat berbeda"


def says_sil(s):
    if s >= .5:
        return "terpisah jelas"
    if s >= .25:
        return "cukup terpisah"
    if s >= .1:
        return "lemah (cluster saling tumpang-tindih)"
    return "sangat lemah / hampir tidak ada struktur cluster"


def need(df, msg):
    if df is None or len(df) == 0:
        st.warning(msg, icon="⚠️")
        st.stop()
    return df


def seg(label, options, key, fmt=None):
    """Pilihan satu-dari-banyak berbentuk tombol; tidak pernah mengembalikan None."""
    options = list(options)
    kw = dict(format_func=fmt) if fmt else {}
    v = st.segmented_control(label, options, default=options[0], key=key, **kw)
    return options[0] if v is None else v


def group_filters(key="g"):
    """Filter mesin x fitur x section — hanya section yang matriks jaraknya sudah dihitung."""
    c1, c2, c3 = st.columns([1, 1.3, 2])
    with c1:
        machine = seg("Mesin", D.MACHINES, f"{key}m")
    with c2:
        feat = seg("Fitur", D.FEATURE_TYPES, f"{key}f")
    secs = D.sections_for(machine, feat)
    if not secs:
        st.warning("Belum ada matriks jarak untuk pilihan ini. Jalankan scripts/08 & 09.", icon="⚠️")
        st.stop()
    with c3:
        sec = seg("Section", secs, f"{key}s{machine}{feat}", fmt=lambda s: f"{s:02d}")
    return machine, int(sec), feat


def tidy(fig, h=420):
    top = 44 if fig.layout.title.text else (34 if (fig.layout.legend.y or 0) > 1 else 12)
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=top, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", legend_bgcolor="rgba(0,0,0,0)")
    if fig.layout.legend.orientation is None:
        fig.update_layout(legend=dict(orientation="h", y=-0.2, x=0))
    fig.update_xaxes(gridcolor=GRIDC, zerolinecolor=GRIDC)
    fig.update_yaxes(gridcolor=GRIDC, zerolinecolor=GRIDC)
    fig.update_polars(bgcolor="rgba(0,0,0,0)", radialaxis_gridcolor=GRIDC, angularaxis_gridcolor=GRIDC)
    return fig


def chart(fig, h=420):
    """Grafik tanpa panel (dipakai di dalam panel lain)."""
    st.plotly_chart(tidy(fig, h), width="stretch", theme="streamlit", config={"displaylogo": False, "displayModeBar": False})


def show(fig, h=420, title=None, desc=None, howto=None):
    """Grafik di dalam panel berbingkai, dengan judul, keterangan singkat, dan 'cara membaca' (opsional)."""
    with panel(title, desc):
        chart(fig, h)
        if howto:
            how(howto)


# ------------------------------------------------------------------ elemen gaya big-screen
def countup(value):
    """'1,234' / '0.11%' / '24.5×' -> span yang angkanya dianimasikan naik dari 0 di browser."""
    m = re.match(r"^(\D*?)(-?[\d,]*\.?\d+)(.*)$", str(value))
    if not m:
        return str(value)
    pre, num, suf = m.groups()
    dec = len(num.split(".")[1]) if "." in num else 0
    return (f'<span class="countup" data-to="{num.replace(",", "")}" data-dec="{dec}" data-pre="{pre}" '
            f'data-suf="{suf}">{value}</span>')


def kpi(value, label, color=BLUE, badge=None, badge_color=None, delay=0):
    b = (f'<span class="badge" style="background:{badge_color or color}">{badge}</span>' if badge else "")
    return (f'<div class="kpi" style="animation-delay:{delay:.2f}s"><div class="row"><div class="v">{countup(value)}</div>{b}</div>'
            f'<div class="l"><span class="dot" style="background:{color};box-shadow:0 0 8px {color}"></span>{label}</div></div>')


def kpi_row(items):
    cols = st.columns(len(items))
    for i, (c, it) in enumerate(zip(cols, items)):
        it = list(it) + [None] * (5 - len(it))
        c.markdown(kpi(*it[:5], delay=.07 * i), unsafe_allow_html=True)


def gauge(frac, label, color, h=210):
    """Donut melingkar ala '75% Succeed'."""
    frac = float(np.clip(frac, 0, 1))
    fig = go.Figure(go.Pie(values=[frac, 1 - frac], hole=.8, sort=False, direction="clockwise", rotation=0,
                           marker=dict(colors=[color, "rgba(128,150,200,0.18)"], line=dict(width=0)),
                           textinfo="none", hoverinfo="skip"))
    fig.update_layout(showlegend=False, height=h, margin=dict(l=4, r=4, t=4, b=4),
                      annotations=[dict(text=f"<b>{frac * 100:.0f}%</b>", y=.56, showarrow=False,
                                        font=dict(size=30, color=color)),
                                   dict(text=label, y=.36, showarrow=False, font=dict(size=13))])
    return fig


def rank_tiles(items):
    """items: [(nilai, keterangan)] maks 5 -> 'No.1 … No.5'."""
    cells = "".join(f'<div><div class="n">No.{i + 1}</div><div class="v">{v}</div><div class="s" title="{s}">{s}</div></div>'
                    for i, (v, s) in enumerate(items[:5]))
    st.markdown(f'<div class="rank">{cells}</div>', unsafe_allow_html=True)


def rank_bars(items, fmt="{:.3f}"):
    """items: [(label, nilai)] -> bar horizontal bercahaya, panjang relatif ke nilai maksimum."""
    if not items:
        return
    mx = max(v for _, v in items) or 1
    rows = "".join(f'<div class="it"><div class="h"><span>{l}</span><span>{fmt.format(v)}</span></div>'
                   f'<div class="tr"><div class="fl" style="width:{max(4, 100 * v / mx):.0f}%"></div></div></div>'
                   for l, v in items)
    st.markdown(f'<div class="bars">{rows}</div>', unsafe_allow_html=True)


def audio(url_or_path):
    p = D.url_to_path(url_or_path) if isinstance(url_or_path, str) else url_or_path
    if p is None:
        st.info("Audio tidak tersedia di versi ini.", icon="ℹ️")
    else:
        st.audio(str(p))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def grp_label(df):
    return df.machine + " s" + df.section.astype(int).astype(str).str.zfill(2) + " " + df.feature


# ------------------------------------------------------------------ sidebar
def _stage_status():
    """Tahap pipeline mana yang sudah selesai (dicek dari file hasil)."""
    has = lambda p: p.exists()  # noqa: E731
    feats = any(D.FEATURES.glob("*/*/*.npy")) or any(D.PACK.glob("features_*.npz"))
    return [
        ("Dataset & inventaris", has(D.MANIFEST)),
        ("Preprocessing audio", has(D.PROC_MANIFEST)),
        ("Ekstraksi fitur", feats),
        ("Jarak DTW & FastDTW", bool(D.groups_available())),
        ("Fuzzy clustering", any(D.CLUSTERING.glob("labels/*.csv"))),
        ("Evaluasi", has(D.RESULTS / "metrics" / "cluster_metrics.csv")),
        ("Prediksi k-NN", has(D.PRED_RES / "metrics.csv")),
        ("Paket dashboard", has(D.PACK / "info.json")),
    ]


def sidebar():
    """Isi sidebar di bawah menu: status pipeline, ringkasan cepat, tips."""
    st.markdown(CSS, unsafe_allow_html=True)
    stages = _stage_status()
    done = sum(ok for _, ok in stages)
    rows = "".join(f'<div class="sb-st {"ok" if ok else "no"}" style="animation-delay:{.05 * i:.2f}s">'
                   f'<span class="ic" style="animation-delay:{.08 * i:.2f}s">{"✓" if ok else i + 1}</span>{name}</div>'
                   for i, (name, ok) in enumerate(stages))
    st.markdown(f'<div class="sb-h">Status pipeline</div><div class="sb-card">'
                f'<div class="sb-kv"><span>{done} dari {len(stages)} tahap</span><b>{done / len(stages) * 100:.0f}%</b></div>'
                f'<div class="sb-prog"><div style="width:{done / len(stages) * 100:.0f}%"></div></div>{rows}</div>',
                unsafe_allow_html=True)

    man = D.csv(D.MANIFEST)
    am = D.csv(D.RESULTS / "metrics" / "approximation_metrics.csv")
    pm = D.csv(D.PRED_RES / "metrics.csv")
    kv = []
    if man is not None:
        kv.append(("Rekaman", f"{len(man):,}"))
    if am is not None and len(am):
        kv += [("Grup jarak", str(len(am))), ("Korelasi DTW↔FastDTW", f"{am.pearson.mean():.3f}"),
               ("MAPE FastDTW", f"{am.MAPE_pct.mean():.2f}%")]
    if pm is not None and len(pm):
        acc = pm.groupby("method").accuracy.mean()
        if "FastDTW" in acc:
            kv.append(("Akurasi FastDTW", f"{acc['FastDTW'] * 100:.1f}%"))
    if kv:
        st.markdown('<div class="sb-h">Ringkasan cepat</div><div class="sb-card">'
                    + "".join(f'<div class="sb-kv"><span>{k}</span><b>{countup(v)}</b></div>' for k, v in kv)
                    + "</div>", unsafe_allow_html=True)
    st.markdown('<div class="sb-ft">🌗 Ganti tema: menu <b>⋮</b> (kanan atas) → Light / Dark.<br>'
                '📘 Bingung istilah? Buka <b>Panduan &amp; Istilah</b>.<br><br>MIMII DUE · DTW vs FastDTW</div>',
                unsafe_allow_html=True)
