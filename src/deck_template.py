"""HTML/CSS/JS template for the Tactical DNA presentation deck.

Bold editorial / cinematic design system:
  - fixed 16:9 stage scaled to the viewport (consistent type, no fluid vw drift)
  - strict layout grid: every slide shares margins; kicker + headline are
    top-anchored in the same place on every slide
  - cinematic dark theme with York-red as a sharp accent
  - presentation-grade charts: no axes / gridlines, large glowing points,
    direct labels — legible from the back row

render(data, figs, plotly_js) -> full self-contained HTML string.
"""
from __future__ import annotations

import json


def render(data: dict, figs: dict, plotly_js: str) -> str:
    return _DOC.replace("/*PLOTLY*/", plotly_js) \
              .replace("/*DATA*/", json.dumps(data)) \
              .replace("/*CSS*/", _CSS) \
              .replace("/*SLIDES*/", _SLIDES) \
              .replace("/*JS*/", _JS)


# ===========================================================================
# DESIGN SYSTEM
# ===========================================================================
_CSS = r"""
:root{
  /* York University official brand red (Pantone 200 C) */
  --red:#E31837; --red2:#b3142c; --red-soft:#E31837;
  --bg:#ffffff; --bg2:#f5f6f8; --panel:#f7f8fa; --line:#e3e4ea;
  --ink:#16161d; --mut:#5a5a6a; --mut2:#9a9aac;
  --gk:#8a8f9c; --def:#2f6fd0; --mid:#1f9e57; --fwd:#e1334a; --c0:#2f6fd0; --c1:#ef7a2e;
  /* fixed type scale on a 1280x720 stage */
  --s-kick:15px; --s-h1:108px; --s-h2:56px; --s-lead:27px; --s-body:21px;
  --s-small:16px; --s-stat:104px;
}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;background:#0a0a0a;overflow:hidden;
  font-family:'Inter','Helvetica Neue',Arial,sans-serif;-webkit-font-smoothing:antialiased}

/* fixed stage, scaled to fit the screen by JS */
#stage{position:absolute;top:50%;left:50%;width:1280px;height:720px;
  transform:translate(-50%,-50%);transform-origin:center center;background:var(--bg)}
.slide{position:absolute;inset:0;display:none;color:var(--ink);
  padding:76px 96px 70px;overflow:hidden;background:#fff}
.slide.on{display:block}

/* clean white backgrounds; accent slides get a subtle red edge */
.slide.accent{background:#fff}
.slide.accent:before{content:"";position:absolute;left:0;top:0;bottom:0;width:8px;
  background:linear-gradient(180deg,var(--red),var(--red2))}
.slide.light{background:#fff;color:#16161d}

/* ---- shared header grid: kicker + headline pinned identically ---- */
.kick{font-size:var(--s-kick);font-weight:800;letter-spacing:.28em;text-transform:uppercase;
  color:var(--red-soft);margin-bottom:14px}
.slide.light .kick{color:var(--red)}
.kick b{color:var(--mid)}
h1{font-size:var(--s-h1);font-weight:850;line-height:.96;letter-spacing:-.03em}
h2{font-size:var(--s-h2);font-weight:850;line-height:1.04;letter-spacing:-.02em;max-width:26ch}
/* fixed header zone: kicker + headline live here on every content slide,
   and the content canvas always begins below it — no overlaps */
.hgroup{position:absolute;left:96px;right:96px;top:64px;height:118px}
.hgroup .kick{margin-bottom:10px}
.head{margin-bottom:30px}
.lead{font-size:var(--s-lead);line-height:1.42;color:var(--mut);max-width:30ch;font-weight:450}
.slide.light .lead{color:#444}
.body{font-size:var(--s-body);line-height:1.5;color:var(--mut)}
.em{color:var(--ink)} .slide.light .em{color:#111}
.red{color:var(--red-soft)} .slide.light .red{color:var(--red)}
.tnum{font-variant-numeric:tabular-nums}

/* content area below the header — uniform start so headlines never collide */
.canvas{position:absolute;left:96px;right:96px;top:204px;bottom:72px}

/* minimal footer: a discreet page number, bottom-right */
.foot{position:absolute;right:96px;bottom:34px;z-index:6}
.pgnum{font-size:14px;color:var(--mut2);letter-spacing:.06em}
.pgnum b{color:var(--red);font-weight:800;font-size:16px;margin-right:4px}
.pgnum .of{color:#bdbdc8}

/* two-column split */
.split{position:absolute;inset:0;display:grid;grid-template-columns:var(--cols,1fr 1fr);gap:54px;align-items:center}
.split.r{--cols:0.92fr 1.08fr}
.split.l{--cols:1.12fr 0.88fr}

/* ---- pills / chips ---- */
.pill{display:inline-block;background:var(--red);color:#fff;font-weight:800;
  padding:5px 14px;border-radius:9px;font-size:18px;letter-spacing:.01em}
.tag{font-size:13px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:var(--red-soft)}
.slide.light .tag{color:var(--red)}

/* ---- numbered cards (problems / findings) ---- */
.cards{display:grid;grid-template-columns:1fr 1fr 1fr;gap:26px;height:100%;align-content:start;padding-top:30px}
.card{background:#fff;border:1px solid var(--line);
  border-radius:18px;padding:28px 26px;position:relative;overflow:hidden;
  box-shadow:0 10px 30px rgba(20,20,40,.07)}
.card:before{content:"";position:absolute;top:0;left:0;right:0;height:4px;
  background:linear-gradient(90deg,var(--red),var(--red2))}
.card .ix{font-size:34px;line-height:1;margin-bottom:14px}
.ibadge{width:54px;height:54px;border-radius:14px;background:rgba(227,24,55,.08);
  display:flex;align-items:center;justify-content:center;margin-bottom:18px}
.ibadge svg{width:30px;height:30px}
.card h3{font-size:27px;font-weight:850;letter-spacing:-.01em;margin:6px 0 10px;color:var(--ink)}
.card p{font-size:17px;line-height:1.5;color:var(--mut)}
.card p span[style*="#fff"]{color:var(--ink)!important}
.card .big{font-size:30px;font-weight:850;color:var(--red);margin:2px 0 8px}

/* ---- stats row ---- */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:30px;align-content:center;height:100%}
.stat{text-align:center}
.stat .n{font-size:var(--s-stat);font-weight:850;line-height:.9;color:var(--ink);letter-spacing:-.04em}
.stat .n .u{font-size:.42em;color:var(--red-soft);vertical-align:super;font-weight:800}
.stat .l{font-size:16px;color:var(--mut);font-weight:600;margin-top:12px;letter-spacing:.02em}

/* ---- data tables ---- */
table.t{border-collapse:collapse;width:100%;font-size:20px}
table.t th{font-size:13px;font-weight:800;letter-spacing:.1em;text-transform:uppercase;
  color:var(--mut);text-align:left;padding:0 14px 12px;border-bottom:1px solid var(--line)}
table.t td{padding:13px 14px;border-bottom:1px solid #ececed;color:var(--ink)}
table.t td.num{text-align:right;font-variant-numeric:tabular-nums;font-weight:800}
table.t tr.hi td{background:rgba(227,24,55,.10)}
table.t tr.hi td:first-child{box-shadow:inset 4px 0 0 var(--red)}
.up{color:#1f9e57;font-weight:850}.down{color:#e1334a;font-weight:850}

/* ---- chart frame (presentation-grade) ---- */
.figframe{position:absolute;inset:0;border-radius:18px;overflow:hidden}
.fig{width:100%;height:100%}
/* make the Plotly zoom toolbar dark + visible on the white slides */
.fig .modebar{background:rgba(255,255,255,.9)!important;border-radius:8px;
  box-shadow:0 2px 8px rgba(20,20,40,.12)}
.fig .modebar-btn .icon path{fill:#16161d!important}
.fig .modebar-btn:hover .icon path{fill:var(--red)!important}
.figcap{position:absolute;top:0;left:0;font-size:14px;font-weight:600;color:var(--mut);letter-spacing:.02em}

/* takeaway callout band */
.callout{position:absolute;left:96px;right:96px;bottom:54px;display:flex;align-items:center;gap:16px;
  background:rgba(227,24,55,.05);border:1px solid rgba(227,24,55,.18);border-radius:14px;
  padding:16px 22px;font-size:19px;line-height:1.4;color:var(--ink)}
.callout .co-ic{flex:0 0 auto;width:30px;height:30px;border-radius:50%;background:var(--red);
  color:#fff;font-weight:850;display:flex;align-items:center;justify-content:center;font-size:18px}

/* tall Xavi portrait panel (slide 3) — full-body photo, caption overlaid */
.xaviframe{position:relative;height:100%;border-radius:16px;overflow:hidden;
  box-shadow:0 12px 32px rgba(20,20,40,.14);align-self:stretch}
.xaviframe img{width:100%;height:100%;object-fit:cover;object-position:top center;display:block}
.xaviframe:after{content:"";position:absolute;inset:0;
  background:linear-gradient(180deg,rgba(0,0,0,0) 55%,rgba(0,0,0,.72) 100%)}
.xavicap{position:absolute;left:14px;right:14px;bottom:12px;z-index:2}
.xavicap b{display:block;color:#fff;font-size:18px;font-weight:850;line-height:1.1}
.xavicap span{display:block;color:#ffd2da;font-size:12.5px;font-weight:600;margin-top:2px}
.xaviframe:before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;z-index:2;
  background:var(--red)}
.lg{display:flex;gap:22px;font-size:16px;font-weight:700;color:var(--mut);margin-bottom:10px}
.lg span{display:flex;align-items:center;gap:8px}
.lg i{width:13px;height:13px;border-radius:50%}

/* search bar (P3) */
.sbar{display:flex;gap:12px;align-items:center;margin-bottom:10px}
.sbar b{font-size:18px;color:var(--ink)}
.sbar input{font-size:18px;padding:9px 14px;border-radius:10px;border:1px solid var(--line);
  background:var(--panel);color:var(--ink);min-width:280px;outline:none}
.sbar input:focus{border-color:var(--red)}
.sbar button{font-size:16px;font-weight:800;padding:9px 18px;border:none;border-radius:10px;
  background:var(--red);color:#fff;cursor:pointer}
.sbar .hint{font-size:16px;color:var(--mut)}

/* ML pipeline flow */
.flow{display:flex;align-items:center;gap:16px;justify-content:center;margin-bottom:30px}
.flow .node{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  padding:16px 20px;text-align:center;font-weight:800;font-size:18px}
.flow .node small{display:block;font-weight:600;color:var(--mut);font-size:13px;margin-top:4px}
.flow .ar{color:var(--red);font-size:26px;font-weight:800}
.trip{display:grid;grid-template-columns:1fr 1fr 1fr;gap:22px}
.trip .b{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--red);
  border-radius:14px;padding:20px;font-size:17px;line-height:1.5;color:var(--mut)}
.trip .b b{color:var(--red-soft)}

/* bullet list */
ul.k{list-style:none;font-size:21px;line-height:1.8}
ul.k li{position:relative;padding-left:30px;margin-bottom:10px;color:var(--mut)}
ul.k li b{color:var(--ink)}
ul.k li:before{content:"";position:absolute;left:0;top:13px;width:11px;height:11px;
  background:var(--red);border-radius:3px;transform:rotate(45deg)}
.colh{font-size:20px;font-weight:850;color:var(--red-soft);margin-bottom:14px;letter-spacing:.01em}

/* GitHub repo card on the close slide */
.repo{display:inline-flex;align-items:center;gap:14px;margin-top:26px;text-decoration:none;
  background:#fff;border:1px solid var(--line);border-radius:14px;padding:14px 22px;
  box-shadow:0 8px 24px rgba(20,20,40,.08);transition:transform .15s,box-shadow .15s}
.repo:hover{transform:translateY(-2px);box-shadow:0 12px 30px rgba(20,20,40,.14)}
.repo span{display:flex;flex-direction:column;text-align:left}
.repo b{font-size:18px;font-weight:850;color:var(--ink);line-height:1.2}
.repo small{font-size:13px;color:var(--mut);font-weight:600;margin-top:2px}

/* methods slide: feature families + notes */
.featgrid{display:flex;flex-direction:column;gap:14px}
.featcard{display:flex;align-items:center;gap:18px;background:#fff;border:1px solid var(--line);
  border-radius:14px;padding:16px 20px;box-shadow:0 6px 18px rgba(20,20,40,.05)}
.featcard .fnum{flex:0 0 auto;width:58px;height:58px;border-radius:12px;background:rgba(227,24,55,.08);
  color:var(--red);font-size:28px;font-weight:850;display:flex;align-items:center;justify-content:center}
.featcard .fbody b{display:block;font-size:19px;font-weight:850;color:var(--ink);margin-bottom:3px}
.featcard .fbody span{font-size:15px;line-height:1.4;color:var(--mut)}
.mini-note{margin-top:22px;font-size:14px;line-height:1.5;color:var(--mut2);
  border-left:3px solid var(--line);padding-left:14px}

/* image placeholders */
.ph{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;
  border:2px dashed var(--red-soft);border-radius:14px;background:rgba(225,24,55,.06);
  color:var(--red-soft);text-align:center;padding:14px}
.ph .pi{font-size:30px;opacity:.85}.ph .pl{font-size:13px;font-weight:800;letter-spacing:.04em}
.ph .ps{font-size:11px;font-weight:600;opacity:.8}
.ph.bgph{position:absolute;inset:0;border-radius:0;border-width:3px;justify-content:flex-end;
  padding-bottom:36px;z-index:0}
.ph.bgph .pi{position:absolute;top:30px;left:50%;transform:translateX(-50%)}
.ph.rnd{border-radius:50%;aspect-ratio:1}
.crests{display:flex;gap:30px;align-items:center;justify-content:flex-start}
.crests figure{text-align:center;margin:0}
.crestbox{width:72px;height:72px;display:flex;align-items:center;justify-content:center}
.crestbox img{max-width:64px;max-height:64px;width:auto;height:auto;object-fit:contain;display:block}
.crests figcaption{font-size:13px;color:var(--mut);font-weight:700;margin-top:10px;letter-spacing:.02em}

/* reveal animation */
.fx{opacity:0;transform:translateY(18px);transition:opacity .6s cubic-bezier(.2,.7,.2,1),transform .6s cubic-bezier(.2,.7,.2,1)}
.slide.on .fx.show{opacity:1;transform:none}

/* photo backgrounds (images pushed to GitHub alongside the HTML) */
.photobg{position:absolute;inset:0;z-index:0;overflow:hidden}
.photobg img{width:100%;height:100%;object-fit:cover}
.photobg:after{content:"";position:absolute;inset:0;
  background:linear-gradient(180deg,rgba(10,8,14,.62),rgba(10,8,14,.78))}
.photobg.dim:after{background:linear-gradient(180deg,rgba(8,8,14,.72),rgba(8,8,14,.86))}
/* York logo */
.ylogo{position:absolute;z-index:5;height:46px;width:auto}
.ylogo.tl{top:34px;left:96px}
.ylogo.tr{top:36px;right:90px;height:34px;opacity:.92}
/* generic inset photo (slide 3 Xavi) */
.photo{width:100%;height:100%;object-fit:cover;border-radius:16px;display:block}

/* title slide: text left, photo panel right */
.titlegrid{position:absolute;inset:0;display:grid;grid-template-columns:0.92fr 1.08fr}
.titleleft{display:flex;flex-direction:column;justify-content:center;padding:0 40px 0 96px}
.titlephoto{position:relative;overflow:hidden}
.titlephoto img{width:100%;height:100%;object-fit:cover}
.titlephoto:before{content:"";position:absolute;inset:0;z-index:2;
  background:linear-gradient(90deg,#fff 0%,rgba(255,255,255,.55) 8%,rgba(255,255,255,0) 22%)}
.redrule{width:64px;height:5px;background:var(--red);border-radius:3px}

/* hero (title) */
#hero{position:absolute;inset:0;z-index:1;opacity:.4}
.center{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;
  justify-content:center;text-align:center;z-index:2;padding:0 96px}
.mark{width:62px;height:62px}
.mark.big{width:84px;height:84px}

/* progress + dots */
#dots{position:absolute;bottom:14px;left:50%;transform:translateX(-50%);display:flex;gap:6px;z-index:9}
#dots .d{width:7px;height:7px;border-radius:50%;background:#d6d6dd;transition:all .3s}
#dots .d.on{background:var(--red);width:20px;border-radius:4px}
#bar{position:absolute;bottom:0;left:0;height:4px;background:var(--red);width:0;z-index:9;transition:width .4s}
"""


# ===========================================================================
# York shield (clean original recreation)
_MARK = """<svg class="mark" viewBox="0 0 100 100"><path d="M12 8 H88 V60 Q88 84 50 96 Q12 84 12 60 Z" fill="#E11837"/><path d="M30 26 L50 50 L70 26 H58 L50 37 L42 26 Z" fill="#fff"/><rect x="45" y="48" width="10" height="26" fill="#fff"/></svg>"""
_MARKBIG = _MARK.replace('class="mark"', 'class="mark big"')


def foot(n):
    # York logo top-right on every content slide + a discreet page number bottom-right
    return ('<img src="YorkuLogo.png" class="ylogo tr" alt="York University">'
            f'<div class="foot"><span class="pgnum tnum">'
            f'<b>{n:02d}</b><span class="of">/ 14</span></span></div>')


# ===========================================================================
# SLIDES
# ===========================================================================
_SLIDES = f"""
<!-- 1 TITLE -->
<section class="slide titleslide" data-t="30">
  <img src="YorkuLogo.png" class="ylogo tl fx" alt="York University">
  <div class="titlegrid">
    <div class="titleleft">
      <div class="kick fx" style="letter-spacing:.18em">Final Presentation · EECS 4414</div>
      <h1 class="fx" style="margin:14px 0 22px;font-size:88px;line-height:.98">Tactical<br>DNA</h1>
      <div class="redrule fx"></div>
      <div class="lead fx" style="max-width:22ch;font-size:26px;font-weight:500;color:var(--ink);margin-top:22px">
        Coaches, Styles &amp; Player Roles from Football Passing Networks</div>
      <div class="fx" style="margin-top:30px;font-size:18px;font-weight:600;color:var(--mut)">
        Mahyar Jaberi&nbsp;&nbsp;·&nbsp;&nbsp;mhyrjbr@my.yorku.ca</div>
    </div>
    <div class="titlephoto fx"><img src="Slide1_YorkuBuilding.jpeg" alt="York University"></div>
  </div>
</section>

<!-- 2 HOOK -->
<section class="slide" data-t="40">
  <div class="photobg dim"><img src="AlianzArenaOutside_Slide2.jpg" alt=""></div>
  <div class="center" style="z-index:2">
    <div class="kick fx" style="color:#fff;opacity:.85">The question</div>
    <h2 class="fx" style="max-width:20ch;font-size:60px;margin:14px auto 0;line-height:1.1;color:#fff;
      text-shadow:0 2px 24px rgba(0,0,0,.6)">
      Pundits <span style="color:var(--red)">say</span> &ldquo;possession football.&rdquo;<br>
      Can we <span style="color:var(--red)">measure</span> it?</h2>
    <div class="fx" style="max-width:60ch;margin:30px auto 0;font-size:22px;line-height:1.5;color:#eaeaf2;
      text-shadow:0 1px 16px rgba(0,0,0,.7)">
      &ldquo;Gegenpressing,&rdquo; &ldquo;style of play,&rdquo; a coach&rsquo;s &ldquo;fingerprint&rdquo;:<br>rich language, almost never quantified. We turn passing into a
      <b style="color:#fff">measurable mathematical object</b>.</div>
  </div>
</section>

<!-- 3 THE IDEA + live Clasico -->
<section class="slide" data-t="40">
  <div class="hgroup fx"><div class="kick">The idea</div>
    <h2 style="font-size:50px">A passing network is a tactical fingerprint</h2></div>
  <div class="canvas"><div style="position:absolute;inset:0;display:grid;grid-template-columns:0.62fr 0.78fr 1.2fr;gap:40px">
    <!-- tall Xavi portrait panel (full body) -->
    <div class="xaviframe fx">
      <img src="Xavi_slide3.jpg" alt="Xavi">
      <div class="xavicap"><b>Xavi Hern&aacute;ndez</b><span>top PageRank · the metronome</span></div>
    </div>
    <div style="display:flex;flex-direction:column;justify-content:center">
      <div class="lead fx" style="max-width:22ch;font-size:24px">
        Players are <span class="em">nodes</span>; a completed pass is a directed
        <span class="em">edge</span>; weight = how often.</div>
      <div class="fx" style="margin-top:22px"><span class="pill">G = (V, E, w)</span></div>
      <div class="body fx" style="margin-top:24px;font-size:17px;max-width:24ch">
        <span class="em">Xavi</span> sits dead-centre, exactly the player the eye would pick.</div>
    </div>
    <div style="position:relative;height:100%" class="fx">
      <div class="figcap">Bar&ccedil;a 5&ndash;0 Real Madrid · 2010 · node size = PageRank</div>
      <div class="figframe" style="top:34px"><div id="clasicoChart" class="fig"></div></div>
    </div>
  </div></div>
  {foot(3)}
</section>

<!-- 4 THREE QUESTIONS -->
<section class="slide" data-t="30">
  <div class="hgroup fx"><div class="kick">Three inference tasks · one graph</div>
    <h2>From one graph, three questions</h2></div>
  <div class="canvas" style="top:184px">
    <div class="cards">
      <div class="card fx">
        <div class="ibadge"><svg viewBox="0 0 24 24" fill="none" stroke="#E31837" stroke-width="2"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.3" fill="#E31837"/></svg></div>
        <div class="tag">P1 · Supervised</div>
        <h3>Who coached it?</h3>
        <p>Predict the <span class="em" style="color:#fff">coach</span> from passing structure alone. Does that signature follow them to a new club?</p></div>
      <div class="card fx">
        <div class="ibadge"><svg viewBox="0 0 24 24" fill="#E31837"><circle cx="7" cy="8" r="2.4"/><circle cx="16" cy="6.5" r="2.4"/><circle cx="6" cy="16" r="2.4"/><circle cx="15" cy="16.5" r="2.8"/><circle cx="11" cy="11.5" r="2"/></svg></div>
        <div class="tag">P2 · Unsupervised</div>
        <h3>What style is it?</h3>
        <p>Cluster matches into <span class="em" style="color:#fff">tactical archetypes</span> with no labels. Does possession vs. direct emerge on its own?</p></div>
      <div class="card fx">
        <div class="ibadge"><svg viewBox="0 0 24 24" fill="#E31837"><circle cx="12" cy="8" r="3.4"/><path d="M5 20 a7 7 0 0 1 14 0 Z"/></svg></div>
        <div class="tag">P3 · Embedding</div>
        <h3>What role is this?</h3>
        <p>Embed <span class="em" style="color:#fff">players</span> into a role space, recovering position, and find structural &ldquo;successors&rdquo; to great playmakers.</p></div>
    </div>
  </div>
  {foot(4)}
</section>

<!-- 5 DATA -->
<section class="slide" data-t="40">
  <div class="hgroup fx"><div class="kick">The corpus · StatsBomb open data</div>
    <h2>775 passing networks, built end-to-end</h2></div>
  <div class="canvas" style="bottom:180px;top:230px">
    <div class="stats">
      <div class="stat fx"><div class="n tnum" data-count="775">0</div><div class="l">team-match networks</div></div>
      <div class="stat fx"><div class="n tnum" data-count="42">0</div><div class="l">teams</div></div>
      <div class="stat fx"><div class="n tnum" data-count="271">0</div><div class="l">players embedded</div></div>
      <div class="stat fx"><div class="n tnum" data-count="11">0</div><div class="l">nodes / network</div></div>
    </div>
  </div>
  <!-- real team crests -->
  <div style="position:absolute;left:96px;right:96px;bottom:84px" class="fx">
    <div class="crests">
      <figure><div class="crestbox"><img src="FcBarcelona_Logo_Slide5.cc.svg" alt="Barcelona"></div><figcaption>Barcelona</figcaption></figure>
      <figure><div class="crestbox"><img src="Spain_Logo_Slide5.cc.svg" alt="Spain"></div><figcaption>Spain</figcaption></figure>
      <figure><div class="crestbox"><img src="Argentina_Logo_Slide5.cc.svg" alt="Argentina"></div><figcaption>Argentina</figcaption></figure>
      <figure><div class="crestbox"><img src="Bayern_Logo_Slide5.cc.svg" alt="Bayern"></div><figcaption>Bayern</figcaption></figure>
      <span class="body" style="font-size:17px;align-self:center;margin-left:8px">+ 38 more · 16 La&nbsp;Liga seasons · 8 coaches · 2 World&nbsp;Cups</span>
    </div>
  </div>
  {foot(5)}
</section>

<!-- 6 METHODS & FEATURES -->
<section class="slide" data-t="55">
  <div class="hgroup fx"><div class="kick">Methodology · how a graph becomes features</div>
    <h2 style="font-size:50px">From passing to a 50-dim fingerprint</h2></div>
  <div class="canvas" style="top:206px">
    <div style="display:grid;grid-template-columns:0.9fr 1.1fr;gap:50px;height:100%;align-content:start">
      <!-- left: construction + notation -->
      <div class="fx">
        <div class="colh">Network construction</div>
        <div class="body" style="font-size:17px;line-height:1.55;color:var(--mut)">
          Each team-match becomes a directed, weighted graph
          <span class="pill" style="font-size:15px;padding:3px 10px">G&nbsp;=&nbsp;(V,&nbsp;E,&nbsp;w)</span><br><br>
          <b class="em">V</b> = the 11 starters &nbsp;·&nbsp; <b class="em">E</b> = completed open-play passes &nbsp;·&nbsp;
          <b class="em">w</b> = pass count.
          We take only events <b class="em">before the first substitution</b>, so every graph
          has exactly 11 nodes and stays comparable.</div>
        <div class="mini-note">k-means (WCSS) and modularity are NP-hard; we use Lloyd &amp; Ward heuristics.
          Fixing motifs to 3 nodes keeps the triadic census polynomial.</div>
      </div>
      <!-- right: the 50 features by family -->
      <div class="fx">
        <div class="colh">The embedding &phi;(G) &middot; 50 features, three families</div>
        <div class="featgrid">
          <div class="featcard"><div class="fnum">20</div><div class="fbody">
            <b>Centrality</b><span>weighted in/out-degree, betweenness, eigenvector, PageRank,
            each summarised by mean, variance, skew &amp; Gini</span></div></div>
          <div class="featcard"><div class="fnum">16</div><div class="fbody">
            <b>Motifs</b><span>triadic census: all 13 connected 3-node directed motifs, normalised</span></div></div>
          <div class="featcard"><div class="fnum">8</div><div class="fbody">
            <b>Global structure</b><span>density, weighted clustering, avg shortest-path, algebraic
            connectivity (Fiedler), spectral radius</span></div></div>
        </div>
      </div>
    </div>
  </div>
  {foot(6)}
</section>

<!-- 7 P1 RESULTS -->
<section class="slide" data-t="60">
  <div class="hgroup fx"><div class="kick">Problem 1 · Coach identification</div>
    <h2 style="font-size:48px">A modest signal, and an honest catch</h2></div>
  <div class="canvas"><div class="split l" style="position:absolute;inset:0;align-items:center">
    <div>
      <table class="t fx">
        <tr><th>Model</th><th style="text-align:right">Random CV</th><th style="text-align:right">Leakage-free</th></tr>
        <tr><td>Majority baseline</td><td class="num">0.263</td><td class="num" style="color:#bbb">n/a</td></tr>
        <tr><td>Logistic Reg.</td><td class="num">0.244</td><td class="num">0.187</td></tr>
        <tr class="hi"><td><b class="em">Random Forest</b></td><td class="num">0.358</td><td class="num">0.250</td></tr>
        <tr><td>XGBoost</td><td class="num">0.298</td><td class="num">0.256</td></tr>
      </table>
      <div class="body fx" style="margin-top:22px;font-size:18px;max-width:36ch">
        RF beats baseline by <span class="up">+0.095</span>, but a leakage-free,
        season-grouped split collapses it to <span class="em">0.250</span>. The model
        partly memorised <span class="em">which season</span>, not which coach.</div>
    </div>
    <div style="position:relative;height:100%" class="fx">
      <div class="figcap"><b style="color:var(--ink)">Confusion matrix</b> · Random Forest · rows = true coach, columns = predicted</div>
      <div class="figframe" style="top:34px"><div id="confChart" class="fig"></div></div>
    </div>
  </div></div>
  {foot(7)}
</section>

<!-- 7 P1 TRANSFER -->
<section class="slide" data-t="60">
  <div class="hgroup fx"><div class="kick">Problem 1 · The marquee test</div>
    <h2>Does a coach&rsquo;s fingerprint <span class="red">travel?</span></h2></div>
  <div class="canvas" style="bottom:118px"><div style="position:absolute;inset:0;display:grid;grid-template-columns:1fr 1.05fr;gap:50px;align-items:center">
    <div>
      <div class="lead fx" style="font-size:21px;max-width:30ch">
        Train on Bar&ccedil;a; test three of its coaches managing a
        <span class="em">different team</span>.</div>
      <table class="t fx" style="margin-top:20px">
        <tr><th>Coach</th><th>New team</th><th style="text-align:right">Recovered</th></tr>
        <tr><td>Guardiola</td><td>Bayern &rsquo;15/16</td><td class="num down">0 / 2</td></tr>
        <tr class="hi"><td>Luis Enrique</td><td>Spain &rsquo;22</td><td class="num up">0.42</td></tr>
        <tr><td>Martino</td><td>Mexico &rsquo;22</td><td class="num">0.11</td></tr>
      </table>
    </div>
    <div style="position:relative;height:100%" class="fx">
      <div class="figcap"><b style="color:var(--ink)">Recovery rate per model</b> · dashed line = uniform chance (0.125)</div>
      <div class="figframe" style="top:34px"><div id="transferChart" class="fig"></div></div>
    </div>
  </div></div>
  <div class="callout fx">
    <span class="co-ic">!</span>
    <div>Mean recovery <b class="em">~0.18</b>, barely above chance, so the fingerprint is largely
    <b class="red">club- and roster-bound</b>, not coach-bound.</div>
  </div>
  {foot(8)}
</section>

<!-- 8 P2 LIVE -->
<section class="slide" data-t="60">
  <div class="hgroup fx"><div class="kick">Problem 2 · Tactical archetypes <b>· live</b></div>
    <h2 style="font-size:48px">Styles emerge as a continuum, not boxes</h2></div>
  <div class="canvas">
    <div class="lg fx"><span><i style="background:var(--c1)"></i> Possession (Bar&ccedil;a, Spain, Portugal, Croatia)</span>
      <span><i style="background:var(--c0)"></i> Direct (World-Cup sides)</span>
      <span style="color:var(--mut2);font-weight:600">· 774 networks · UMAP, k=2 clusters</span></div>
    <div class="figframe" style="top:40px"><div id="p2Chart" class="fig"></div></div>
  </div>
  {foot(9)}
</section>

<!-- 9 P3 LIVE -->
<section class="slide" data-t="80">
  <div class="hgroup fx"><div class="kick">Problem 3 · Player role space <b>· live demo</b></div>
    <h2 style="font-size:46px">Type a player, watch its structural twins light up</h2></div>
  <div class="canvas">
    <div class="sbar fx"><b>Find a player:</b>
      <input list="plist" id="pSearch" placeholder="try: Xavi · Modric · Busquets">
      <datalist id="plist"></datalist><button id="pClear">Clear</button>
      <span class="hint" id="pInfo">Role purity <b style="color:#16161d">0.88</b> among the 5 nearest neighbours, 3× chance.</span></div>
    <div class="lg fx" style="margin-top:6px"><span><i style="background:var(--gk)"></i> GK</span><span><i style="background:var(--def)"></i> DEF</span>
      <span><i style="background:var(--mid)"></i> MID</span><span><i style="background:var(--fwd)"></i> FWD</span></div>
    <div class="figframe" style="top:96px"><div id="p3Chart" class="fig"></div></div>
  </div>
  {foot(10)}
</section>

<!-- 10 ML PIPELINE -->
<section class="slide" data-t="50">
  <div class="hgroup fx"><div class="kick">Under the hood · the machine-learning pipeline</div>
    <h2 style="font-size:44px;max-width:none;white-space:nowrap">Graphs &rarr; features &rarr; models &rarr; insight</h2></div>
  <div class="canvas">
    <div class="flow fx" style="margin-top:8px">
      <div class="node">Passing<br>graph G</div><span class="ar">→</span>
      <div class="node">&phi;(G) · 50 features<small>centrality · global · motifs</small></div><span class="ar">→</span>
      <div class="node">ML models</div><span class="ar">→</span>
      <div class="node">Coach /<br>cluster / role</div>
    </div>
    <div class="trip fx">
      <div class="b"><b>Supervised (P1).</b> Logistic Regression, Random Forest &amp; XGBoost; class-weighted; stratified + season-grouped CV to expose leakage.</div>
      <div class="b"><b>Unsupervised (P2).</b> k-means &amp; Ward; model selection by silhouette &amp; elbow; UMAP projection.</div>
      <div class="b"><b>Representation (P3).</b> Per-player ego-vectors → PCA &amp; UMAP; k-NN retrieval for structural analogues.</div>
    </div>
    <div class="body fx" style="text-align:center;margin-top:28px;font-size:19px">
      One recurring finding: <span class="em">centrality</span>, not motifs, carries the discriminative signal.</div>
  </div>
  {foot(11)}
</section>

<!-- 11 FINDINGS -->
<section class="slide accent" data-t="40">
  <div class="hgroup fx"><div class="kick">What we learned</div><h2>Three findings</h2></div>
  <div class="canvas">
    <div class="cards">
      <div class="card fx"><div class="tag">Coaches</div><div class="big">Club &gt; coach</div>
        <p>A &ldquo;fingerprint&rdquo; is mostly the <span style="color:#fff">squad</span>. Under honest evaluation the coach signal nearly vanishes, a result naive accuracy hid.</p></div>
      <div class="card fx"><div class="tag">Styles</div><div class="big">A continuum</div>
        <p>Tactics live on a <span style="color:#fff">possession&ndash;direct</span> axis, recovered with no labels and cutting across leagues, not discrete boxes.</p></div>
      <div class="card fx"><div class="tag">Players</div><div class="big">Roles are real</div>
        <p><span style="color:#fff">0.88</span> position purity. Xavi&rsquo;s neighbours <i>are</i> the deep playmakers; Modri&cacute; matches creators across nations.</p></div>
    </div>
  </div>
  {foot(12)}
</section>

<!-- 12 IMPACT + FUTURE -->
<section class="slide" data-t="50">
  <div class="hgroup fx" style="height:130px"><div class="kick">Why it matters · where it goes next</div>
    <h2>From a class project to a scouting tool</h2></div>
  <div class="canvas" style="top:222px"><div style="position:absolute;inset:0;display:grid;grid-template-columns:1fr 1fr;gap:64px;align-content:center">
    <div class="fx"><div class="colh">Applications today</div>
      <ul class="k" style="font-size:20px;line-height:2.1">
        <li><b>Recruitment:</b> &ldquo;find players who fill this role&rdquo; via k-NN in role space</li>
        <li><b>Opponent scouting:</b> auto-summarise a team&rsquo;s passing style</li>
        <li><b>Cross-context analogues:</b> Modri&cacute;-like creators across leagues</li></ul></div>
    <div class="fx"><div class="colh">Future work</div>
      <ul class="k" style="font-size:20px;line-height:2.1">
        <li><b>De-bias the corpus:</b> add Real Madrid &amp; Atl&eacute;tico as their own teams</li>
        <li><b>Temporal networks:</b> 15-min windows for in-match adaptation</li>
        <li><b>Graph neural nets:</b> learn the embedding end-to-end</li>
        <li><b>Player2Vec:</b> contrastive role embeddings for transfer search</li></ul></div>
  </div></div>
  {foot(13)}
</section>

<!-- 13 CLOSE -->
<section class="slide accent" data-t="10">
  <img src="YorkuLogo.png" class="ylogo tl fx" alt="York University">
  <div class="center">
    <h2 class="fx" style="font-size:66px;max-width:none;color:var(--ink)">Tactical&nbsp;DNA</h2>
    <div class="lead fx" style="max-width:none;color:var(--red);font-size:26px;margin-top:12px;font-weight:600">
      Qualitative tactics, made measurable.</div>
    <div class="fx" style="margin-top:30px;font-size:22px;font-weight:800;color:var(--ink)">Thank you. Questions?</div>
    <a class="repo fx" href="https://github.com/mahyar-jbr/tactical-dna" target="_blank">
      <svg viewBox="0 0 24 24" width="22" height="22" fill="#16161d"><path d="M12 2a10 10 0 0 0-3.16 19.49c.5.09.68-.22.68-.48v-1.7c-2.78.6-3.37-1.34-3.37-1.34-.45-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.6.07-.6 1 .07 1.53 1.03 1.53 1.03.9 1.53 2.36 1.09 2.94.83.09-.65.35-1.09.63-1.34-2.22-.25-4.55-1.11-4.55-4.94 0-1.09.39-1.98 1.03-2.68-.1-.25-.45-1.27.1-2.65 0 0 .84-.27 2.75 1.02a9.5 9.5 0 0 1 5 0c1.91-1.29 2.75-1.02 2.75-1.02.55 1.38.2 2.4.1 2.65.64.7 1.03 1.59 1.03 2.68 0 3.84-2.34 4.69-4.57 4.94.36.31.68.92.68 1.85v2.74c0 .27.18.58.69.48A10 10 0 0 0 12 2Z"/></svg>
      <span><b>github.com/mahyar-jbr/tactical-dna</b><small>Run it yourself · use your own data · full paper &amp; outputs</small></span>
    </a>
    <div class="fx" style="margin-top:18px;font-size:16px;color:var(--mut)">
      Mahyar Jaberi · mhyrjbr@my.yorku.ca · EECS 4414, York University</div>
  </div>
</section>
"""


# ===========================================================================
# JS
# ===========================================================================
_JS = r"""
const D=window.__DATA__;
const RED='#E31837',RED2='#b3142c',INK='#16161d',MUT='#5a5a6a';
const LINE={GK:'#8a8f9c',DEF:'#2f6fd0',MID:'#1f9e57',FWD:'#e1334a'},CL={0:'#2f6fd0',1:'#ef7a2e'};
const slides=[...document.querySelectorAll('.slide')];let cur=0;

// scale the fixed 1280x720 stage to fill the screen
function fit(){const s=Math.min(innerWidth/1280,innerHeight/720);
  document.getElementById('stage').style.transform=`translate(-50%,-50%) scale(${s})`;}
addEventListener('resize',fit);fit();

// dots
const dwrap=document.getElementById('dots');
slides.forEach(()=>{const d=document.createElement('div');d.className='d';dwrap.appendChild(d);});
const dots=[...dwrap.children];

function show(i){
  if(i<0||i>=slides.length)return;
  slides[cur].classList.remove('on');cur=i;const s=slides[cur];s.classList.add('on');
  dots.forEach((d,j)=>d.classList.toggle('on',j===cur));
  document.getElementById('bar').style.width=((cur+1)/slides.length*100)+'%';
  const fx=[...s.querySelectorAll('.fx')];
  fx.forEach(f=>f.classList.remove('show'));
  fx.forEach((f,k)=>setTimeout(()=>f.classList.add('show'),70+k*95));
  if(s.querySelector('#clasicoChart')&&!s.dataset.d){drawClasico();s.dataset.d=1;}
  if(s.querySelector('#confChart')&&!s.dataset.d){drawConf();s.dataset.d=1;}
  if(s.querySelector('#transferChart')&&!s.dataset.d){drawTransfer();s.dataset.d=1;}
  if(s.querySelector('#p2Chart')&&!s.dataset.d){drawP2();s.dataset.d=1;}
  if(s.querySelector('#p3Chart')&&!s.dataset.d){drawP3();s.dataset.d=1;}
  if(s.querySelector('[data-count]'))counts(s);
}
const next=()=>show(Math.min(cur+1,slides.length-1));
const prev=()=>show(Math.max(cur-1,0));
addEventListener('keydown',e=>{
  if(['ArrowRight','PageDown',' '].includes(e.key))next();
  else if(['ArrowLeft','PageUp'].includes(e.key))prev();
  else if(e.key==='f'||e.key==='F'){document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen();}
  else if(e.key==='Home')show(0);else if(e.key==='End')show(slides.length-1);
});
dots.forEach((d,i)=>d.addEventListener('click',()=>show(i)));
addEventListener('click',e=>{if(e.target.closest('input,button,.fig,#dots,a'))return;
  e.clientX>innerWidth*.6?next():e.clientX<innerWidth*.4?prev():0;});
addEventListener('load',()=>{show(0);buildPlist();});

function counts(s){s.querySelectorAll('[data-count]').forEach(el=>{
  if(el.dataset.done)return;el.dataset.done=1;const end=+el.dataset.count,t0=performance.now();
  (function step(t){const p=Math.min((t-t0)/1100,1),e=1-Math.pow(1-p,3);
    el.textContent=Math.round(end*e);if(p<1)requestAnimationFrame(step);})(performance.now());});}

// ---- Plotly: presentation-grade (no axes, no grid, transparent) ----
function bareLayout(extra){return Object.assign({
  paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
  margin:{l:8,r:8,t:8,b:8},showlegend:false,
  xaxis:{visible:false,showgrid:false,zeroline:false,fixedrange:true,automargin:true},
  yaxis:{visible:false,showgrid:false,zeroline:false,fixedrange:true,automargin:true},
  font:{family:'Inter,Arial',color:'#16161d',size:15},
  hoverlabel:{font:{size:15,color:'#16161d'},bgcolor:'#fff',bordercolor:'#16161d'}
},extra||{});}
// Toolbar visible on hover (so you can zoom during the live demo), scroll-zoom on.
// modeBarButtons trimmed to the useful ones; dark colour so it shows on white.
const CFG={displayModeBar:'hover',responsive:true,scrollZoom:true,displaylogo:false,
  modeBarButtonsToRemove:['select2d','lasso2d','autoScale2d','toImage'],
  modeBarButtonsToAdd:[]};

function drawClasico(){
  const N=D.clasico.nodes,E=D.clasico.edges,id={};N.forEach((n,i)=>id[n.id]=i);
  const pr=N.map(n=>n.pr),pmin=Math.min(...pr),pmax=Math.max(...pr);
  const ex=[],ey=[];E.forEach(e=>{const a=N[id[e.s]],b=N[id[e.t]];ex.push(a.x,b.x,null);ey.push(a.y,b.y,null);});
  const edge={x:ex,y:ey,mode:'lines',type:'scatter',hoverinfo:'skip',line:{color:'rgba(20,20,40,.14)',width:1}};
  const node={x:N.map(n=>n.x),y:N.map(n=>n.y),mode:'markers+text',type:'scatter',
    text:N.map(n=>n.label),textposition:'top center',textfont:{size:13,color:'#16161d'},
    marker:{size:N.map(n=>18+62*(n.pr-pmin)/(pmax-pmin+1e-9)),color:pr,
      colorscale:[[0,'#ffd9a0'],[.5,'#E31837'],[1,'#7a0f1f']],
      line:{color:'rgba(255,255,255,.85)',width:1.2}},
    hovertext:N.map(n=>`<b>${n.label}</b> · PR ${n.pr.toFixed(3)}`),hoverinfo:'text'};
  Plotly.newPlot('clasicoChart',[edge,node],bareLayout({yaxis:{visible:false,showgrid:false,zeroline:false,fixedrange:true,scaleanchor:'x'}}),CFG);
  const sz=node.marker.size;Plotly.restyle('clasicoChart',{'marker.size':[sz.map(()=>4)]},[1]);
  let s=0;const t=setInterval(()=>{s+=.1;if(s>=1){s=1;clearInterval(t);}
    Plotly.restyle('clasicoChart',{'marker.size':[sz.map(v=>4+(v-4)*s)]},[1]);},26);
}

function drawConf(){
  const C=D.confmat,cls=C.classes,z=C.cmn;
  const txt=z.map(r=>r.map(v=>v>=.01?v.toFixed(2):''));
  const tr={z:z,x:cls,y:cls,type:'heatmap',xgap:3,ygap:3,zmin:0,zmax:1,showscale:false,
    colorscale:[[0,'#f4f5f7'],[.5,'#f3a3b0'],[1,'#E31837']],
    text:txt,texttemplate:'%{text}',textfont:{size:12,color:'#16161d'},
    hovertemplate:'true %{y} → %{x}: %{z}<extra></extra>'};
  Plotly.newPlot('confChart',[tr],bareLayout({margin:{l:78,r:6,t:6,b:64},
    xaxis:{visible:true,tickangle:-40,tickfont:{size:12,color:'#5a5a6a'},showgrid:false,fixedrange:true},
    yaxis:{visible:true,autorange:'reversed',tickfont:{size:12,color:'#5a5a6a'},showgrid:false,fixedrange:true,scaleanchor:null}}),CFG);
}

function drawTransfer(){
  const T=D.transfer,M=[['LR','#2f6fd0'],['RF','#1f9e57'],['XGB','#E31837']];
  const tr=M.map(([m,c])=>({x:T.coaches,y:T[m],type:'bar',name:m,marker:{color:c},
    hovertemplate:'%{x}<br>'+m+': %{y:.2f}<extra></extra>'}));
  Plotly.newPlot('transferChart',tr,bareLayout({barmode:'group',showlegend:true,
    legend:{orientation:'h',y:1.14,x:.5,xanchor:'center',font:{color:'#5a5a6a',size:14}},
    margin:{l:40,r:10,t:30,b:50},
    yaxis:{visible:true,range:[0,1],tickfont:{size:12,color:'#5a5a6a'},showgrid:true,gridcolor:'rgba(20,20,40,.08)',fixedrange:true,scaleanchor:null},
    xaxis:{visible:true,tickfont:{size:13,color:'#5a5a6a'},showgrid:false,fixedrange:true},
    shapes:[{type:'line',x0:-.5,x1:2.5,y0:T.chance,y1:T.chance,line:{color:RED,width:2,dash:'dash'}}],
    annotations:[{x:2.4,y:T.chance+.05,text:'chance 0.125',showarrow:false,font:{color:RED,size:12},xanchor:'right'}]}),CFG);
}

function drawP2(){
  const t=[0,1].map(c=>{const idx=D.p2.cl.map((v,i)=>v===c?i:-1).filter(i=>i>=0);
    return {x:idx.map(i=>D.p2.x[i]),y:idx.map(i=>D.p2.y[i]),mode:'markers',type:'scatter',
      marker:{size:9,color:CL[c],opacity:.82,line:{width:0}},
      customdata:idx.map(i=>[D.p2.team[i],D.p2.comp[i]]),
      hovertemplate:'<b>%{customdata[0]}</b> · %{customdata[1]}<extra></extra>'};});
  const lay=bareLayout();lay.annotations=[];
  D.p2.lm.forEach((lm,i)=>{if(lm)lay.annotations.push({x:D.p2.x[i],y:D.p2.y[i],text:'<b>'+lm+'</b>',
    showarrow:true,arrowcolor:'#16161d',arrowwidth:2,arrowhead:2,ax:36,ay:-32,
    font:{size:14,color:'#fff'},bgcolor:RED,borderpad:4,bordercolor:'#fff',borderwidth:1});});
  Plotly.newPlot('p2Chart',t,lay,CFG);
}

let p3hl=null;
function drawP3(){
  const ls=['GK','DEF','MID','FWD'];
  const tr=ls.map(L=>{const idx=D.p3.line.map((v,i)=>v===L?i:-1).filter(i=>i>=0);
    return {name:L,x:idx.map(i=>D.p3.x[i]),y:idx.map(i=>D.p3.y[i]),mode:'markers',type:'scatter',
      marker:{size:11,color:LINE[L],opacity:.85,line:{width:0}},
      customdata:idx.map(i=>[D.p3.name[i],D.p3.pos[i],D.p3.team[i],D.p3.app[i]]),
      hovertemplate:'<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[2]} · %{customdata[3]} apps<extra></extra>'};});
  tr.push({x:[],y:[],mode:'lines',type:'scatter',hoverinfo:'skip',line:{color:'#16161d',width:1.4}});
  tr.push({x:[],y:[],mode:'markers',type:'scatter',hoverinfo:'skip',marker:{size:20,color:'rgba(0,0,0,0)',line:{color:'#16161d',width:3}}});
  // a faint label marking the goalkeeper island (it sits far from the outfield)
  tr.push({x:[10.05],y:[9.4],mode:'text',type:'scatter',hoverinfo:'skip',
    text:['goalkeepers'],textfont:{size:13,color:'#8a8f9c'},textposition:'top center'});
  const lay=bareLayout({margin:{l:20,r:20,t:20,b:20}});
  lay.annotations=['Xavi','Andrés Iniesta','Sergio Busquets','Luka Modrić'].map((nm,k)=>{
    const i=D.p3.name.indexOf(nm);const o=[[60,46],[66,2],[60,-44],[-78,56]][k];
    return i<0?null:{x:D.p3.x[i],y:D.p3.y[i],text:'<b>'+nm+'</b>',showarrow:true,arrowcolor:'#16161d',
      arrowwidth:2,arrowhead:2,ax:o[0],ay:o[1],font:{size:13,color:'#fff'},
      bgcolor:RED,borderpad:4,bordercolor:'#fff',borderwidth:1};}).filter(Boolean);
  Plotly.newPlot('p3Chart',tr,lay,CFG);
  p3hl={link:tr.length-2,ring:tr.length-1};
  document.getElementById('p3Chart').on('plotly_click',ev=>{
    const nm=ev.points[0].customdata&&ev.points[0].customdata[0];if(nm)hi(nm);});
}
function hi(name){const i=D.p3.name.indexOf(name);if(i<0)return;
  const nb=D.p3.nbr[i],lx=[],ly=[],px=[],py=[];
  nb.forEach(j=>{lx.push(D.p3.x[i],D.p3.x[j],null);ly.push(D.p3.y[i],D.p3.y[j],null);px.push(D.p3.x[j]);py.push(D.p3.y[j]);});
  px.push(D.p3.x[i]);py.push(D.p3.y[i]);
  Plotly.restyle('p3Chart',{x:[lx],y:[ly]},[p3hl.link]);
  Plotly.restyle('p3Chart',{x:[px],y:[py]},[p3hl.ring]);
  document.getElementById('pInfo').innerHTML='<b style="color:#16161d">'+name+'</b> → '+nb.slice(0,6).map(j=>D.p3.name[j]).join(', ')+'…';
  document.getElementById('pSearch').value=name;}
function buildPlist(){const dl=document.getElementById('plist');if(!dl)return;
  D.p3.name.forEach(n=>{const o=document.createElement('option');o.value=n;dl.appendChild(o);});
  document.getElementById('pSearch').addEventListener('change',e=>hi(e.target.value));
  document.getElementById('pClear').addEventListener('click',()=>{if(!p3hl)return;
    Plotly.restyle('p3Chart',{x:[[]],y:[[]]},[p3hl.link,p3hl.ring]);
    document.getElementById('pSearch').value='';
    document.getElementById('pInfo').innerHTML='Role purity <b style="color:#16161d">0.88</b> among the 5 nearest neighbours, 3× chance.';});}

"""


# ===========================================================================
_DOC = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tactical DNA · Mahyar Jaberi · EECS 4414</title>
<style>/*CSS*/</style>
<script>/*PLOTLY*/</script>
</head><body>
<div id="stage">/*SLIDES*/</div>
<div id="chrome"><div id="dots"></div><div id="bar"></div></div>
<script>window.__DATA__=/*DATA*/;</script>
<script>/*JS*/</script>
</body></html>"""
