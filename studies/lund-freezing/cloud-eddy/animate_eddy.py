"""Render a perspective view of the x–z slice directly from archived GPU output.

Run with the study interpreter: python animate_eddy.py
"""
import json
from pathlib import Path
import subprocess
import tempfile

from matplotlib import get_data_path
import numpy as np
import pandas as pd
from PIL import Image, ImageColor, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent / "results"
WIDTH, HEIGHT = 1120, 630
SCALE = 2
OUTPUT_SIZE = (1680, 945)
STEP = 20  # seconds between archived particle positions
TRAIL_SECONDS = 240
BG, INK, MUTED = "#fcfbf7", "#173039", "#48616a"
LINE, FLOW, PANEL = "#c4cfca", "#3d7080", "#f1f3ed"
RESIDENT, INCOMING, TAG = "#66358a", "#a44118", "#101b23"
WATER = [(246, 249, 244), (193, 228, 223), (113, 190, 190),
         (53, 154, 164), (19, 117, 139), (15, 84, 116)]
TAG_ID = 15887
FONT_DIR = Path(get_data_path()) / "fonts" / "ttf"


def font(size, bold=False):
    face = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(FONT_DIR / face), size * SCALE)


FONTS = {(size, bold): font(size, bold) for size, bold in
         [(12, False), (13, False), (14, False), (15, False), (16, False),
          (14, True), (16, True), (19, True), (24, True), (28, True)]}


def point(x, z):
    """Oblique projection of the one simulated plane. All data use this plane."""
    return 78 + 656*x/1600, 460 - 39*x/1600 - 281*z/600


def water_color(value):
    i = np.interp(np.clip(value, 0, .45), [0,.05,.12,.22,.33,.45], np.arange(6))
    j = min(int(i), 4)
    return tuple(round(a+(b-a)*(i-j)) for a,b in zip(WATER[j], WATER[j+1]))


class Canvas:
    def __init__(self):
        self.image = Image.new("RGB", (WIDTH*SCALE, HEIGHT*SCALE), BG)
        self.d = ImageDraw.Draw(self.image)

    def xy(self, points):
        return [(round(SCALE*x), round(SCALE*y)) for x,y in points]

    def line(self, points, color, width=1):
        self.d.line(self.xy(points), fill=color, width=round(SCALE*width), joint="curve")

    def poly(self, points, color):
        self.d.polygon(self.xy(points), fill=color)

    def text(self, x, y, label, size=14, color=INK, bold=False, anchor="la"):
        self.d.text((x*SCALE,y*SCALE), label, font=FONTS[size,bold],
                    fill=color, anchor=anchor)

    def dot(self, x, y, radius, fill, outline=None, width=1):
        self.d.ellipse(self.xy([(x-radius,y-radius),(x+radius,y+radius)]),
                       fill=fill,outline=outline,width=round(width*SCALE))

    def box(self, x1,y1,x2,y2,color):
        self.d.rectangle(self.xy([(x1,y1),(x2,y2)]),fill=color)

    def event(self, x, y, incoming):
        if incoming:
            self.poly([(x,y-3.8),(x+3.4,y+2.9),(x-3.4,y+2.9)], INCOMING)
        else:
            self.dot(x,y,2.8,RESIDENT)

    def arrow(self, x, y, dx, dy):
        tip=(x+dx,y+dy)
        length=np.hypot(dx,dy)
        nx,ny=dx/length,dy/length
        stem=(tip[0]-4*nx,tip[1]-4*ny)
        self.line([(x,y),tip],FLOW,1.3)
        self.poly([tip,(stem[0]-2*ny,stem[1]+2*nx),
                       (stem[0]+2*ny,stem[1]-2*nx)],FLOW)


def tracer(c, samples, color, selected):
    """Join only live samples on the same side of the periodic boundary."""
    if not len(samples):
        return
    # Show only the continuous tail behind the current position.
    wraps = np.flatnonzero(np.abs(np.diff(samples[:, 0])) > 800)
    if len(wraps):
        samples = samples[wraps[-1]+1:]
    screen = np.asarray([point(x,z) for x,z in samples])
    x,y = screen[-1]
    if len(screen)>1:
        c.line(screen, "white", 4 if selected else 3)
        c.line(screen, color, 2.4 if selected else 1.8)
        # Direct the head along the latest resolved motion. A ring remains
        # visible around the selected core. The arrowhead sits on its tail.
        direction = screen[-1]-screen[-2]
        length = np.linalg.norm(direction)
        if length>0.01:
            unit = direction/length
            normal = np.array([-unit[1],unit[0]])
            tip = screen[-1] - (7 if selected else 0)*unit
            base = tip-7*unit
            c.poly([tip,base+3.4*normal,base-3.4*normal],color)
    else:
        c.dot(x,y,2.6,color,"white")


def render():
    fields=pd.read_csv(ROOT/"fields.csv")
    all_paths=pd.read_csv(ROOT/"tagged_paths.csv.gz").sort_values(["id","time_s"])
    tracks={}
    for core, group in all_paths.groupby("id"):
        np.testing.assert_array_equal(group.time_s, np.arange(STEP,21601,STEP))
        tracks[core]=(group[["x_m","z_m"]].to_numpy(),group.fate.to_numpy(),
                      INCOMING if group.birth_s.iloc[0]>=0 else RESIDENT)
    assert len(tracks)==6
    paths=all_paths.query("id == @TAG_ID")
    events=pd.read_csv(ROOT/"freezing_ids.csv")
    meta,=json.loads((ROOT/"tracks_runs.json").read_text())
    history=pd.read_csv(ROOT/"tracks_history.csv").set_index("t_h")
    history.index=np.rint(history.index*3600).astype(int)
    assert meta['seed']==20261125 and meta['n']==10 and meta['hours']==6
    assert len(history)==37 and (fields.groupby('t_s').size()==100).all()
    for seconds,field in fields.groupby('t_s'):
        np.testing.assert_allclose(field.liquid_g_m3.sum()*6,history.loc[seconds,'LWP'])
    tag_event=events.loc[events.id==TAG_ID].iloc[0]
    live=paths[paths.fate==1]
    first_ice=live[live.phase==2].iloc[0]
    assert first_ice.time_s==tag_event.first_freeze_s
    states=paths.loc[paths.phase.diff().ne(0)|paths.fate.diff().ne(0)].iloc[1:]
    labels=["Enters as a dry core","Forms a liquid droplet","Dries completely",
            "Forms a droplet again","Freezes near 564 m","Falls out below the cloud"]
    assert list(zip(states.fate,states.phase))==[(1,0),(1,1),(1,0),(1,1),(1,2),(3,2)]
    blocks={int(t):g for t,g in fields.groupby('t_s')}
    phase_colors={0:"#96703d",1:"#0b7882",2:"#634599"}

    def frame(t):
        c=Canvas()
        c.text(36,25,"How an entraining cloud forms ice",28,bold=True)
        c.text(37,67,"Particle-resolved GPU simulation  ·  Reference eddy with fresh dust-like inflow",16,MUTED)
        c.text(1082,31,f"{t//3600:d}:{(t%3600)//60:02d}:{t%60:02d}",24,bold=True,anchor="ra")
        c.line([(36,99),(1082,99)],LINE)
        c.text(38,105,"Cloud cross-section",19,bold=True)
        c.text(38,132,"1,600 m wide × 600 m high",13,MUTED)
        corners=[point(0,0),point(1600,0),point(1600,600),point(0,600)]
        back=[(x+27,y-19) for x,y in corners]
        for i,j,col in [(1,2,"#d9e5df"),(2,3,"#e7ede7")]:
            c.poly([corners[i],corners[j],back[j],back[i]],col)
            c.line([back[i],back[j],corners[j]],LINE)
        # Label the single out-of-plane cell on its exaggerated perspective edge.
        c.line([(748,129),(727,114),(700,114)],MUTED,1)
        c.text(695,114,"1 m into page",13,MUTED,anchor="rm")
        field=blocks[t//600*600]
        for row in field.itertuples():
            x,z=row.x,row.z
            c.poly([point(x-80,z-30),point(x+80,z-30),
                    point(x+80,z+30),point(x-80,z+30)],water_color(row.liquid_g_m3))
        c.line(corners+[corners[0]],"#76968f",1.3)
        # Velocity prescribed by cloud.geometry, evaluated at regularly spaced points.
        for x in np.linspace(100,1500,8):
            for z in np.linspace(75,525,4):
                k=2*np.pi/1600; eta=z/600
                u=-np.sin(k*x)*(meta['eddy']*np.pi*np.cos(np.pi*eta)+3*meta['entrain']*eta**2)/(k*600)
                w=np.cos(k*x)*(meta['eddy']*np.sin(np.pi*eta)+meta['entrain']*eta**3)
                a,b=point(x,z); d,e=point(x+90*u,z+90*w)
                c.arrow(a,b,d-a,e-b)
        a,b=point(800,600)
        c.text(a+12,b-36,"Dry air enters",13,FLOW)
        c.arrow(a,b-28,0,31)
        for z in (0,300,600):
            x,y=point(0,z);c.text(x-8,y,f"{z} m",12,MUTED,anchor="rm")
        for x in (0,800,1600):
            a,b=point(x,0);c.text(a,b+7,f"{x:,} m",12,MUTED,anchor="ma")
        recent=events[(events.first_freeze_s>t-120)&(events.first_freeze_s<=t)]
        for e in recent.itertuples():
            c.event(*point(e.x_m,e.z_m),e.source=="incoming")
        ix=t//STEP-1
        for core in sorted(tracks, key=lambda core: core==TAG_ID):
            coords,fates,color=tracks[core]
            if fates[ix]!=1:
                continue
            lo=max(0,ix-TRAIL_SECONDS//STEP)
            valid=fates[lo:ix+1]==1
            tracer(c,coords[lo:ix+1][valid],color,core==TAG_ID)
        state=paths.loc[paths.time_s==t].iloc[0]
        if state.fate==1:
            a,b=point(state.x_m,state.z_m)
            c.dot(a,b,6.5,"white",TAG,2);c.dot(a,b,3,phase_colors[state.phase])
        c.text(40,507,"Simulated liquid water (g/m³)",14,bold=True)
        for i in range(214):
            c.line([(40+i,539),(40+i,551)],water_color(.45*i/213),2)
        for i,value in [(0,"0"),(95,"0.20"),(213,"0.45")]:
            c.text(40+i,559,value,12,MUTED,anchor="ma")
        c.text(301,507,"First freezes in the past 2 minutes",14,bold=True)
        for x,label,new in [(305,"Resident",False),(425,"Incoming",True)]:
            c.event(x,545,new);c.text(x+11,535,label,14)
        c.arrow(597,545,23,-8);c.text(592,507,"Prescribed airflow",14,bold=True)
        c.text(37,599,"Particle trails",14,bold=True)
        c.text(153,599,"Purple = resident   ·   Orange = incoming   ·   Ring = core 15887",14,MUTED)
        c.line([(36,590),(1082,590)],LINE)

        c.box(790,114,1082,574,PANEL)
        c.text(809,132,"One incoming particle",19,bold=True)
        c.dot(816,176,6,"white",TAG,2);c.dot(816,176,2,TAG)
        c.text(829,165,"Tracked core 15887",14,MUTED)
        current=-1
        for i,(s,label) in enumerate(zip(states.itertuples(),labels)):
            y=208+32*i
            if t>=s.time_s:current=i
            on=t>=s.time_s
            c.dot(815,y+8,3.6,TAG if on else PANEL,TAG if on else "#aab8b1")
            c.text(827,y,f"{s.time_s/3600:.2f} h",13,INK if on else MUTED)
            c.text(883,y,label,13,INK if on else MUTED)
        c.text(810,414,["Awaiting entry",*labels][current+1],16,bold=True)
        c.text(810,450,"Particle height (0–600 m)",13,MUTED)
        left,base,wide,tall=813,541,240,65
        c.line([(left,base-tall),(left,base),(left+wide,base)],"#a9b8b1")
        recorded=live[live.time_s<=t]
        coords=[(left+wide*b.time_s/21600,base-tall*b.z_m/600) for b in recorded.itertuples()]
        if len(coords)>1:c.line(coords,TAG,1.4)
        if state.fate==1:
            c.dot(*coords[-1],3,phase_colors[state.phase])
        c.text(left,548,"0 h",12,MUTED);c.text(left+wide,548,"6 h",12,MUTED,anchor="ra")
        return c.image.resize(OUTPUT_SIZE,Image.Resampling.LANCZOS)

    times=list(range(600,21601,STEP))
    keys=[frame(t) for t in (600,9000,15000,21600)]
    width,height=OUTPUT_SIZE
    sample=Image.new("RGB",(width,height*len(keys)))
    for i,im in enumerate(keys):
        sample.paste(im,(0,i*height))
    colors=sample.quantize(232).getpalette()[:232*3]
    fixed=[RESIDENT,INCOMING,TAG,FLOW,INK,MUTED,PANEL,BG,LINE,*phase_colors.values(),"white"]
    for color in fixed:
        colors.extend(ImageColor.getrgb(color))
    colors.extend([0]*(768-len(colors)))
    palette=Image.frombytes("RGB",(16,16),bytes(colors)).convert("RGBA")
    palette.putpixel((15,15),(0,0,0,0))
    gif=ROOT/"cloud-dynamics.gif"
    # Encode transparent frame differences as we render to bound memory usage.
    with tempfile.TemporaryDirectory(dir=ROOT) as temp:
        palette_path=Path(temp)/"palette.png"
        output=Path(temp)/gif.name
        palette.save(palette_path)
        command=["ffmpeg","-v","error","-filter_complex_threads","1","-f","rawvideo",
                 "-pixel_format","rgb24","-video_size",f"{width}x{height}","-framerate","25",
                 "-i","pipe:0","-i",str(palette_path),"-filter_complex",
                 "[0:v][1:v]paletteuse=dither=none:diff_mode=rectangle",
                 "-loop","0","-final_delay","160",str(output)]
        with subprocess.Popen(command,stdin=subprocess.PIPE) as proc:
            for t in times:
                proc.stdin.write(frame(t).tobytes())
            proc.stdin.close()
            if proc.wait()!=0:
                raise RuntimeError("GIF encoding failed")
        with Image.open(output) as movie:
            assert movie.n_frames==len(times) and movie.size==OUTPUT_SIZE
            assert movie.info['loop']==0
        output.replace(gif)
    print(f"{gif.name}: {len(times)} frames from {times[0]} to {times[-1]} s; {gif.stat().st_size:,} bytes")


if __name__ == "__main__":
    render()
