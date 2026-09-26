import streamlit as st
import datetime
import time
import base64
import s3fs
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio
import os

st.set_page_config(page_title="GOES-16/19 Satellite Viewer", layout="wide", initial_sidebar_state="expanded")

st.markdown("<h2 style='color:#38bdf8;'>🛰️ Visualizador de Satélites GOES-16 / GOES-19</h2>", unsafe_allow_html=True)

# ==========================================
# PALETAS DE CORES E LÓGICA DE PROCESSAMENTO
# ==========================================

def create_cirrus_colormap_soft():
    colors = [
        (0.00, '#0c0f12'), (0.03, '#261f18'), (0.08, '#6e5437'),
        (0.16, '#c49a5c'), (0.28, '#e8cf9b'), (0.42, '#cce0f5'),
        (0.65, '#ffffff'), (1.00, '#ffffff')
    ]
    return mcolors.LinearSegmentedColormap.from_list('cirrus_soft_ref', colors)

def create_exact_custom_ir():
    raw_table = [
        (-90, (144, 36, 145)), (-80, (205, 205, 205)), (-70, (50, 2, 4)),
        (-60, (254, 77, 5)), (-50, (165, 252, 13)), (-40, (2, 181, 34)),
        (-30, (4, 60, 146)), (-25, (11, 185, 216)), (-20, (188, 188, 186)),
        (-10, (162, 162, 162)), (0, (135, 135, 135)), (10, (109, 109, 109)),
        (20, (83, 83, 83)), (30, (56, 56, 56)), (40, (30, 30, 30))
    ]
    raw_table.sort(key=lambda x: x[0])
    min_t, max_t = -90.0, 40.0
    colors = [( (t - min_t) / (max_t - min_t), (r/255.0, g/255.0, b/255.0) ) for t, (r, g, b) in raw_table]
    return mcolors.LinearSegmentedColormap.from_list('exact_custom_ir', colors)

def fetch_goes_data(sat, band, date_obj, hour_utc, minute_utc):
    fs = s3fs.S3FileSystem(anon=True)
    doy = date_obj.strftime('%j')
    year = date_obj.strftime('%Y')
    path = f"{sat}/ABI-L2-CMIPF/{year}/{doy}/{hour_utc:02d}/"
    files = fs.ls(path)
    target_files = [f for f in files if f"M6{band}" in f or f"M3{band}" in f or f"M4{band}" in f]
    
    if not target_files:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em: s3://{path}")
    
    best_file = target_files[0]
    min_diff = 999
    for f in target_files:
        try:
            fn = f.split('/')[-1]
            f_min = int(fn.split('_s')[1][9:11])
            diff = abs(f_min - minute_utc)
            if diff < min_diff:
                min_diff = diff
                best_file = f
        except Exception:
            continue
    return fs, best_file

def overlay_glm_lightning(fs, sat, date_obj, hour_utc, minute_utc, bbox, ax, ccrs_plate, band='C13'):
    try:
        doy = date_obj.strftime('%j')
        year = date_obj.strftime('%Y')
        path = f"{sat}/GLM-L2-LCFA/{year}/{doy}/{hour_utc:02d}/"
        files = fs.ls(path)
        if not files: return
            
        selected_lats, selected_lons = [], []
        for f in files:
            try:
                f_min = int(f.split('/')[-1].split('_s')[1][9:11])
                if abs(f_min - minute_utc) > 7: continue
            except Exception: pass
                
            with fs.open(f) as s3_file:
                with xr.open_dataset(s3_file, engine='h5netcdf') as ds_glm:
                    lats = ds_glm['group_lat'].values if 'group_lat' in ds_glm else ds_glm['flash_lat'].values
                    lons = ds_glm['group_lon'].values if 'group_lon' in ds_glm else ds_glm['flash_lon'].values
                    mask = (lons >= bbox['min_lon']) & (lons <= bbox['max_lon']) & (lats >= bbox['min_lat']) & (lats <= bbox['max_lat'])
                    selected_lats.extend(lats[mask])
                    selected_lons.extend(lons[mask])
                
        if selected_lats:
            visible_bands = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']
            glm_color = '#ffff00' if band in visible_bands else '#ffffff'
            ax.plot(selected_lons, selected_lats, 'o', color=glm_color, markersize=4.5, transform=ccrs_plate, zorder=10)
    except Exception:
        pass

def process_scene(fs, s3_file, bbox, date_obj, hour_utc, minute_utc, sat, show_glm, interpolation, band, output_filename):
    local_nc = "temp_goes_data.nc"
    if os.path.exists(local_nc): os.remove(local_nc)
    fs.get(s3_file, local_nc)
    
    with xr.open_dataset(local_nc, engine='h5netcdf') as ds:
        proj_info = ds.goes_imager_projection
        h, lon_0, sweep = float(proj_info.perspective_point_height), float(proj_info.longitude_of_projection_origin), str(proj_info.sweep_angle_axis)
        p, pc = ccrs.Geostationary(central_longitude=lon_0, satellite_height=h, sweep_axis=sweep), ccrs.PlateCarree()
        
        x1_m, y1_m = p.transform_point(bbox['min_lon'], bbox['min_lat'], pc)
        x2_m, y2_m = p.transform_point(bbox['max_lon'], bbox['max_lat'], pc)
        x_min_m, x_max_m = sorted([x1_m, x2_m])
        y_min_m, y_max_m = sorted([y1_m, y2_m])
        
        y_slice = slice(y_max_m / h, y_min_m / h) if ds.y[0] > ds.y[-1] else slice(y_min_m / h, y_max_m / h)
        x_slice = slice(x_min_m / h, x_max_m / h) if ds.x[0] < ds.x[-1] else slice(x_max_m / h, x_min_m / h)
            
        cmi_data = np.nan_to_num(ds['CMI'].sel(x=x_slice, y=y_slice).values, nan=0.0)
        
        fig = plt.figure(figsize=(10, 10), dpi=160, facecolor='#05070a')
        ax = plt.axes(projection=p, facecolor='#05070a')
        ax.set_extent([x_min_m, x_max_m, y_min_m, y_max_m], crs=p)
        
        if band == 'C04':
            cmap, norm = create_cirrus_colormap_soft(), mcolors.PowerNorm(gamma=0.75, vmin=0.001, vmax=0.32)
        elif band == 'C13':
            cmap, norm = create_exact_custom_ir(), mcolors.Normalize(vmin=183.15, vmax=313.15)
        elif band in ['C08', 'C09', 'C10']:
            cmap, norm = 'YlGnBu_r', mcolors.Normalize(vmin=195.0, vmax=265.0)
        elif band == 'C07':
            cmap, norm = 'hot', mcolors.Normalize(vmin=210.0, vmax=340.0)
        elif band in ['C01', 'C02', 'C03', 'C05', 'C06']:
            cmap, norm = 'gray', mcolors.Normalize(vmin=0.0, vmax=0.85)
        else:
            cmap, norm = 'gray_r', mcolors.Normalize(vmin=190.0, vmax=300.0)
            
        ax.imshow(cmi_data, origin='upper', extent=[x_min_m, x_max_m, y_min_m, y_max_m], transform=p, cmap=cmap, norm=norm, interpolation=interpolation)
        ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.9, zorder=5)
        ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.6, linestyle=':', zorder=5)
        
        if show_glm:
            overlay_glm_lightning(fs, sat, date_obj, hour_utc, minute_utc, bbox, ax, pc, band=band)
            
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0, facecolor='#05070a')
        plt.close(fig)
        
    if os.path.exists(local_nc): os.remove(local_nc)
    return output_filename

# ==========================================
# INTERFACE SIDEBAR (CONTROLES)
# ==========================================

st.sidebar.header("⚙️ Configurações")
sat = st.sidebar.selectbox("Satélite", [('GOES-16 (East)', 'noaa-goes16'), ('GOES-19 (East)', 'noaa-goes19')], format_func=lambda x: x[0])[1]
band = st.sidebar.selectbox("Produto / Banda", [
    ('Banda 01 (Visível Azul)', 'C01'), ('Banda 02 (Visível Vermelho)', 'C02'),
    ('Banda 03 (Infravermelho Próximo)', 'C03'), ('Banda 04 (Cirrus)', 'C04'),
    ('Banda 05 (Neve / Gelo)', 'C05'), ('Banda 06 (Tamanho Partícula)', 'C06'),
    ('Banda 07 (IR Curto / Incêndios)', 'C07'), ('Banda 08 (Vapor de Água Alta)', 'C08'),
    ('Banda 09 (Vapor de Água Média)', 'C09'), ('Banda 10 (Vapor de Água Baixa)', 'C10'),
    ('Banda 11 (Fase Topo Nuvem)', 'C11'), ('Banda 12 (Ozônio)', 'C12'),
    ('Banda 13 (IR WeatherNerds)', 'C13'), ('Banda 14 (IR Janela Longa)', 'C14'),
    ('Banda 15 (IR Sujo)', 'C15'), ('Banda 16 (CO2)', 'C16')
], index=12, format_func=lambda x: x[0])[1]

date_val = st.sidebar.date_input("Data", datetime.date.today() - datetime.timedelta(days=2))
hour_val = st.sidebar.slider("Hora (UTC)", 0, 23, 18)
min_val = st.sidebar.selectbox("Minuto (UTC)", [0, 10, 20, 30, 40, 50], index=0)
out_fmt = st.sidebar.radio("Formato de Saída", ['PNG (Estático)', 'GIF (Animação)'])
style = st.sidebar.selectbox("Estilo", [('Pixelado (Bruto)', 'nearest'), ('Suavizado', 'bilinear')], format_func=lambda x: x[0])[1]
show_glm = st.sidebar.checkbox("⚡ Descargas Elétricas (GLM)", value=False)

if out_fmt == 'GIF (Animação)':
    st.sidebar.subheader("🎬 Opções de GIF")
    dur_hours = st.sidebar.selectbox("Duração", [1, 2, 3, 6], index=2)
    step_mins = st.sidebar.selectbox("Intervalo entre quadros", [10, 15, 20, 30], index=1)
    fps_val = st.sidebar.selectbox("Velocidade (FPS)", [1, 2, 4], index=1)

st.sidebar.subheader("📍 Coordenadas (Bounding Box)")
c1, c2 = st.sidebar.columns(2)
min_lon = c1.number_input("Lon Min", value=-60.0, step=1.0)
max_lon = c2.number_input("Lon Max", value=-40.0, step=1.0)
min_lat = c1.number_input("Lat Min", value=-25.0, step=1.0)
max_lat = c2.number_input("Lat Max", value=-5.0, step=1.0)

bbox = {"min_lon": min_lon, "max_lon": max_lon, "min_lat": min_lat, "max_lat": max_lat}

# ==========================================
# EXECUTAR PROCESSAMENTO
# ==========================================

if st.button("🚀 Gerar Imagem de Satélite", type="primary"):
    try:
        if 'PNG' in out_fmt:
            with st.spinner("🔍 Conectando à NOAA S3 e processando imagem..."):
                fs, s3_file = fetch_goes_data(sat, band, date_val, hour_val, min_val)
                out_file = process_scene(fs, s3_file, bbox, date_val, hour_val, min_val, sat, show_glm, style, band, 'goes_result.png')
                st.image(out_file, use_container_width=True)
                with open(out_file, "rb") as f:
                    st.download_button("📥 Download PNG", f, file_name="goes_result.png", mime="image/png")
        else:
            frames = []
            start_dt = datetime.datetime.combine(date_val, datetime.time(hour_val, min_val))
            time_steps = list(range(0, dur_hours * 60 + 1, step_mins))
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, offset in enumerate(time_steps):
                frame_dt = start_dt + datetime.timedelta(minutes=offset)
                f_date, f_hour, f_min = frame_dt.date(), frame_dt.hour, frame_dt.minute
                
                status_text.text(f"🎬 Processando quadro {idx+1}/{len(time_steps)} ({f_hour:02d}:{f_min:02d} UTC)...")
                progress_bar.progress((idx + 1) / len(time_steps))
                
                try:
                    fs, frame_s3 = fetch_goes_data(sat, band, f_date, f_hour, f_min)
                    f_name = f"temp_{f_hour}_{f_min}.png"
                    process_scene(fs, frame_s3, bbox, f_date, f_hour, f_min, sat, show_glm, style, band, f_name)
                    frames.append(imageio.v2.imread(f_name))
                    if os.path.exists(f_name): os.remove(f_name)
                except Exception as err:
                    st.warning(f"Aviso no quadro {f_hour}:{f_min} - {err}")
            
            if frames:
                gif_file = "goes_animation.gif"
                imageio.mimsave(gif_file, frames, fps=fps_val)
                st.image(gif_file, use_container_width=True)
                with open(gif_file, "rb") as f:
                    st.download_button("📥 Download GIF", f, file_name="goes_animation.gif", mime="image/gif")
    except Exception as e:
        st.error(f"❌ Erro ao processar: {e}")
