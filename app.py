import os
import datetime
import base64
import s3fs
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import imageio

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA E ESTILO TEMA ESCURO
# ==========================================

st.set_page_config(
    page_title="GOES Satellite Data Processor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS Personalizado (Tema Escuro Profissional)
st.markdown("""
    <style>
    .stApp {
        background-color: #05070a;
        color: #e2e8f0;
    }
    div[data-testid="stSidebar"] {
        background-color: #0b0f17;
        border-right: 1px solid #1e293b;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 600;
    }
    .stButton>button {
        background: #0284c7;
        color: #ffffff;
        border: none;
        border-radius: 4px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        width: 100%;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #0369a1;
        color: #ffffff;
    }
    .status-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. PALETAS DE CORES E FUNÇÕES TÉCNICAS
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
        (-60, (254, 77, 5)),   (-50, (165, 252, 13)),  (-40, (2, 181, 34)),
        (-30, (4, 60, 146)),   (-25, (11, 185, 216)),  (-20, (188, 188, 186)),
        (-10, (162, 162, 162)),(0,   (135, 135, 135)), (10,  (109, 109, 109)),
        (20,  (83, 83, 83)),   (30,  (56, 56, 56)),    (40,  (30, 30, 30))
    ]
    raw_table.sort(key=lambda x: x[0])
    min_t, max_t = -90.0, 40.0
    colors = []
    for t, rgb in raw_table:
        norm_pos = (t - min_t) / (max_t - min_t)
        colors.append((norm_pos, (rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)))
    return mcolors.LinearSegmentedColormap.from_list('exact_custom_ir', colors)

def fetch_goes_data(sat, band, date_obj, hour_utc, minute_utc):
    fs = s3fs.S3FileSystem(anon=True)
    doy = date_obj.strftime('%j')
    year = date_obj.strftime('%Y')
    path = f"{sat}/ABI-L2-CMIPF/{year}/{doy}/{hour_utc:02d}/"
    files = fs.ls(path)
    target_files = [f for f in files if f"M6{band}" in f or f"M3{band}" in f or f"M4{band}" in f]
    
    if not target_files:
        raise FileNotFoundError(f"Dados indisponiveis no servidor NOAA para o caminho: s3://{path}")
    
    best_file = target_files[0]
    min_diff = 999
    for f in target_files:
        try:
            fn = f.split('/')[-1]
            time_str = fn.split('_s')[1][:11]
            f_min = int(time_str[9:11])
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
        
        if not files:
            return 0
            
        selected_lats, selected_lons = [], []
        for f in files:
            try:
                fn = f.split('/')[-1]
                time_str = fn.split('_s')[1][:11]
                f_min = int(time_str[9:11])
                if abs(f_min - minute_utc) > 5:
                    continue
            except Exception:
                pass
                
            with fs.open(f) as s3_file:
                with xr.open_dataset(s3_file, engine='h5netcdf') as ds_glm:
                    if 'group_lat' in ds_glm:
                        lats, lons = ds_glm['group_lat'].values, ds_glm['group_lon'].values
                    elif 'flash_lat' in ds_glm:
                        lats, lons = ds_glm['flash_lat'].values, ds_glm['flash_lon'].values
                    else:
                        continue
                    
                    mask = (lons >= bbox['min_lon']) & (lons <= bbox['max_lon']) & \
                           (lats >= bbox['min_lat']) & (lats <= bbox['max_lat'])
                    selected_lats.extend(lats[mask])
                    selected_lons.extend(lons[mask])
                
        if selected_lats:
            visible_bands = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']
            glm_color = '#ffff00' if band in visible_bands else '#ffffff'
            ax.plot(
                selected_lons, selected_lats, 'o', color=glm_color, 
                markersize=4.5, transform=ccrs_plate, zorder=10,
                markeredgecolor=glm_color, markeredgewidth=0.0
            )
        return len(selected_lats)
    except Exception:
        return 0

def process_and_render_scene(fs, s3_file, bbox, date_obj, hour_utc, minute_utc, sat, show_glm=False, interpolation='nearest', band='C13', output_filename='output.png'):
    local_nc = "temp_goes_data.nc"
    if os.path.exists(local_nc):
        os.remove(local_nc)
        
    fs.get(s3_file, local_nc)
    
    with xr.open_dataset(local_nc, engine='h5netcdf') as ds:
        proj_info = ds.goes_imager_projection
        h = float(proj_info.perspective_point_height)
        lon_0 = float(proj_info.longitude_of_projection_origin)
        sweep = str(proj_info.sweep_angle_axis)
        
        p = ccrs.Geostationary(central_longitude=lon_0, satellite_height=h, sweep_axis=sweep)
        pc = ccrs.PlateCarree()
        
        x1_m, y1_m = p.transform_point(bbox['min_lon'], bbox['min_lat'], pc)
        x2_m, y2_m = p.transform_point(bbox['max_lon'], bbox['max_lat'], pc)
        
        x_min_m, x_max_m = sorted([x1_m, x2_m])
        y_min_m, y_max_m = sorted([y1_m, y2_m])
        
        x_left_rad, x_right_rad = x_min_m / h, x_max_m / h
        y_bottom_rad, y_top_rad = y_min_m / h, y_max_m / h
        
        y_slice = slice(y_top_rad, y_bottom_rad) if ds.y[0] > ds.y[-1] else slice(y_bottom_rad, y_top_rad)
        x_slice = slice(x_left_rad, x_right_rad) if ds.x[0] < ds.x[-1] else slice(x_right_rad, x_left_rad)
            
        data_slice = ds['CMI'].sel(x=x_slice, y=y_slice)
        if data_slice.size == 0:
            data_slice = ds['CMI']
            
        cmi_data = np.nan_to_num(data_slice.values, nan=0.0)
        
        fig = plt.figure(figsize=(10, 10), dpi=180, facecolor='#05070a')
        ax = plt.axes(projection=p, facecolor='#05070a')
        ax.set_extent([x_min_m, x_max_m, y_min_m, y_max_m], crs=p)
        
        if band == 'C04':
            cmap = create_cirrus_colormap_soft()
            norm = mcolors.PowerNorm(gamma=0.75, vmin=0.001, vmax=0.32)
        elif band == 'C13':
            cmap = create_exact_custom_ir()
            norm = mcolors.Normalize(vmin=183.15, vmax=313.15)
        elif band in ['C08', 'C09', 'C10']:
            cmap = 'YlGnBu_r'
            norm = mcolors.Normalize(vmin=195.0, vmax=265.0)
        elif band == 'C07':
            cmap = 'hot'
            norm = mcolors.Normalize(vmin=210.0, vmax=340.0)
        elif band in ['C01', 'C02', 'C03', 'C05', 'C06']:
            cmap = 'gray'
            norm = mcolors.Normalize(vmin=0.0, vmax=0.85)
        else:
            cmap = 'gray_r'
            norm = mcolors.Normalize(vmin=190.0, vmax=300.0)
            
        ax.imshow(
            cmi_data, origin='upper',
            extent=[x_min_m, x_max_m, y_min_m, y_max_m],
            transform=p, cmap=cmap, norm=norm, interpolation=interpolation
        )
        
        ax.add_feature(cfeature.COASTLINE, edgecolor='#475569', linewidth=0.8, zorder=5)
        ax.add_feature(cfeature.BORDERS, edgecolor='#334155', linewidth=0.5, linestyle=':', zorder=5)
        
        if show_glm:
            overlay_glm_lightning(fs, sat, date_obj, hour_utc, minute_utc, bbox, ax, pc, band=band)
            
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0, facecolor='#05070a')
        plt.close(fig)
        
    if os.path.exists(local_nc):
        os.remove(local_nc)
    return output_filename

# ==========================================
# 3. INTERFACE DE UTILIZADOR
# ==========================================

st.title("SISTEMA DE PROCESSAMENTO DE DADOS SATELLITE GOES")
st.markdown("---")

# Painel Lateral
st.sidebar.header("CONFIGURAÇÃO DE PARÂMETROS")

sat = st.sidebar.selectbox("Satélite", [('GOES-16 (East)', 'noaa-goes16'), ('GOES-19 (East Operacional)', 'noaa-goes19')], format_func=lambda x: x[0])[1]

bands_dict = {
    'Banda 13 (Infravermelho Custom - 10.3 µm)': 'C13',
    'Banda 04 (Cirrus / Visível Aprimorado - 1.37 µm)': 'C04',
    'Banda 02 (Visível Vermelho Alta Res. - 0.64 µm)': 'C02',
    'Banda 07 (Infravermelho Curto - 3.9 µm)': 'C07',
    'Banda 08 (Vapor de Água Alta Troposfera - 6.2 µm)': 'C08',
    'Banda 09 (Vapor de Água Média Troposfera - 6.9 µm)': 'C09',
    'Banda 10 (Vapor de Água Baixa Troposfera - 7.3 µm)': 'C10'
}
band_label = st.sidebar.selectbox("Produto / Banda", list(bands_dict.keys()))
band = bands_dict[band_label]

date_val = st.sidebar.date_input("Data de Observação", datetime.date.today() - datetime.timedelta(days=2))
col_h, col_m = st.sidebar.columns(2)
hour_val = col_h.selectbox("Hora (UTC)", list(range(24)), index=18)
min_val = col_m.selectbox("Minuto", list(range(0, 60, 10)), index=0)

out_fmt = st.sidebar.selectbox("Formato de Saída", ['PNG (Imagem Estática)', 'GIF (Animação Temporal)'])
style = st.sidebar.selectbox("Interpolação Espacial", [('Bruta (Nearest)', 'nearest'), ('Suavizada (Bilinear)', 'bilinear')], format_func=lambda x: x[0])[1]
show_glm = st.sidebar.checkbox("Atividade Elétrica / Descargas GLM (5 min)", value=False)

if 'GIF' in out_fmt:
    st.sidebar.subheader("Parâmetros da Animação")
    gif_duration = st.sidebar.slider("Duração Total (Horas)", 1, 6, 3)
    gif_interval = st.sidebar.select_slider("Intervalo entre Quadros (Min)", options=[10, 15, 20, 30], value=15)
    gif_fps = st.sidebar.slider("Quadros por Segundo (FPS)", 1, 5, 2)

# Área de Seleção Geográfica (Mapa Folium)
st.subheader("Área de Interesse Geográfico (Bounding Box)")
st.caption("Utilize a ferramenta de desenho de retângulo no canto esquerdo do mapa para delimitar a região geográfica.")

# Inicializar coordenadas padrão
min_lon, max_lon, min_lat, max_lat = -60.0, -40.0, -25.0, -5.0

# Mapa Interativo Folium
m = folium.Map(location=[-14.2350, -51.9253], zoom_start=4, tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri")
draw = Draw(export=False, draw_options={'rectangle': True, 'polyline': False, 'polygon': False, 'circle': False, 'marker': False, 'circlemarker': False})
draw.add_to(m)

map_data = st_folium(m, width=1000, height=400)

# Atualizar coordenadas se o utilizador desenhar no mapa
if map_data and map_data.get("last_active_drawing"):
    coords = map_data["last_active_drawing"]["geometry"]["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    min_lon, max_lon = round(min(lons), 2), round(max(lons), 2)
    min_lat, max_lat = round(min(lats), 2), round(max(lats), 2)

col1, col2, col3, col4 = st.columns(4)
c_min_lon = col1.number_input("Longitude Mínima", value=min_lon)
c_max_lon = col2.number_input("Longitude Máxima", value=max_lon)
c_min_lat = col3.number_input("Latitude Mínima", value=min_lat)
c_max_lat = col4.number_input("Latitude Máxima", value=max_lat)

bbox = {"min_lon": c_min_lon, "max_lon": c_max_lon, "min_lat": c_min_lat, "max_lat": c_max_lat}

st.markdown("---")

# Botão de Execução
if st.button("GERAR PROCESSAMENTO"):
    status_box = st.empty()
    status_box.info("A estabelecer ligação aos servidores da NOAA no AWS S3...")
    
    try:
        if 'PNG' in out_fmt:
            fs, s3_file = fetch_goes_data(sat, band, date_val, hour_val, min_val)
            status_box.info("A descarregar ficheiros NetCDF e a processar geometria...")
            
            out_file = process_and_render_scene(
                fs, s3_file, bbox, date_val, hour_val, min_val, sat, 
                show_glm=show_glm, interpolation=style, band=band, output_filename='goes_result.png'
            )
            
            status_box.empty()
            st.subheader("Resultado do Processamento")
            st.image(out_file, use_column_width=True)
            
            with open(out_file, "rb") as file:
                st.download_button(
                    label="BAIXAR IMAGEM (PNG)",
                    data=file,
                    file_name="goes_observacao.png",
                    mime="image/png"
                )
                
        else:
            frames = []
            fs, _ = fetch_goes_data(sat, band, date_val, hour_val, min_val)
            start_dt = datetime.datetime.combine(date_val, datetime.time(hour_val, min_val))
            time_steps = list(range(0, (gif_duration * 60) + 1, gif_interval))
            
            progress_bar = st.progress(0)
            
            for idx, offset in enumerate(time_steps):
                frame_dt = start_dt + datetime.timedelta(minutes=offset)
                f_date = frame_dt.date()
                f_hour = frame_dt.hour
                f_min = frame_dt.minute
                
                status_box.info(f"A processar quadro {idx+1}/{len(time_steps)} - Horário: {f_hour:02d}:{f_min:02d} UTC")
                
                try:
                    _, frame_s3 = fetch_goes_data(sat, band, f_date, f_hour, f_min)
                    frame_name = f"temp_{f_hour}_{f_min}.png"
                    
                    process_and_render_scene(
                        fs, frame_s3, bbox, f_date, f_hour, f_min, sat, 
                        show_glm=show_glm, interpolation=style, band=band, output_filename=frame_name
                    )
                    frames.append(imageio.v2.imread(frame_name))
                    if os.path.exists(frame_name):
                        os.remove(frame_name)
                except Exception as frame_err:
                    st.warning(f"Aviso no quadro {f_hour}:{f_min} UTC: {frame_err}")
                
                progress_bar.progress((idx + 1) / len(time_steps))
                
            if frames:
                status_box.info("A compilar ficheiro GIF...")
                gif_file = 'goes_animation.gif'
                imageio.mimsave(gif_file, frames, fps=gif_fps)
                
                status_box.empty()
                st.subheader("Resultado da Animação")
                st.image(gif_file, use_column_width=True)
                
                with open(gif_file, "rb") as file:
                    st.download_button(
                        label="BAIXAR ANIMAÇÃO (GIF)",
                        data=file,
                        file_name="goes_animacao.gif",
                        mime="image/gif"
                    )

    except Exception as e:
        status_box.empty()
        st.error(f"Erro de Execução: {str(e)}")
