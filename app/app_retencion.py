import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from openai import OpenAI
import requests
import joblib 
import pandas as pd 
import numpy as np
import io 
import os

# 1. Configuración de la página (Debe ser el primer comando de Streamlit)
st.set_page_config(page_title="Xaubet Intelligence", page_icon="🏋️‍♂️", layout="wide")

# Inicialización de estados de la sesión para el control de la automatización
if 'generando' not in st.session_state:
    st.session_state.generando = False
if 'df_resultado_rescate' not in st.session_state:
    st.session_state.df_resultado_rescate = None

# NUEVO: Estados para el generador individual (Pestaña 1)
if 'generando_indiv' not in st.session_state:
    st.session_state.generando_indiv = False
if 'email_individual' not in st.session_state:
    st.session_state.email_individual = None

@st.cache_resource 
def cargar_modelo():
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    
    ruta = os.path.join(directorio_actual, 'modelo_xgboost.pkl')
    
    return joblib.load(ruta)

modelo = cargar_modelo()

def comprobar_ollama():
    try:
        respuesta = requests.get("http://localhost:11434/", timeout=1)
        return respuesta.status_code == 200
    except:
        return False

servidor_online = comprobar_ollama()
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# 2. Título y estado del servidor
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🏋️‍♂️ Xaubet Intelligence")
    st.markdown("Plataforma predictiva y generativa para la retención de socios.")
with col2:
    if servidor_online:
        st.success("🟢 Servidor IA: Conectado")
    else:
        st.error("🔴 Servidor IA: Apagado")
        st.caption("Abre Ollama para habilitar la generación de correos.")

st.divider()

# 3. Barra Lateral (Inputs y Configuración)
st.sidebar.header("⚙️ Datos del Socio")

edad = st.sidebar.slider("Edad", 16, 90, 21)
dias_sin_venir = st.sidebar.slider("Días sin venir", 0, 365, 45)
zona_proximidad = st.sidebar.selectbox("Ubicación", options=[0, 1], format_func=lambda x: "0 - Cerca" if x == 0 else "1 - Lejos")
total_visitas = st.sidebar.number_input("Total de visitas", value=75, max_value=730)

lista_cuotas = [
    'AB ADULTS', 'AB PETIT PINEDA', 'AB. ATURAT', 'AB. BABY 0-5', 'AB. CAP DE SET. JOVE', 
    'AB. CAP DE SET.FAM', 'AB. DIVERSITAT FUNCIONAL', 'AB. FAMILIAR', 
    'AB. INFANTIL 9-13', 'AB. JOVE 14-17', 'AB. PENSIONISTA', 
    'AB. PETIT MINI  6-8', 'AB.CAP DE SET. 9-13 ANYS', 'AB.CAP DE SET. BABY 0-5 ANYS', 
    'AB.CAP DE SET. MINI  6-8 ANYS', 'AB.CAP DE SET. PENSIONISTA', 
    'AB.CAP DE SET.ADULT', 'AB.CAP DE SET.DIVERISTAT FUNCI', 'ABONAMENT ADULT BONOS', 
    'ACOMP. DISCAPACITAT DEPENENT', 'ADULT MIG DIA 12 A 15', 
    'ADULT MIGDIA CAP SETMANA', 'CAP DE SETMANA PETIT PINEDA', 'COLECTIVOS', 
    'COL·LECTIU ESPORTIU NATACIÓ', 'COL·LECTIU PINEDA', 'COL·LECTIUS 2.0', 
    'FAMILIA MONOPARENTAL', 'FAMILIA NOMBROSA', 'FIDELITZACIÓ 10-14 ANYS ADULT', 
    'FIDELITZACIÓ 15 ANYS ADULT', 'JOVE BONIFICADA JOVE FIT', 'QUOTA BONIFICADA 60%', 
    'QUOTA MANTENIMENT'
]
tipo_socio_txt = st.sidebar.selectbox("Tipo de Cuota", lista_cuotas)

st.sidebar.divider()
st.sidebar.header("📧 Configuración de la IA")
tono_ia = st.sidebar.selectbox("Tono del mensaje", ["Empático y motivador", "Formal y directo", "Urgente e incentivador"])
regalo_ia = st.sidebar.selectbox("Incentivo a ofrecer", ["Sesión gratis con entrenador", "15% descuento próximo mes"])

# Lista exacta de las columnas esperadas
columnas_esperadas = [
    'edad', 'zona_proximidad', 'total_visitas', 'dias_desde_ultima_visita', 
    'dia_favorito', 'sexo_Mujer', 'sexo_No binario', 'tipo_socio_AB PETIT PINEDA', 
    'tipo_socio_AB. ATURAT', 'tipo_socio_AB. BABY 0-5', 'tipo_socio_AB. CAP DE SET. JOVE', 
    'tipo_socio_AB. CAP DE SET.FAM', 'tipo_socio_AB. DIVERSITAT FUNCIONAL', 'tipo_socio_AB. FAMILIAR', 
    'tipo_socio_AB. INFANTIL 9-13', 'tipo_socio_AB. JOVE 14-17', 'tipo_socio_AB. PENSIONISTA', 
    'tipo_socio_AB. PETIT MINI  6-8', 'tipo_socio_AB.CAP DE SET. 9-13 ANYS', 
    'tipo_socio_AB.CAP DE SET. BABY 0-5 ANYS', 'tipo_socio_AB.CAP DE SET. MINI  6-8 ANYS', 
    'tipo_socio_AB.CAP DE SET. PENSIONISTA', 'tipo_socio_AB.CAP DE SET.ADULT', 
    'tipo_socio_AB.CAP DE SET.DIVERISTAT FUNCI', 'tipo_socio_ABONAMENT ADULT BONOS', 
    'tipo_socio_ACOMP. DISCAPACITAT DEPENENT', 'tipo_socio_ADULT MIG DIA 12 A 15', 
    'tipo_socio_ADULT MIGDIA CAP SETMANA', 'tipo_socio_CAP DE SETMANA PETIT PINEDA', 
    'tipo_socio_COLECTIVOS', 'tipo_socio_COL·LECTIU ESPORTIU NATACIÓ', 'tipo_socio_COL·LECTIU PINEDA', 
    'tipo_socio_COL·LECTIUS 2.0', 'tipo_socio_FAMILIA MONOPARENTAL', 'tipo_socio_FAMILIA NOMBROSA', 
    'tipo_socio_FIDELITZACIÓ 10-14 ANYS ADULT', 'tipo_socio_FIDELITZACIÓ 15 ANYS ADULT', 
    'tipo_socio_JOVE BONIFICADA JOVE FIT', 'tipo_socio_QUOTA BONIFICADA 60%', 'tipo_socio_QUOTA MANTENIMENT'
]

# Preparación de datos y predicción
datos_dict = {col: 0 for col in columnas_esperadas}
datos_dict['edad'] = edad
datos_dict['dias_desde_ultima_visita'] = dias_sin_venir
datos_dict['zona_proximidad'] = zona_proximidad
datos_dict['total_visitas'] = total_visitas
datos_dict['dia_favorito'] = 2 

nombre_columna_cuota = f"tipo_socio_{tipo_socio_txt}"
if nombre_columna_cuota in datos_dict:
    datos_dict[nombre_columna_cuota] = 1

datos_entrada = pd.DataFrame([datos_dict], columns=columnas_esperadas)
probabilidad_real = modelo.predict_proba(datos_entrada)[0][1] * 100

# --- CREACIÓN DE LOS DIFERENTES MENUS ---
tab1, tab2, tab3 = st.tabs(["🎯 Simulador Individual", "📂 Carga Masiva", "✉️ Automatización de Rescate"])

# ==========================================
# PESTAÑA 1: SIMULADOR INDIVIDUAL (MODIFICADA 🛠️)
# ==========================================
with tab1:
    if probabilidad_real >= 65:
        st.error(f"🚨 ESTADO: RIESGO CRÍTICO ({probabilidad_real:.1f}%). Intervención inmediata requerida.")
    elif probabilidad_real >= 30:
        st.warning(f"⚠️ ESTADO: RIESGO MODERADO ({probabilidad_real:.1f}%). Monitorear evolución.")
    else:
        st.success(f"✅ ESTADO: SOCIO SEGURO ({probabilidad_real:.1f}%). Fidelidad estable.")
        
    st.write("") 

    col_grafico, col_texto = st.columns([1, 1])

    with col_grafico:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probabilidad_real,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Riesgo de Baja Real (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps' : [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 65], 'color': "gold"},
                    {'range': [65, 100], 'color': "salmon"}],
                'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 65}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col_texto:
        st.markdown("### 📝 Plan de Acción:")
        if probabilidad_real >= 65:
            st.error("Se recomienda lanzar una campaña de recuperación inmediata.")
            
            # Botones de control individual
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                btn_gen_indiv = st.button(
                    "✨ Generar Email de Rescate", 
                    type="primary", 
                    disabled=not servidor_online or st.session_state.generando_indiv,
                    use_container_width=True
                )
            with col_b2:
                btn_cancel_indiv = st.button(
                    "🛑 Cancelar", 
                    type="secondary", 
                    disabled=not st.session_state.generando_indiv,
                    use_container_width=True
                )

            # Lógica de los botones
            if btn_gen_indiv:
                st.session_state.generando_indiv = True
                st.session_state.email_individual = None
                st.rerun()

            if btn_cancel_indiv:
                st.session_state.generando_indiv = False
                st.warning("⚠️ Generación cancelada por el usuario.")
                st.rerun()

            # Bucle de generación
            if st.session_state.generando_indiv:
                with st.spinner(f"Redactando email con tóno {tono_ia} y ofreciendo {regalo_ia}..."):
                    if total_visitas < 10:
                        r_visitas = "Anímale a retomar el impulso inicial. Recuérdale que los comienzos cuestan, pero lo importante es volver a dar el primer paso."
                    elif total_visitas <= 50:
                        r_visitas = "Menciónale que tenía un ritmo estupendo y anímale a recuperar esa buena rutina de entrenamiento que ya había conseguido."
                    else:
                        r_visitas = "Hazle saber que valoramos muchísimo su larga y constante trayectoria con nosotros. Anímale a no perder todo ese progreso acumulado."

                    if zona_proximidad == 0:
                        r_distancia = "Destaca la enorme comodidad de tener el gimnasio a un paso de casa para que no le dé pereza volver."
                    else:
                        r_distancia = "Empatiza con su situación. Sabemos que venir desde más lejos requiere un esfuerzo extra, pero anímale diciéndole que merece la pena."

                    prompt_maestro = f"""
                    Actúa como el Director de Fidelización del centro deportivo Can Xaubet.
                    Escribe un email persuasivo, cálido y humano para recuperar a un socio de {edad} años.
                    
                    CONTEXTO Y OBJETIVO:
                    - {r_visitas}
                    - {r_distancia}
                    - Queremos que sienta que le echamos de menos, sin sonar desesperados ni robóticos.
                    
                    REGLAS ESTRICTAS DE REDACCIÓN:
                    1. SALUDO: Empieza SIEMPRE el email con "Estimado/a socio/a," (no inventes nombres).
                    2. TONO: {tono_ia}. Redacta en español de España fluido, natural y con energía.
                    3. INCENTIVO: Ofrécele este regalo de reencuentro: {regalo_ia}.
                    4. PROHIBIDO: Nunca menciones la cantidad de días exactos que lleva sin venir ni sus visitas exactas.
                    5. CIERRE OBLIGATORIO: Tienes que despedirte siempre firmando como "El equipo de Can Xaubet" o "Director de Fidelización, Can Xaubet".
                    6. LONGITUD: Breve y directo, máximo 120 palabras.
                    """
                    try:
                        respuesta = client.chat.completions.create(
                            model="gemma2:9b",
                            messages=[
                                {"role": "system", "content": "Eres un experto en redacción corporativa persuasiva, empatía y retención de clientes."}, 
                                {"role": "user", "content": prompt_maestro}
                            ],
                            temperature=0.5
                        )
                        st.session_state.email_individual = respuesta.choices[0].message.content
                    except Exception as e:
                        st.session_state.email_individual = f"Error: {e}"
                
                # Una vez generado, cambiamos el estado y recargamos
                st.session_state.generando_indiv = False
                st.rerun()

            # Mostrar el correo guardado en el estado de la sesión
            if st.session_state.email_individual:
                if "Error:" in st.session_state.email_individual:
                    st.error(st.session_state.email_individual)
                else:
                    st.info(st.session_state.email_individual)

        elif probabilidad_real >= 30:
            st.warning("Trato preferencial recomendado en recepción.")
        else:
            st.success("Mantener comunicación habitual.")

# ==========================================
# PESTAÑA 2: CARGA MASIVA
# ==========================================
with tab2:
    st.header("📂 Análisis por Lotes")
    archivo = st.file_uploader("Sube tu archivo CSV de socios para hacer un análisis por lotes", type="csv")
    
    if archivo:
        df_masivo = pd.read_csv(archivo)
        
        st.markdown("**Vista previa: 5 primeros usuarios del archivo cargado**")
        st.dataframe(df_masivo.head(), use_container_width=True)
        st.write("") 
        
        if st.button("🚀 Procesar toda la base de datos", type="primary"):
            with st.spinner("Analizando..."):
                lista_p = []
                for _, fila in df_masivo.iterrows():
                    d = {col: 0 for col in columnas_esperadas}
                    d['edad'] = fila.get('edad', 35)
                    d['dias_desde_ultima_visita'] = fila.get('dias_sin_venir', 0)
                    d['zona_proximidad'] = fila.get('zona_proximidad', 0)
                    d['total_visitas'] = fila.get('total_visitas', 0)
                    d['dia_favorito'] = 2
                    
                    if 'tipo_socio' in fila:
                        nc = f"tipo_socio_{fila['tipo_socio']}"
                        if nc in d: d[nc] = 1
                        
                    row = pd.DataFrame([d], columns=columnas_esperadas)
                    lista_p.append(modelo.predict_proba(row)[0][1] * 100)
                
                df_masivo['Probabilidad_Baja (%)'] = lista_p
                df_masivo['Estado'] = np.select([(df_masivo['Probabilidad_Baja (%)'] >= 65), (df_masivo['Probabilidad_Baja (%)'] >= 30)], ["Riesgo Crítico", "Riesgo Moderado"], default="Seguro")
                df_final = df_masivo.sort_values(by='Probabilidad_Baja (%)', ascending=False)
                st.dataframe(df_final, use_container_width=True)
                
                st.divider()
                df_criticos = df_final[df_final['Estado'] == "Riesgo Crítico"]
                if not df_criticos.empty:
                    csv = df_criticos.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Descargar Lista de Rescate (CSV)", data=csv, file_name='socios_criticos.csv', mime='text/csv')

                st.write("")
                resumen = df_final['Estado'].value_counts().reset_index()
                fig_pie = px.pie(resumen, values='count', names='Estado', title="Distribución de Riesgos Real",
                                 color='Estado', color_discrete_map={'Riesgo Crítico':'salmon', 'Riesgo Moderado':'gold', 'Seguro':'lightgreen'})
                st.plotly_chart(fig_pie)

# ==========================================
# PESTAÑA 3: AUTOMATIZACIÓN 
# ==========================================
with tab3:
    st.header("✉️ Generador Masivo de Emails")
    st.markdown("Genera de forma masiva y totalmente personalizada correos electrónicos para todos los socios en riesgo crítico.")
    archivo_rescate = st.file_uploader("Sube el archivo de Socios Críticos", type=["xlsx", "csv"])

    if archivo_rescate:
        df_rescate = pd.read_excel(archivo_rescate) if archivo_rescate.name.endswith('.xlsx') else pd.read_csv(archivo_rescate)
        st.write(f"Cargados {len(df_rescate)} socios.")
        
        # Estructura de botones en columnas utilizando el estado dinámico
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            # Deshabilitado si ya está generando o si el servidor está caído
            btn_generar = st.button(
                "✨ Iniciar Generación Automática", 
                type="primary", 
                disabled=st.session_state.generando or not servidor_online,
                use_container_width=True
            )
            
        with col_btn2:
            # Únicamente habilitado si la generación está en progreso
            btn_cancelar = st.button(
                "🛑 Cancelar Generación", 
                type="secondary", 
                disabled=not st.session_state.generando,
                use_container_width=True
            )

        # Acción al pulsar "Iniciar"
        if btn_generar:
            st.session_state.generando = True
            st.session_state.df_resultado_rescate = None  
            st.rerun()

        # Acción al pulsar "Cancelar"
        if btn_cancelar:
            st.session_state.generando = False
            st.warning("⚠️ Generación cancelada por el usuario.")
            st.rerun()

        # Bucle de ejecución controlado por el Estado de la Sesión
        if st.session_state.generando:
            progreso = st.progress(0)
            status_text = st.empty()
            emails_generados = []
            
            for i, socio in df_rescate.iterrows():
                status_text.text(f"Analizando trayectoria y generando correo para el socio {i+1} de {len(df_rescate)}...")
                
                # 1. REGLA DE VISITAS
                v = socio['total_visitas'] if 'total_visitas' in df_rescate.columns else 20
                if v < 10:
                    regla_v = "Anímale a retomar el impulso inicial. Recuérdale que los comienzos cuestan, pero lo importante es volver a dar el primer paso."
                elif v <= 50:
                    regla_v = "Menciónale que tenía un ritmo estupendo y anímale a recuperar esa buena rutina de entrenamiento que ya había conseguido."
                else:
                    regla_v = "Hazle saber que valoramos muchísimo su larga y constante trayectoria con nosotros. Anímale a no perder todo ese progreso acumulado."

                # 2. REGLA DE DISTANCIA
                zona_val = socio['zona_proximidad'] if 'zona_proximidad' in df_rescate.columns else 0
                if zona_val == 0:
                    regla_distancia = "Destaca la enorme comodidad de tener el gimnasio a un paso de casa para que no le dé pereza volver."
                else:
                    regla_distancia = "Empatiza con su situación. Sabemos que venir desde más lejos requiere un esfuerzo extra, pero anímale diciéndole que merece la pena."

                # DATOS DEL SOCIO 
                edad_val = socio['edad'] if 'edad' in df_rescate.columns else 35

                prompt_bulk = f"""
                Actúa como el Director de Fidelización del centro deportivo Can Xaubet.
                Escribe un email persuasivo, cálido y humano para recuperar a un socio de {edad_val} años.
                
                CONTEXTO Y OBJETIVO:
                - {regla_v}
                - {regla_distancia}
                - Queremos que sienta que le echamos de menos, sin sonar desesperados ni robóticos.
                
                REGLAS ESTRICTAS DE REDACCIÓN:
                1. SALUDO: Empieza SIEMPRE el email con "Estimado/a socio/a," (no inventes nombres).
                2. TONO: {tono_ia}. Redacta en español de España fluido, natural y con energía.
                3. INCENTIVO: Ofrécele este regalo de reencuentro: {regalo_ia}.
                4. PROHIBIDO: Nunca menciones la cantidad de días exactos que lleva sin venir ni sus visitas exactas.
                5. CIERRE OBLIGATORIO: Tienes que despedirte siempre firmando como "El equipo de Can Xaubet" o "Director de Fidelización, Can Xaubet".
                6. LONGITUD: Breve y directo, máximo 120 palabras.
                """
                
                try:
                    res = client.chat.completions.create(
                        model="gemma2:9b",
                        messages=[
                            {"role": "system", "content": "Eres un experto en redacción corporativa persuasiva, empatía y retención de clientes."}, 
                            {"role": "user", "content": prompt_bulk}
                        ],
                        temperature=0.5
                    )
                    emails_generados.append(res.choices[0].message.content)
                except:
                    emails_generados.append("Error en generación")
                
                progreso.progress((i + 1) / len(df_rescate))
            
            # Al finalizar correctamente la iteración
            df_rescate['Email_Generado'] = emails_generados
            st.session_state.df_resultado_rescate = df_rescate
            st.session_state.generando = False
            st.rerun()

        # Mostrar los resultados persistentes si existen en la sesión
        if st.session_state.df_resultado_rescate is not None:
            st.success("¡Emails generados con éxito!")
            df_final_mostrar = st.session_state.df_resultado_rescate
            columnas_mostrar = ['id_socio', 'Email_Generado'] if 'id_socio' in df_final_mostrar.columns else ['Email_Generado']
            st.dataframe(df_final_mostrar[columnas_mostrar], use_container_width=True)
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_final_mostrar.to_excel(writer, index=False)
            st.download_button(
                "📥 Descargar Reporte Final (Excel)", 
                data=buffer.getvalue(), 
                file_name='reporte_final_emails.xlsx', 
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
