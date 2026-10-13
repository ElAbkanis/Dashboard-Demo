# =========================================================
# IMPORTACIONES DE LIBRERIAS
# =========================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import shutil
import sys
import subprocess
import threading



from plotly.subplots import make_subplots
from scipy.stats.mstats import winsorize
from dash import Dash, dcc, html, dash_table, Input, Output
from scipy.stats import t



from pathlib import Path
from werkzeug.serving import make_server

# =========================================================
# CONFIGURACIÓN
# =========================================================

DIRECTORIO_PROYECTO = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent
)
DIRECTORIO_ASSETS = (
    Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    / "assets"
)
ARCHIVO = DIRECTORIO_PROYECTO / "Datos" / "Septiembre2026.csv"

# Multiplicador que veníamos evaluando.
# 1.96 corresponde a límites de referencia del 95 %.
# Para límites Shewhart clásicos de 3 sigma, cambiar a 3.0.
MULTIPLO_CONTROL = 1.96

PORCENTAJE_WINSORIZACION = 0.15


# =========================================================
# CARGA Y PREPARACIÓN DE DATOS
# =========================================================

data = pd.read_csv(
    ARCHIVO)
data["ResultValue"] = data["ResultValue"].str.replace(",", ".", regex=False)
data["ResultValue"] = pd.to_numeric(data["ResultValue"], errors="coerce")
data.head()

columnas_requeridas = {
    "Fecha", "Punto Muestreo", "Componente", "ResultValue", "Muestreo"
}
faltantes = columnas_requeridas - set(data.columns)

if faltantes:
    raise ValueError(
        f"Faltan columnas requeridas en el archivo: {faltantes}"
    )

data["Fecha"] = pd.to_datetime(
    data["Fecha"],
    dayfirst=True,
    errors="coerce"
)
data["ResultValue"] = pd.to_numeric(
    data["ResultValue"], errors="coerce"
)
data["Componente"] = data["Componente"].astype("string").str.strip()
data["Muestreo"] = data["Muestreo"].astype("string").str.strip().str.upper()
data["Muestreo"] = data["Muestreo"].replace({"COMPOSITO": "COMPOSITE"})

data = data.dropna(
    subset=["Fecha", "Componente", "ResultValue", "Muestreo"]
).copy()

if data.empty:
    raise ValueError("El archivo no contiene datos válidos.")

componentes = sorted(data["Componente"].unique().tolist())
tipos_muestreo = sorted(data["Muestreo"].unique().tolist())

fecha_min = data["Fecha"].min().date()
fecha_max = data["Fecha"].max().date()


# =========================================================
# ESTILO GENERAL
# =========================================================

COLOR_TEXTO = "#1D2939"
COLOR_SECUNDARIO = "#667085"
COLOR_FONDO = "#F4F6F8"
COLOR_BORDE = "#D0D5DD"
COLOR_CENTRAL = "#2878B5"
COLOR_CONTROL = "#D92D20"


# =========================================================
# TABLA DE RESUMEN ESTADÍSTICO
# =========================================================

def crear_resumen(df, componente):

    # Se excluyen los ceros, siguiendo el criterio acordado.
    valores = df.loc[
        df["Componente"].eq(componente)
        & (df["ResultValue"] != 0),
        "ResultValue"
    ].dropna()

    if valores.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No hay resultados distintos de cero para este período.",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False
        )
        fig.update_layout(
            template="plotly_white",
            height=180
        )
        return fig

    n = len(valores)
    media = valores.mean()
    mediana = valores.median()
    desviacion = valores.std(ddof=1) if n > 1 else np.nan

    cv = (
        desviacion / abs(media) * 100
        if n > 1 and media != 0
        else np.nan
    )

    media_winsorizada = winsorize(
        valores.to_numpy(),
        limits=[
            PORCENTAJE_WINSORIZACION,
            PORCENTAJE_WINSORIZACION
        ]
    ).mean()

    def fmt(x, dec=3):
        return "—" if pd.isna(x) else f"{x:,.{dec}f}"

    indicadores = [
        "N válido (sin ceros)",
        "Media",
        "Mediana",
        "Desviación estándar",
        "Mínimo",
        "Máximo",
        "CV (%)",
        "Percentil 25",
        "Percentil 75",
        "Media winsorizada"
    ]

    resultados = [
        f"{n:,}",
        fmt(media),
        fmt(mediana),
        fmt(desviacion),
        fmt(valores.min()),
        fmt(valores.max()),
        fmt(cv, 2),
        fmt(valores.quantile(0.25)),
        fmt(valores.quantile(0.75)),
        fmt(media_winsorizada)
    ]

    colores = [
        "#F0F1F3" if i % 2 == 0 else "#FFFFFF"
        for i in range(len(indicadores))
    ]

    fig = go.Figure([
        go.Table(
            columnwidth=[0.65, 0.35],
            header=dict(
                values=["<b>Indicador</b>", "<b>Valor</b>"],
                fill_color="#BFC3C9",
                align=["left", "right"],
                font=dict(
                    color=COLOR_TEXTO,
                    size=13,
                    family="Arial"
                ),
                line_color=COLOR_BORDE,
                height=34
            ),
            cells=dict(
                values=[indicadores, resultados],
                fill_color=[colores, colores],
                align=["left", "right"],
                font=dict(
                    color=COLOR_TEXTO,
                    size=12,
                    family="Arial"
                ),
                line_color="#E1E5EA",
                height=29
            )
        )
    ])

    fig.update_layout(
        title=dict(
            text=f"<b>Resumen estadístico — {componente}</b>",
            x=0.02,
            font=dict(size=19, color=COLOR_TEXTO)
        ),
        template="plotly_white",
        height=390,
        margin=dict(l=10, r=10, t=55, b=5)
    )

    return fig
# =========================================================
# VIOLIN PLOT DEL RESUMEN ESTADÍSTICO
# =========================================================
def crear_violinplot(df, componente):

    valores = df.loc[
        df["Componente"].eq(componente)
        & (df["ResultValue"] != 0),
        "ResultValue"
    ].dropna()

    fig = go.Figure()

    if valores.empty:
        fig.add_annotation(
            text="Sin resultados válidos",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False
        )
    else:
        fig.add_trace(
            go.Violin(
                y=valores.to_numpy(),
                name=componente,
                points="all",
                jitter=0.3,
                pointpos=0,
                width=0.55,
                fillcolor="rgba(40, 120, 181, 0.20)",
                line=dict(
                    color=COLOR_CENTRAL,
                    width=2
                ),
                marker=dict(
                    color=COLOR_CENTRAL,
                    size=5,
                    opacity=0.65
                ),
                hovertemplate=(
                    "Resultado: %{y:.4f}"
                    "<extra></extra>"
                )
            )
        )

    fig.update_layout(
        title=dict(
            text="<b>Distribución de resultados</b>",
            x=0.03,
            font=dict(size=19, color=COLOR_TEXTO)
        ),
        template="plotly_white",
        height=390,
        margin=dict(l=15, r=15, t=55, b=35),
        showlegend=False,
        xaxis=dict(
            title=None,
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title="Resultado",
            gridcolor="#EAECF0",
            zeroline=False
        ),
        font=dict(
            family="Arial",
            color=COLOR_TEXTO
        )
    )

    return fig


def crear_grafico_puntos_muestreo(df):
    puntos = (
        df["Punto Muestreo"]
        .dropna()
        .astype("string")
        .str.strip()
    )
    puntos = puntos[puntos.ne("")]
    frecuencias = puntos.value_counts().sort_values()

    fig = go.Figure()

    if frecuencias.empty:
        fig.add_annotation(
            text="No hay puntos de muestreo para los filtros seleccionados.",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False
        )
    else:
        fig.add_trace(
            go.Bar(
                x=frecuencias.index.tolist(),
                y=frecuencias.to_numpy(),
                orientation="v",
                marker_color=COLOR_CENTRAL,
                text=frecuencias.to_numpy(),
                textposition="outside",
                hovertemplate=(
                    "Punto de muestreo: %{x}<br>"
                    "Registros: %{y}<extra></extra>"
                )
            )
        )
        fig.update_xaxes(
            categoryorder="array",
            categoryarray=frecuencias.index.tolist(),
            tickangle=-45,
            automargin=True
        )

    fig.update_layout(
        title=dict(
            text="<b>Registros por punto de muestreo</b>",
            x=0.03,
            font=dict(size=19, color=COLOR_TEXTO)
        ),
        template="plotly_white",
        height=460,
        margin=dict(l=15, r=20, t=55, b=110),
        showlegend=False,
        xaxis=dict(
            title="Punto de muestreo",
            categoryorder="array",
            categoryarray=frecuencias.index.tolist(),
            tickangle=-45,
            automargin=True
        ),
        yaxis=dict(
            title="Cantidad de registros",
            gridcolor="#EAECF0",
            rangemode="tozero",
            range=[
                0,
                float(frecuencias.max()) * 1.1 if not frecuencias.empty else 1
            ]
        ),
        font=dict(
            family="Arial",
            color=COLOR_TEXTO
        )
    )

    return fig



# =========================================================
# CARTA DE CONTROL
# =========================================================

def crear_shewhart(df, componente):

    puntos = df.loc[
        df["Componente"].eq(componente)
        & (df["ResultValue"] != 0),
        ["Fecha", "ResultValue"]
    ].dropna().sort_values("Fecha").copy()

    puntos = puntos.reset_index(drop=True)
    puntos["N_Dato"] = np.arange(1, len(puntos) + 1)

    fig = go.Figure()

    if puntos.empty:
        fig.add_annotation(
            text="No hay datos para generar la carta de control.",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False
        )
        fig.update_layout(template="plotly_white", height=480)
        return fig

    valores = puntos["ResultValue"].to_numpy()

    # Estimación robusta del centro y de la dispersión.
    valores_win = winsorize(
        valores,
        limits=[
            PORCENTAJE_WINSORIZACION,
            PORCENTAJE_WINSORIZACION
        ]
    )

    centro = float(np.mean(valores_win))
    desviacion = (
        float(np.std(valores_win, ddof=1))
        if len(valores_win) > 1
        else 0.0
    )

    lsc = centro + MULTIPLO_CONTROL * desviacion
    lic = centro - MULTIPLO_CONTROL * desviacion

    # Observaciones originales: no se eliminan los atípicos.
    colores_puntos = np.where(
        (valores > lsc) | (valores < lic),
        COLOR_CONTROL,
        COLOR_CENTRAL
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.015,
        column_widths=[0.82, 0.18],
    )

    fig.add_trace(
        go.Scatter(
            x=puntos["N_Dato"],
            y=puntos["ResultValue"],
            mode="lines+markers",
            name="Resultados",
            line=dict(color="#98A2B3", width=1),
            marker=dict(
                color=colores_puntos,
                size=7
            ),
            customdata=puntos[["Fecha"]].to_numpy(),
            hovertemplate=(
                "Dato: %{x}<br>"
                "Fecha: %{customdata[0]|%d-%m-%Y}<br>"
                "Resultado: %{y:.4f}"
                "<extra></extra>"
            )
        )
    )

    # Líneas de referencia
    lineas = [
        (centro, "LC", COLOR_CENTRAL, "solid"),
        (lsc, "LSC", COLOR_CONTROL, "dash"),
        (lic, "LIC", COLOR_CONTROL, "dash")
    ]

    for valor, nombre, color, estilo in lineas:
        fig.add_hline(
            y=valor,
            line_color=color,
            line_dash=estilo,
            line_width=2,
            annotation_text=f"{nombre}: {valor:.3f}",
            annotation_position="top right",
            annotation_font=dict(
                color=color,
                size=12
            )
        )

    # Línea de tendencia descriptiva
    if len(puntos) >= 3:
        x = puntos["N_Dato"].to_numpy(dtype=float)
        y = np.asarray(valores, dtype=float)

        n = len(x)
        pendiente, intercepto = np.polyfit(x, y, 1)
        y_estimado = pendiente * x + intercepto

        # Error residual de la regresión
        residuos = y - y_estimado
        s = np.sqrt(np.sum(residuos**2) / (n - 2))

        # Error estándar de la media estimada
        x_media = np.mean(x)
        Sxx = np.sum((x - x_media)**2)

        x_linea = np.linspace(x.min(), x.max(), 200)
        y_linea = pendiente * x_linea + intercepto

        se_media = s * np.sqrt(
            1/n + (x_linea - x_media)**2 / Sxx
        )

        t_critico = t.ppf(0.975, df=n - 2)
        limite_sup = y_linea + t_critico * se_media
        limite_inf = y_linea - t_critico * se_media

        # Límite inferior: invisible
        fig.add_trace(
            go.Scatter(
                x=x_linea,
                y=limite_inf,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            )
        )

        # Límite superior y relleno de incertidumbre
        fig.add_trace(
            go.Scatter(
                x=x_linea,
                y=limite_sup,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(127, 86, 217, 0.18)",
                name="IC 95 % de la tendencia",
                hoverinfo="skip"
            )
        )

        # Recta de regresión
        fig.add_trace(
            go.Scatter(
                x=x_linea,
                y=y_linea,
                mode="lines",
                name="Tendencia lineal",
                line=dict(
                    color="#7F56D9",
                    width=2,
                    dash="dot"
                ),
                hoverinfo="skip"
            )
        )


    # Histograma lateral: usa todos los resultados válidos.
    fig.add_trace(
        go.Histogram(
            y= puntos["ResultValue"],
            nbinsy=50,
            orientation="h",
            marker=dict(
                color="#0072B2",
                line=dict(color="white", width=0.5),
            ),
            name="Distribución",
            hovertemplate=(
                "Resultado: %{y}<br>"
                "Frecuencia: %{x}<extra></extra>"
            ),
        ),
        row=1, col=2,
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Carta de control — {componente}</b>",
            x=0.02,
            font=dict(size=20, color=COLOR_TEXTO)
        ),
        template="plotly_white",
        height=520,
        margin=dict(l=60, r=155, t=75, b=60),
        hovermode="closest",
        legend=dict(
            orientation="h",
            y=1.02,
            x=0
        ),
        xaxis_title="Número de observación (orden cronológico)",
        yaxis_title="Resultado",
    )

    return fig


# =========================================================
# INTERFAZ DASH: UNA SOLA PESTAÑA
# =========================================================

app = Dash(__name__, assets_folder=str(DIRECTORIO_ASSETS))

app.layout = html.Div(
    style={
        "backgroundColor": COLOR_FONDO,
        "minHeight": "100vh",
        "padding": "22px",
        "fontFamily": "Arial, sans-serif",
        "color": COLOR_TEXTO
    },
    children=[

        html.Div(
            style={
                "backgroundColor": "#24364B",
                "padding": "22px",
                "borderRadius": "10px",
                "marginBottom": "20px",
                "color": "white"
            },
            children=[
                html.H1(
                    "Pizarra quimiométrica",
                    style={"margin": "0 0 8px 0"}
                ),
                html.Div(
                    "Control de datos elementales",
                    style={"color": "#D9E2F0"}
                )
            ]
        ),

        dcc.Tabs(
            value="datos-elementales",
            children=[
                dcc.Tab(
                    label="Datos elementales",
                    value="datos-elementales",
                    style={
                        "backgroundColor": "#E8EDF3",
                        "color": COLOR_TEXTO,
                        "padding": "12px 20px",
                        "border": "none",
                        "borderRadius": "10px",
                        "fontWeight": "bold",
                        "marginRight": "8px"
                    },
                    selected_style={
                        "backgroundColor": "#24364B",
                        "color": "white",
                        "padding": "12px 20px",
                        "border": "none",
                        "borderRadius": "10px",
                        "fontWeight": "bold",
                        "marginRight": "8px"
                    },
                    children=[

                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns":
                                    "minmax(0, 2.4fr) minmax(260px, 1fr)",
                                "gap": "20px",
                                "alignItems": "start",
                                "backgroundColor": "white",
                                "padding": "20px",
                                "borderRadius": "10px",
                                "marginTop": "15px",
                                "marginBottom": "20px",
                                "border": f"1px solid {COLOR_BORDE}"
                            },
                            children=[
                                dcc.Graph(
                                    id="grafico-shewhart",
                                    config={"displayModeBar": True},
                                    style={
                                        "height": "520px",
                                        "gridColumn": "1",
                                        "gridRow": "1 / span 3",
                                        "minWidth": "0"
                                    }
                                ),
                                html.H3(
                                    "Parámetros de trabajo",
                                    style={
                                        "gridColumn": "2",
                                        "marginTop": "0"
                                    }
                                ),

                                html.Div(
                                    style={
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "gap": "16px",
                                        "gridColumn": "2"
                                    },
                                    children=[

                                        html.Div([
                                            html.Label(
                                                "Fecha inicial",
                                                style={
                                                    "fontWeight": "bold",
                                                    "display": "block",
                                                    "marginBottom": "6px"
                                                }
                                            ),
                                            dcc.DatePickerSingle(
                                                id="fecha-inicio",
                                                date=fecha_min,
                                                min_date_allowed=fecha_min,
                                                max_date_allowed=fecha_max,
                                                display_format="DD-MM-YYYY",
                                                style={"display": "block"}
                                            )
                                        ]),

                                        html.Div([
                                            html.Label(
                                                "Fecha final",
                                                style={
                                                    "fontWeight": "bold",
                                                    "display": "block",
                                                    "marginBottom": "6px"
                                                }
                                            ),
                                            dcc.DatePickerSingle(
                                                id="fecha-fin",
                                                date=fecha_max,
                                                min_date_allowed=fecha_min,
                                                max_date_allowed=fecha_max,
                                                display_format="DD-MM-YYYY",
                                                style={"display": "block"}
                                            )
                                        ]),

                                        html.Div([
                                            html.Label(
                                                "Elemento químico",
                                                style={
                                                    "fontWeight": "bold",
                                                    "display": "block",
                                                    "marginBottom": "6px"
                                                }
                                            ),
                                            dcc.Dropdown(
                                                id="selector-componente",
                                                options=[
                                                    {"label": c, "value": c}
                                                    for c in componentes
                                                ],
                                                value=(
                                                    "Cu" if "Cu" in componentes
                                                    else componentes[0]
                                                ),
                                                clearable=False
                                            )
                                        ]),

                                        html.Div([
                                            html.Label(
                                                "Muestreo",
                                                style={
                                                    "fontWeight": "bold",
                                                    "display": "block",
                                                    "marginBottom": "6px"
                                                }
                                            ),
                                            dcc.Dropdown(
                                                id="selector-muestreo",
                                                options=[
                                                    {"label": tipo, "value": tipo}
                                                    for tipo in tipos_muestreo
                                                ],
                                                value=(
                                                    "PUNTUAL"
                                                    if "PUNTUAL" in tipos_muestreo
                                                    else tipos_muestreo[0]
                                                ),
                                                clearable=False
                                            )
                                        ])
                                    ]
                                ),

                                html.Div(
                                    id="mensaje-datos",
                                    style={
                                        "marginTop": "15px",
                                        "color": COLOR_SECUNDARIO,
                                        "gridColumn": "2"
                                    }
                                )
                            ]
                        ),
                                                  
                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "1fr 2fr 1fr",
                                    "gap": "15px",
                                    "marginBottom": "20px",
                                    "alignItems": "stretch"
                                },
                                children=[

                                    html.Div(
                                        style={
                                            "backgroundColor": "white",
                                            "padding": "12px",
                                            "borderRadius": "10px",
                                            "border": f"1px solid {COLOR_BORDE}",
                                            "minWidth": "0"
                                        },
                                        children=[
                                            dcc.Graph(
                                                id="resumen-estadistico",
                                                config={"displayModeBar": False},
                                                style={"height": "390px"}
                                            )
                                        ]
                                    ),

                                    html.Div(
                                        style={
                                            "backgroundColor": "white",
                                            "padding": "12px",
                                            "borderRadius": "10px",
                                            "border": f"1px solid {COLOR_BORDE}",
                                            "minWidth": "0"
                                        },
                                        children=[
                                            dcc.Graph(
                                                id="grafico-puntos-muestreo",
                                                config={"displayModeBar": False},
                                                style={"height": "460px"}
                                            )
                                        ]
                                    ),

                                    html.Div(
                                        style={
                                            "backgroundColor": "white",
                                            "padding": "12px",
                                            "borderRadius": "10px",
                                            "border": f"1px solid {COLOR_BORDE}",
                                            "minWidth": "0"
                                        },
                                        children=[
                                            dcc.Graph(
                                                id="violinplot-estadistico",
                                                className="violinplot-bottom-modebar",
                                                config={"displayModeBar": True},
                                                style={"height": "390px"}                                      
                                            )
                                        ]
                                    
                                    ),                                    
                                ]
                            ),

                            html.Div(
                                style={
                                    "backgroundColor": "white",
                                    "padding": "18px",
                                    "borderRadius": "10px",
                                    "border": f"1px solid {COLOR_BORDE}",
                                    "marginBottom": "20px"
                                },
                                children=[
                                    html.H3(
                                        "Datos completos del elemento seleccionado",
                                        style={"marginTop": "0"}
                                    ),
                                    dash_table.DataTable(
                                        id="tabla-datos-completos",
                                        columns=[
                                            {"name": columna, "id": columna}
                                            for columna in data.columns
                                        ],
                                        page_action="native",
                                        page_size=20,
                                        sort_action="native",
                                        style_table={
                                            "overflowX": "auto",
                                            "maxHeight": "600px",
                                            "overflowY": "auto"
                                        },
                                        style_header={
                                            "backgroundColor": "#24364B",
                                            "color": "white",
                                            "fontWeight": "bold",
                                            "textAlign": "left",
                                            "whiteSpace": "normal",
                                            "height": "auto"
                                        },
                                        style_cell={
                                            "padding": "10px",
                                            "textAlign": "left",
                                            "minWidth": "110px",
                                            "maxWidth": "280px",
                                            "whiteSpace": "normal",
                                            "height": "auto"
                                        },
                                        style_data_conditional=[
                                            {
                                                "if": {"row_index": "odd"},
                                                "backgroundColor": "#F4F6F8"
                                            }
                                        ]
                                    )
                                ]
                            ),

                    ]
                )
            ]
        )
    ]
)


# =========================================================
# ACTUALIZACIÓN INTERACTIVA
# =========================================================
@app.callback(
    Output("resumen-estadistico", "figure"),
    Output("grafico-puntos-muestreo", "figure"),
    Output("violinplot-estadistico", "figure"),
    Output("grafico-shewhart", "figure"),
    Output("mensaje-datos", "children"),
    Output("tabla-datos-completos", "data"),
    Input("fecha-inicio", "date"),
    Input("fecha-fin", "date"),
    Input("selector-componente", "value"),
    Input("selector-muestreo", "value")
)
def actualizar_pizarra(inicio, fin, componente, muestreo):

    registros_tabla = []
    if inicio and fin and componente and muestreo:
        fecha_tabla_inicio = pd.Timestamp(inicio)
        fecha_tabla_fin_exclusiva = (
            pd.Timestamp(fin) + pd.Timedelta(days=1)
        )
        if fecha_tabla_inicio < fecha_tabla_fin_exclusiva:
            registros_tabla = data.loc[
                data["Componente"].eq(componente)
                & data["Muestreo"].eq(muestreo)
                & data["ResultValue"].ne(0)
                & data["Fecha"].ge(fecha_tabla_inicio)
                & data["Fecha"].lt(fecha_tabla_fin_exclusiva)
            ].to_dict("records")

    if not inicio or not fin or not componente or not muestreo:
        vacio = go.Figure()
        vacio_puntos = crear_grafico_puntos_muestreo(data.iloc[0:0])
        return (
            vacio, vacio_puntos, vacio, vacio,
            "Selecciona las fechas, el componente y el tipo de muestreo.",
            registros_tabla
        )

    fecha_inicio = pd.Timestamp(inicio)
    fecha_fin_exclusiva = (
        pd.Timestamp(fin) + pd.Timedelta(days=1)
    )

    if fecha_inicio >= fecha_fin_exclusiva:
        vacio = go.Figure()
        vacio.add_annotation(
            text="La fecha inicial debe ser anterior a la fecha final.",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False
        )
        vacio.update_layout(
            template="plotly_white",
            height=180
        )
        vacio_puntos = crear_grafico_puntos_muestreo(data.iloc[0:0])
        return (
            vacio, vacio_puntos, vacio, vacio,
            "Revisa el período seleccionado.",
            registros_tabla
        )

    df = data.loc[
        (data["Fecha"] >= fecha_inicio)
        & (data["Fecha"] < fecha_fin_exclusiva)
        & data["Muestreo"].eq(muestreo)
    ].copy()

    if df.empty:
        vacio = go.Figure()
        vacio.add_annotation(
            text="No hay registros en el período seleccionado.",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False
        )
        vacio.update_layout(
            template="plotly_white",
            height=180
        )
        vacio_puntos = crear_grafico_puntos_muestreo(data.iloc[0:0])
        return (
            vacio, vacio_puntos, vacio, vacio,
            "Sin registros para las fechas y el tipo de muestreo seleccionados.",
            registros_tabla
        )

    df_componente = df.loc[
        df["Componente"].eq(componente)
    ]

    n_total = len(df_componente)
    n_ceros = int(
        (df_componente["ResultValue"] == 0).sum()
    )
    n_validos = int(
        (
            df_componente["ResultValue"].notna()
            & (df_componente["ResultValue"] != 0)
        ).sum()
    )

    fig_resumen = crear_resumen(df, componente)
    datos_componente_validos = df_componente.loc[
        df_componente["ResultValue"].ne(0)
    ]
    fig_puntos_muestreo = crear_grafico_puntos_muestreo(
        datos_componente_validos
    )
    fig_violinplot = crear_violinplot(df, componente)
    fig_shewhart = crear_shewhart(df, componente)

    mensaje = (
        f"Registros del componente: {n_total:,} | "
        f"Resultados distintos de cero: {n_validos:,} | "
        f"Ceros excluidos del análisis: {n_ceros:,} | "
        f"Muestreo: {muestreo}"
    )

    return (
        fig_resumen,
        fig_puntos_muestreo,
        fig_violinplot,
        fig_shewhart,
        mensaje,
        registros_tabla
    )



# =========================================================
# INICIO
# =========================================================

def abrir_en_modo_aplicacion(url):
    navegadores = [
        (
            "Microsoft Edge",
            "msedge.exe",
            "Microsoft",
            "Edge",
            "Application",
            "msedge.exe"
        ),
        ("Google Chrome", "chrome.exe", "Google", "Chrome", "Application",
         "chrome.exe"),
    ]

    ejecutables = []
    for navegador in navegadores:
        nombre, comando, *rutas = navegador
        encontrado = shutil.which(comando)
        if encontrado:
            ejecutables.append((nombre, encontrado))

        for variable in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
            directorio = os.environ.get(variable)
            if directorio:
                ejecutable = Path(directorio).joinpath(*rutas)
                if ejecutable.is_file():
                    ejecutables.append((nombre, str(ejecutable)))

    intentados = set()
    for nombre, ejecutable in ejecutables:
        if ejecutable in intentados:
            continue
        intentados.add(ejecutable)
        try:
            subprocess.Popen(
                [ejecutable, f"--app={url}", "--no-first-run"],
                close_fds=True
            )
            return
        except OSError:
            continue

    raise RuntimeError(
        "No se pudo abrir Pizarra en modo aplicación. "
        "Instala Microsoft Edge o Google Chrome y vuelve a intentarlo."
    )


if __name__ == "__main__":
    if getattr(sys, "frozen", False):
        servidor = make_server(
            "127.0.0.1",
            0,
            app.server,
            threaded=True
        )
        hilo_servidor = threading.Thread(
            target=servidor.serve_forever,
            daemon=True
        )
        hilo_servidor.start()
        url = f"http://127.0.0.1:{servidor.server_port}"
        print("Abriendo Pizarra en una ventana de aplicación.")
        try:
            abrir_en_modo_aplicacion(url)
            hilo_servidor.join()
        except RuntimeError:
            servidor.shutdown()
            hilo_servidor.join()
            raise
        except KeyboardInterrupt:
            servidor.shutdown()
            hilo_servidor.join()
    else:
        app.run(debug=False)
