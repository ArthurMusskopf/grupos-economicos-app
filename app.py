import re
import math
import time
from datetime import datetime

import streamlit as st
import pandas as pd
import networkx as nx

from st_cytoscape import cytoscape
from google.cloud import bigquery
from google.oauth2 import service_account

# ============================================================
# CONFIG STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Grupos Econômicos por CNPJ",
    layout="wide",
)

st.title("🔍 Análise de Grupo Econômico por CNPJ (Base dos Dados / CNPJ)")
st.markdown(
    "Informe um CNPJ e o app vai buscar empresa foco, sócios, "
    "empresas vinculadas e montar o grafo econômico."
)

# ============================================================
# SESSION STATE
# ============================================================

if "logs" not in st.session_state:
    st.session_state["logs"] = []

if "selected_nodes" not in st.session_state:
    st.session_state["selected_nodes"] = []

if "ultimo_cnpj" not in st.session_state:
    st.session_state["ultimo_cnpj"] = None

if "grafo_data" not in st.session_state:
    st.session_state["grafo_data"] = None

# ------------------------------------------------------------
# Logger simples (sidebar)
# ------------------------------------------------------------
log_placeholder = st.sidebar.empty()


def render_logs():
    log_placeholder.text("\n".join(st.session_state["logs"][-40:]))


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state["logs"].append(f"[{ts}] {msg}")
    render_logs()


render_logs()
st.sidebar.markdown("### ⚙️ Log de execução")

# ============================================================
# BIGQUERY CLIENT
# ============================================================


def get_bq_client():
    """
    Cria o cliente BigQuery.

    Opções:
    1) Streamlit Cloud: JSON da service account em st.secrets["gcp_service_account"]
    2) Local: gcloud auth application-default login ou GOOGLE_APPLICATION_CREDENTIALS
    """
    if "gcp_service_account" in st.secrets:
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        project_id = creds.project_id
        client = bigquery.Client(project=project_id, credentials=creds)
        return client

    client = bigquery.Client()
    return client


# ============================================================
# HELPERS DE CNPJ
# ============================================================

def normalizar_cnpj(cnpj_str: str) -> str:
    """Remove tudo que não for dígito e valida tamanho."""
    digitos = re.sub(r"\D", "", cnpj_str or "")
    if len(digitos) != 14:
        raise ValueError("CNPJ deve ter 14 dígitos após remover separadores.")
    return digitos


def extrair_cnpj_basico(cnpj_14: str) -> str:
    """Primeiros 8 dígitos do CNPJ."""
    return cnpj_14[:8]


# ============================================================
# QUERIES BIGQUERY
# ============================================================

def consultar_empresa_foco(client: bigquery.Client, cnpj_basico: str) -> pd.DataFrame:
    log("Consultando empresa foco (tabela empresas)...")
    sql = """
    WITH 
    dicionario_qualificacao_responsavel AS (
        SELECT
            chave AS chave_qualificacao_responsavel,
            valor AS descricao_qualificacao_responsavel
        FROM `basedosdados.br_me_cnpj.dicionario`
        WHERE
            nome_coluna = 'qualificacao_responsavel'
            AND id_tabela = 'empresas'
    ),
    dicionario_porte AS (
        SELECT
            chave AS chave_porte,
            valor AS descricao_porte
        FROM `basedosdados.br_me_cnpj.dicionario`
        WHERE
            nome_coluna = 'porte'
            AND id_tabela = 'empresas'
    )
    SELECT
        dados.ano as ano,
        dados.mes as mes,
        dados.data as data,
        dados.cnpj_basico as cnpj_basico,
        dados.razao_social as razao_social,
        dados.natureza_juridica AS natureza_juridica,
        diretorio_natureza_juridica.descricao AS natureza_juridica_descricao,
        descricao_qualificacao_responsavel AS qualificacao_responsavel,
        dados.capital_social as capital_social,
        descricao_porte AS porte,
        dados.ente_federativo as ente_federativo
    FROM `basedosdados.br_me_cnpj.empresas` AS dados
    LEFT JOIN (
        SELECT DISTINCT id_natureza_juridica, descricao
        FROM `basedosdados.br_bd_diretorios_brasil.natureza_juridica`
    ) AS diretorio_natureza_juridica
        ON dados.natureza_juridica = diretorio_natureza_juridica.id_natureza_juridica
    LEFT JOIN dicionario_qualificacao_responsavel
        ON dados.qualificacao_responsavel = chave_qualificacao_responsavel
    LEFT JOIN dicionario_porte
        ON dados.porte = chave_porte
    WHERE dados.cnpj_basico = @cnpj_basico
    ORDER BY ano DESC, mes DESC, data DESC
    LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("cnpj_basico", "STRING", cnpj_basico)
        ]
    )
    df = client.query(sql, job_config=job_config).to_dataframe()
    return df


def consultar_socios_foco(client: bigquery.Client, cnpj_basico: str) -> pd.DataFrame:
    log("Consultando sócios da empresa foco (tabela socios)...")
    sql = """
    WITH 
    dicionario_tipo AS (
        SELECT
            chave AS chave_tipo,
            valor AS descricao_tipo
        FROM `basedosdados.br_me_cnpj.dicionario`
        WHERE
            nome_coluna = 'tipo'
            AND id_tabela = 'socios'
    ),
    dicionario_qualificacao AS (
        SELECT
            chave AS chave_qualificacao,
            valor AS descricao_qualificacao
        FROM `basedosdados.br_me_cnpj.dicionario`
        WHERE
            nome_coluna = 'qualificacao'
            AND id_tabela = 'socios'
    ),
    dicionario_id_pais AS (
        SELECT
            chave AS chave_id_pais,
            valor AS descricao_id_pais
        FROM `basedosdados.br_me_cnpj.dicionario`
        WHERE
            nome_coluna = 'id_pais'
            AND id_tabela = 'socios'
    ),
    dicionario_qualificacao_representante_legal AS (
        SELECT
            chave AS chave_qualificacao_representante_legal,
            valor AS descricao_qualificacao_representante_legal
        FROM `basedosdados.br_me_cnpj.dicionario`
        WHERE
            nome_coluna = 'qualificacao_representante_legal'
            AND id_tabela = 'socios'
    ),
    dicionario_faixa_etaria AS (
        SELECT
            chave AS chave_faixa_etaria,
            valor AS descricao_faixa_etaria
        FROM `basedosdados.br_me_cnpj.dicionario`
        WHERE
            nome_coluna = 'faixa_etaria'
            AND id_tabela = 'socios'
    )
    SELECT
        dados.ano as ano,
        dados.mes as mes,
        dados.data as data,
        dados.tipo as tipo,
        dados.nome as nome,
        dados.documento as documento,
        dados.qualificacao as qualificacao,
        dados.percentual_capital_social as percentual_capital_social,
        dados.data_entrada_sociedade as data_entrada_sociedade,
        dados.id_pais as id_pais,
        dados.cpf_representante_legal as cpf_representante_legal,
        dados.nome_representante_legal as nome_representante_legal,
        dados.qualificacao_representante_legal as qualificacao_representante_legal,
        dados.faixa_etaria as faixa_etaria,
        descricao_tipo,
        descricao_qualificacao,
        descricao_id_pais,
        descricao_qualificacao_representante_legal,
        descricao_faixa_etaria
    FROM `basedosdados.br_me_cnpj.socios` AS dados
    LEFT JOIN dicionario_tipo
        ON dados.tipo = chave_tipo
    LEFT JOIN dicionario_qualificacao
        ON dados.qualificacao = chave_qualificacao
    LEFT JOIN dicionario_id_pais
        ON dados.id_pais = chave_id_pais
    LEFT JOIN dicionario_qualificacao_representante_legal
        ON dados.qualificacao_representante_legal = chave_qualificacao_representante_legal
    LEFT JOIN dicionario_faixa_etaria
        ON dados.faixa_etaria = chave_faixa_etaria
    WHERE dados.cnpj_basico = @cnpj_basico
    ORDER BY ano DESC, mes DESC, data DESC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("cnpj_basico", "STRING", cnpj_basico)
        ]
    )
    df = client.query(sql, job_config=job_config).to_dataframe()
    return df


def selecionar_qsa_atual(df_socios: pd.DataFrame) -> pd.DataFrame:
    log("Filtrando QSA atual (ano + mes + data mais recente)...")
    if df_socios.empty:
        return df_socios

    df_socios = df_socios.sort_values(["ano", "mes", "data"], ascending=False)

    primeiro = df_socios.iloc[0]
    ano_ref = primeiro["ano"]
    mes_ref = primeiro["mes"]
    data_ref = primeiro["data"]

    df_qsa = df_socios[
        (df_socios["ano"] == ano_ref)
        & (df_socios["mes"] == mes_ref)
        & (df_socios["data"] == data_ref)
    ].copy()

    log(f"QSA selecionado: {len(df_qsa)} registros (ano={ano_ref}, mes={mes_ref}, data={data_ref})")
    return df_qsa


def consultar_empresas_vinculadas_por_nome(
    client: bigquery.Client,
    df_socios_qsa: pd.DataFrame,
    cnpj_basico_foco: str,
) -> pd.DataFrame:
    log("Consultando empresas vinculadas via nomes dos sócios...")

    if df_socios_qsa.empty:
        log("QSA vazio, pulando busca de empresas vinculadas.")
        return pd.DataFrame()

    nomes_socios = (
        df_socios_qsa[df_socios_qsa["nome"].notna()]["nome"].unique().tolist()
    )

    if not nomes_socios:
        log("Nenhum nome de sócio disponível para busca.")
        return pd.DataFrame()

    log(f"{len(nomes_socios)} nomes de sócios únicos para buscar empresas vinculadas.")

    sql = """
    WITH 
    dicionario_tipo AS (
        SELECT
            chave AS chave_tipo,
            valor AS descricao_tipo
        FROM `basedosdados.br_me_cnpj.dicionario`
        WHERE
            nome_coluna = 'tipo'
            AND id_tabela = 'socios'
    ),
    dicionario_qualificacao AS (
        SELECT
            chave AS chave_qualificacao,
            valor AS descricao_qualificacao
        FROM `basedosdados.br_me_cnpj.dicionario`
        WHERE
            nome_coluna = 'qualificacao'
            AND id_tabela = 'socios'
    ),
    dicionario_id_pais AS (
        SELECT
            chave AS chave_id_pais,
            valor AS descricao_id_pais
        FROM `basedosdados.br_me_cnpj.dicionario`
        WHERE
            nome_coluna = 'id_pais'
            AND id_tabela = 'socios'
    ),
    dicionario_qualificacao_representante_legal AS (
        SELECT
            chave AS chave_qualificacao_representante_legal,
            valor AS descricao_qualificacao_representante_legal
        FROM `basedosdados.br_me_cnpj.dicionario`
        WHERE
            nome_coluna = 'qualificacao_representante_legal'
            AND id_tabela = 'socios'
    ),
    dicionario_faixa_etaria AS (
        SELECT
            chave AS chave_faixa_etaria,
            valor AS descricao_faixa_etaria
        FROM `basedosdados.br_me_cnpj.dicionario`
        WHERE
            nome_coluna = 'faixa_etaria'
            AND id_tabela = 'socios'
    ),
    dicionario_qualificacao_responsavel AS (
        SELECT
            chave AS chave_qualificacao_responsavel,
            valor AS descricao_qualificacao_responsavel
        FROM `basedosdados.br_me_cnpj.dicionario`
        WHERE
            nome_coluna = 'qualificacao_responsavel'
            AND id_tabela = 'empresas'
    ),
    dicionario_porte AS (
        SELECT
            chave AS chave_porte,
            valor AS descricao_porte
        FROM `basedosdados.br_me_cnpj.dicionario`
        WHERE
            nome_coluna = 'porte'
            AND id_tabela = 'empresas'
    )
    SELECT
        s.ano as ano,
        s.mes as mes,
        s.data as data,
        s.cnpj_basico as cnpj_basico,
        s.tipo as tipo,
        s.nome as nome_socio,
        s.documento as documento,
        s.qualificacao as qualificacao,
        s.percentual_capital_social as percentual_capital_social,
        s.data_entrada_sociedade as data_entrada_sociedade,
        s.id_pais as id_pais,
        s.cpf_representante_legal as cpf_representante_legal,
        s.nome_representante_legal as nome_representante_legal,
        s.qualificacao_representante_legal as qualificacao_representante_legal,
        s.faixa_etaria as faixa_etaria,
        descricao_tipo,
        descricao_qualificacao,
        descricao_id_pais,
        descricao_qualificacao_representante_legal,
        descricao_faixa_etaria,
        e.razao_social as razao_social,
        e.natureza_juridica as natureza_juridica,
        diretorio_natureza_juridica.descricao AS natureza_juridica_descricao,
        descricao_qualificacao_responsavel AS qualificacao_responsavel,
        e.capital_social as capital_social,
        descricao_porte AS porte,
        e.ente_federativo as ente_federativo
    FROM `basedosdados.br_me_cnpj.socios` AS s
    LEFT JOIN dicionario_tipo
        ON s.tipo = chave_tipo
    LEFT JOIN dicionario_qualificacao
        ON s.qualificacao = chave_qualificacao
    LEFT JOIN dicionario_id_pais
        ON s.id_pais = chave_id_pais
    LEFT JOIN dicionario_qualificacao_representante_legal
        ON s.qualificacao_representante_legal = chave_qualificacao_representante_legal
    LEFT JOIN dicionario_faixa_etaria
        ON s.faixa_etaria = chave_faixa_etaria
    LEFT JOIN `basedosdados.br_me_cnpj.empresas` AS e
        ON s.cnpj_basico = e.cnpj_basico
        AND s.ano = e.ano
        AND s.mes = e.mes
        AND s.data = e.data
    LEFT JOIN (
        SELECT DISTINCT id_natureza_juridica, descricao
        FROM `basedosdados.br_bd_diretorios_brasil.natureza_juridica`
    ) AS diretorio_natureza_juridica
        ON e.natureza_juridica = diretorio_natureza_juridica.id_natureza_juridica
    LEFT JOIN dicionario_qualificacao_responsavel
        ON e.qualificacao_responsavel = chave_qualificacao_responsavel
    LEFT JOIN dicionario_porte
        ON e.porte = chave_porte
    WHERE s.nome IN UNNEST(@nomes_socios)
      AND s.cnpj_basico != @cnpj_basico_foco
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("nomes_socios", "STRING", nomes_socios),
            bigquery.ScalarQueryParameter("cnpj_basico_foco", "STRING", cnpj_basico_foco),
        ]
    )
    df = client.query(sql, job_config=job_config).to_dataframe()
    log(f"{len(df)} registros de empresas vinculadas retornados.")
    return df


# ============================================================
# CONSTRUÇÃO DO GRAFO (NetworkX)
# ============================================================

def construir_grafo(
    df_empresa_foco: pd.DataFrame,
    df_socios_qsa: pd.DataFrame,
    df_empresas_vinc: pd.DataFrame,
    cnpj_basico_foco: str,
) -> nx.Graph:
    log("Construindo grafo (NetworkX)...")
    G = nx.Graph()

    # -----------------------------------------------------------
    # 1) Nó "FOCO" (empresa foco)
    # -----------------------------------------------------------
    row_emp = df_empresa_foco.iloc[0]
    razao_foco = row_emp["razao_social"]
    label_foco = f"{razao_foco}\n(CNPJ: {cnpj_basico_foco})"

    G.add_node(
        "FOCO",
        label=label_foco,
        tipo="empresa",
        nivel="foco",
        nome=razao_foco,
        cnpj_basico=cnpj_basico_foco,
        eh_empresa_foco=True,
    )

    # -----------------------------------------------------------
    # 2) Sócios da empresa foco (nível "socio")
    # -----------------------------------------------------------
    for idx, row in df_socios_qsa.iterrows():
        nome_socio = row.get("nome")
        if not nome_socio or pd.isna(nome_socio):
            continue

        node_id_socio = f"SOCIO_{nome_socio}"

        doc = row.get("documento", "")
        tipo_val = row.get("descricao_tipo", row.get("tipo", ""))

        if not G.has_node(node_id_socio):
            label_socio = f"{nome_socio}\n({tipo_val})"
            G.add_node(
                node_id_socio,
                label=label_socio,
                tipo="socio",
                nivel="socio",
                nome=nome_socio,
                cnpj_basico=None,
            )

        qual = row.get("descricao_qualificacao", row.get("qualificacao", ""))
        perc = row.get("percentual_capital_social")
        perc_str = f"{perc}%" if perc and not pd.isna(perc) else ""

        edge_label = f"{qual}\n{perc_str}".strip()

        G.add_edge(
            "FOCO",
            node_id_socio,
            label=edge_label,
        )

    # -----------------------------------------------------------
    # 3) Empresas vinculadas (nível "empresa_vinc")
    #    Fazemos dedup por (cnpj_basico, nome_socio) último registro
    # -----------------------------------------------------------
    df_vinc_sorted = df_empresas_vinc.copy()
    df_vinc_sorted["data"] = pd.to_datetime(df_vinc_sorted["data"], errors="coerce")
    df_vinc_sorted = df_vinc_sorted.sort_values(
        ["cnpj_basico", "nome_socio", "data"], ascending=[True, True, False]
    )
    df_vinc_dedup = df_vinc_sorted.drop_duplicates(
        subset=["cnpj_basico", "nome_socio"], keep="first"
    )

    for idx, row in df_vinc_dedup.iterrows():
        cnpj_emp = row["cnpj_basico"]
        razao_emp = row["razao_social"]
        nome_socio = row["nome_socio"]

        if not cnpj_emp or pd.isna(cnpj_emp):
            continue
        if not razao_emp or pd.isna(razao_emp):
            razao_emp = f"(CNPJ: {cnpj_emp})"

        node_id_emp = f"EMP_{cnpj_emp}"

        if not G.has_node(node_id_emp):
            label_emp = f"{razao_emp}\n(CNPJ: {cnpj_emp})"
            G.add_node(
                node_id_emp,
                label=label_emp,
                tipo="empresa",
                nivel="empresa_vinc",
                nome=razao_emp,
                cnpj_basico=str(cnpj_emp),
            )

        node_id_socio = f"SOCIO_{nome_socio}"
        if G.has_node(node_id_socio):
            qual = row.get("descricao_qualificacao", row.get("qualificacao", ""))
            perc = row.get("percentual_capital_social")
            perc_str = f"{perc}%" if perc and not pd.isna(perc) else ""

            edge_label = f"{qual}\n{perc_str}".strip()

            G.add_edge(
                node_id_emp,
                node_id_socio,
                label=edge_label,
            )

    log(f"Grafo construído: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas.")
    return G


# ============================================================
# LAYOUT E ELEMENTOS CYTOSCAPE
# ============================================================

def compute_positions(G: nx.Graph, foco_id: str, seed: int = 42, k: float = 2.0):
    """
    Usa spring_layout para posicionar nós, forçando 'foco_id' no centro.
    """
    log("Calculando layout de posições dos nós (spring layout)...")
    pos = nx.spring_layout(G, seed=seed, k=k)

    if foco_id in pos:
        pos[foco_id] = (0, 0)

    return pos


def build_cytoscape_elements(G: nx.Graph, pos: dict, selected_ids: list):
    """
    Monta lista de elementos (nós e arestas) para o cytoscape.
    selected_ids é uma lista de IDs de nós selecionados (para highlight).
    """
    elements = []

    # Converte posições para pixels
    xvals = [xy[0] for xy in pos.values()]
    yvals = [xy[1] for xy in pos.values()]
    xmin, xmax = min(xvals), max(xvals)
    ymin, ymax = min(yvals), max(yvals)

    margin = 100
    canvas_width = 1500
    canvas_height = 1500

    def scale_x(x):
        if xmax == xmin:
            return canvas_width / 2
        return margin + (x - xmin) / (xmax - xmin) * (canvas_width - 2 * margin)

    def scale_y(y):
        if ymax == ymin:
            return canvas_height / 2
        return margin + (y - ymin) / (ymax - ymin) * (canvas_height - 2 * margin)

    # Nós
    for node_id in G.nodes():
        data = G.nodes[node_id]
        nivel = data.get("nivel")
        tipo = data.get("tipo")

        x, y = pos.get(node_id, (0, 0))
        px = scale_x(x)
        py = scale_y(y)

        classes = []
        if nivel == "foco":
            classes.append("foco")
        elif nivel == "socio":
            classes.append("socio")
        elif nivel == "empresa_vinc":
            classes.append("empresa-vinc")

        if node_id in selected_ids:
            classes.append("selected")

        elements.append(
            {
                "data": {
                    "id": node_id,
                    "label": data.get("label", node_id),
                    "tipo": tipo,
                    "nivel": nivel,
                },
                "position": {"x": px, "y": py},
                "classes": " ".join(classes),
                "selected": (node_id in selected_ids),
            }
        )

    # Arestas
    for u, v, edge_data in G.edges(data=True):
        edge_id = f"{u}-{v}"
        
        classes = []
        # Se ambos os nós estão selecionados, destacar a aresta
        if u in selected_ids and v in selected_ids:
            classes.append("highlighted")

        elements.append(
            {
                "data": {
                    "id": edge_id,
                    "source": u,
                    "target": v,
                    "label": edge_data.get("label", ""),
                },
                "classes": " ".join(classes),
            }
        )

    return elements


def get_stylesheet():
    """
    Stylesheet para o Cytoscape.
    """
    return [
        # Nós em geral
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "text-valign": "center",
                "text-halign": "center",
                "font-size": "12px",
                "text-wrap": "wrap",
                "text-max-width": "120px",
                "background-color": "#999",
                "border-width": 2,
                "border-color": "#666",
                "width": 60,
                "height": 60,
            },
        },
        # Nó Foco (empresa foco)
        {
            "selector": "node.foco",
            "style": {
                "background-color": "#FF6B6B",
                "border-color": "#C92A2A",
                "width": 80,
                "height": 80,
                "font-size": "14px",
                "font-weight": "bold",
            },
        },
        # Sócio
        {
            "selector": "node.socio",
            "style": {
                "background-color": "#4ECDC4",
                "border-color": "#0A9698",
                "shape": "ellipse",
            },
        },
        # Empresa Vinculada
        {
            "selector": "node.empresa-vinc",
            "style": {
                "background-color": "#95E1D3",
                "border-color": "#38A89D",
            },
        },
        # Nó selecionado
        {
            "selector": "node.selected",
            "style": {
                "border-width": 4,
                "border-color": "#FFD700",
                "background-color": "#FFA500",
            },
        },
        # Arestas em geral
        {
            "selector": "edge",
            "style": {
                "width": 2,
                "line-color": "#ccc",
                "target-arrow-color": "#ccc",
                "target-arrow-shape": "triangle",
                "curve-style": "bezier",
                "label": "data(label)",
                "font-size": "10px",
                "text-rotation": "autorotate",
                "text-background-opacity": 1,
                "text-background-color": "#fff",
                "text-background-padding": "3px",
            },
        },
        # Aresta destacada (quando ambos nós estão selecionados)
        {
            "selector": "edge.highlighted",
            "style": {
                "width": 4,
                "line-color": "#FFD700",
                "target-arrow-color": "#FFD700",
            },
        },
        # Nós selecionados (estado interno do cytoscape)
        {
            "selector": ":selected",
            "style": {
                "border-width": 4,
                "border-color": "#FFD700",
            },
        },
    ]


# ============================================================
# INTERFACE: BUSCA
# ============================================================

with st.form("form_cnpj"):
    cnpj_input = st.text_input(
        "Digite o CNPJ (14 dígitos, com ou sem separadores):",
        placeholder="00.000.000/0000-00",
    )
    submitted = st.form_submit_button("🔍 Buscar grupo econômico", use_container_width=True)

if submitted and cnpj_input.strip():
    progress = st.empty()

    def set_progress(perc: int, msg: str):
        progress.progress(perc / 100, text=msg)

    try:
        set_progress(5, "Normalizando CNPJ...")
        cnpj_14 = normalizar_cnpj(cnpj_input.strip())
        cnpj_basico = extrair_cnpj_basico(cnpj_14)
        st.session_state["ultimo_cnpj"] = cnpj_basico

        client = get_bq_client()
        log(f"BigQuery conectado no projeto: {client.project}")

        set_progress(15, "Consultando empresa foco...")
        df_empresa_foco = consultar_empresa_foco(client, cnpj_basico)
        if df_empresa_foco.empty:
            st.error("Empresa foco não encontrada para esse CNPJ.")
            progress.empty()
            st.stop()

        set_progress(30, "Consultando sócios da empresa foco...")
        df_socios_foco = consultar_socios_foco(client, cnpj_basico)

        set_progress(45, "Selecionando QSA mais recente...")
        df_socios_qsa = selecionar_qsa_atual(df_socios_foco)

        set_progress(60, "Consultando empresas vinculadas...")
        df_empresas_vinc = consultar_empresas_vinculadas_por_nome(
            client, df_socios_qsa, cnpj_basico
        )

        set_progress(75, "Construindo grafo...")
        G = construir_grafo(df_empresa_foco, df_socios_qsa, df_empresas_vinc, cnpj_basico)

        st.session_state["grafo_data"] = {
            "cnpj_basico": cnpj_basico,
            "df_empresa_foco": df_empresa_foco,
            "df_socios_qsa": df_socios_qsa,
            "df_empresas_vinc": df_empresas_vinc,
            "G": G,
        }
        st.session_state["selected_nodes"] = []

        set_progress(100, "Pronto! Grafo gerado com sucesso.")
        time.sleep(0.5)
        progress.empty()

    except Exception as e:
        st.error(f"Erro durante a execução: {e}")
        log(f"ERRO: {e}")
        progress.empty()

# ------------------------------------------------------------
# VISUALIZAÇÃO DO GRAFO + TABELAS
# ------------------------------------------------------------
grafo_data = st.session_state.get("grafo_data")

if grafo_data is not None:
    cnpj_basico = grafo_data["cnpj_basico"]
    df_empresa_foco = grafo_data["df_empresa_foco"]
    df_socios_qsa = grafo_data["df_socios_qsa"]
    df_empresas_vinc = grafo_data["df_empresas_vinc"]
    G = grafo_data["G"]

    # seleção atual armazenada (multi-seleção)
    selected_ids_state = st.session_state.get("selected_nodes", [])

    # layout fixo para este CNPJ
    pos = compute_positions(G, foco_id="FOCO", seed=42, k=2.0)
    elements = build_cytoscape_elements(G, pos, selected_ids_state)
    stylesheet = get_stylesheet()

    col_graph, col_tables = st.columns([3, 2])

    with col_graph:
        st.markdown("### 🌐 Grafo do grupo econômico")

        selection = cytoscape(
            elements=elements,
            stylesheet=stylesheet,
            layout={"name": "preset", "fit": True, "padding": 140},
            width="100%",
            height="700px",
            selection_type="additive",
            key="grafo-cnpj",
        )

    # nova seleção vinda do front
    new_selected_ids = selection.get("nodes", [])

    # CORREÇÃO: apenas atualiza o session_state, SEM fazer st.rerun()
    # Isso permite que o componente cytoscape mantenha seu estado interno de seleção
    if new_selected_ids != selected_ids_state:
        st.session_state["selected_nodes"] = new_selected_ids

    # usa a seleção do session_state para mostrar detalhes
    selected_ids = st.session_state.get("selected_nodes", [])

    with col_tables:
        st.markdown("### 📊 Detalhes dos nós selecionados")

        if not selected_ids:
            st.info("Selecione um ou mais nós no grafo para ver detalhes aqui.")
        else:
            st.write(f"Nós selecionados: {selected_ids}")

            for node_id in selected_ids:
                attrs = G.nodes[node_id]
                tipo_no = attrs.get("tipo")
                nivel = attrs.get("nivel")
                nome = attrs.get("nome")
                cnpj_b = attrs.get("cnpj_basico")

                st.markdown(f"#### Nó `{node_id}` – {attrs.get('label', '')}")
                info = {
                    "id_no": node_id,
                    "nivel": nivel,
                    "tipo_no": tipo_no,
                    "nome": nome,
                    "cnpj_basico": cnpj_b,
                    "eh_empresa_foco": attrs.get("eh_empresa_foco"),
                }
                st.dataframe(pd.DataFrame([info]))

                # EMPRESA FOCO
                if nivel == "foco":
                    st.markdown("**Empresa foco**")
                    st.dataframe(df_empresa_foco)

                    if not df_socios_qsa.empty:
                        st.markdown("**QSA atual da empresa foco**")
                        cols = [
                            "ano",
                            "mes",
                            "data",
                            "tipo",
                            "nome",
                            "documento",
                            "qualificacao",
                            "data_entrada_sociedade",
                            "faixa_etaria",
                        ]
                        cols = [c for c in cols if c in df_socios_qsa.columns]
                        st.dataframe(df_socios_qsa[cols].sort_values("nome"))

                # SÓCIO (PF/PJ)
                elif nivel == "socio" and nome is not None:
                    # registro do sócio na empresa foco (mais recente primeiro)
                    df_socio_foco = (
                        df_socios_qsa[df_socios_qsa["nome"] == nome]
                        .sort_values("data_entrada_sociedade", ascending=False)
                    )
                    if not df_socio_foco.empty:
                        st.markdown(
                            "**Registro do sócio no QSA da empresa foco "
                            "(mais recente primeiro)**"
                        )
                        st.dataframe(df_socio_foco)

                    # empresas em que o sócio aparece – DEDUP por CNPJ (último registro)
                    df_emp_socio = df_empresas_vinc[
                        df_empresas_vinc["nome_socio"] == nome
                    ].copy()

                    if not df_emp_socio.empty:
                        df_emp_socio["data"] = pd.to_datetime(
                            df_emp_socio["data"], errors="coerce"
                        )
                        df_emp_socio = df_emp_socio.sort_values(
                            ["cnpj_basico", "data"], ascending=[True, False]
                        )
                        df_emp_socio = df_emp_socio.drop_duplicates(
                            subset=["cnpj_basico"], keep="first"
                        )
                        df_emp_socio = df_emp_socio.sort_values("razao_social")

                        st.markdown("**Empresas em que este sócio aparece no QSA**")
                        st.dataframe(df_emp_socio)

                # EMPRESA VINCULADA
                elif nivel == "empresa_vinc" and cnpj_b is not None:
                    df_emp = df_empresas_vinc[
                        df_empresas_vinc["cnpj_basico"].astype(str) == str(cnpj_b)
                    ].copy()

                    if not df_emp.empty:
                        df_emp_info = (
                            df_emp.drop_duplicates(subset=["cnpj_basico"])[
                                [
                                    "cnpj_basico",
                                    "razao_social",
                                    "natureza_juridica_descricao",
                                    "capital_social",
                                ]
                            ]
                        )
                        st.markdown("**Empresa vinculada**")
                        st.dataframe(df_emp_info)

                        # QSA deduplicado por sócio (último registro por nome_socio)
                        df_qsa_emp = df_emp[
                            [
                                "nome_socio",
                                "documento",
                                "qualificacao",
                                "data_entrada_sociedade",
                                "data",
                            ]
                        ].copy()

                        df_qsa_emp["data"] = pd.to_datetime(
                            df_qsa_emp["data"], errors="coerce"
                        )
                        df_qsa_emp = df_qsa_emp.sort_values(
                            ["nome_socio", "data"], ascending=[True, False]
                        )
                        df_qsa_emp = df_qsa_emp.drop_duplicates(
                            subset=["nome_socio"], keep="first"
                        ).sort_values("nome_socio")

                        st.markdown("**QSA (amostra a partir dos sócios do grupo)**")
                        st.dataframe(df_qsa_emp)
