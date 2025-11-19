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

if "layout_seed" not in st.session_state:
    st.session_state["layout_seed"] = 42

if "ultimo_cnpj" not in st.session_state:
    st.session_state["ultimo_cnpj"] = None

# ------------------------------------------------------------
# Logger simples (na sidebar)
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
    1) Streamlit Cloud: guardar o JSON da service account em st.secrets["gcp_service_account"]
    2) Local: usar gcloud auth application-default login ou GOOGLE_APPLICATION_CREDENTIALS
    """
    # 1) Service Account via secrets (ideal pra Cloud)
    if "gcp_service_account" in st.secrets:
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        project_id = creds.project_id
        client = bigquery.Client(project=project_id, credentials=creds)
        return client

    # 2) Tenta ADC (local)
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
        dados.cnpj_basico as cnpj_basico,
        descricao_tipo AS tipo,
        dados.nome as nome,
        dados.documento as documento,
        descricao_qualificacao AS qualificacao,
        dados.data_entrada_sociedade as data_entrada_sociedade,
        descricao_id_pais AS id_pais,
        dados.cpf_representante_legal as cpf_representante_legal,
        dados.nome_representante_legal as nome_representante_legal,
        descricao_qualificacao_representante_legal AS qualificacao_representante_legal,
        descricao_faixa_etaria AS faixa_etaria
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
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("cnpj_basico", "STRING", cnpj_basico)
        ]
    )
    df = client.query(sql, job_config=job_config).to_dataframe()
    return df


def selecionar_qsa_atual(df_socios_foco: pd.DataFrame) -> pd.DataFrame:
    """
    A partir de todos os sócios da empresa foco, pega o QSA mais recente:
    - identifica a maior data,
    - mantém apenas as linhas com essa data,
    - deduplica por nome (um registro por sócio).
    """
    if df_socios_foco.empty:
        return df_socios_foco

    df = df_socios_foco.copy()
    df["data"] = pd.to_datetime(df["data"])
    data_max = df["data"].max()
    df_qsa = df[df["data"] == data_max].copy()

    # remove duplicados por nome, se houver
    df_qsa = df_qsa.sort_values("data_entrada_sociedade", ascending=False)
    df_qsa = df_qsa.drop_duplicates(subset=["nome"]).reset_index(drop=True)

    log(f"QSA atual identificado em {data_max.date()}, {len(df_qsa)} sócios únicos.")
    return df_qsa


def consultar_empresas_vinculadas_por_nome(
    client: bigquery.Client, df_socios_qsa: pd.DataFrame, cnpj_basico_foco: str
) -> pd.DataFrame:
    """
    Consulta empresas vinculadas usando NOME do sócio (não documento).
    Pega todos os registros nos quais o nome aparece na tabela socios
    e junta com empresas para trazer razão social / natureza / capital.
    """
    log("Consultando empresas vinculadas (por nome de sócio)...")

    nomes_socios = df_socios_qsa["nome"].dropna().unique().tolist()
    if not nomes_socios:
        log("Nenhum nome de sócio encontrado no QSA atual.")
        return pd.DataFrame()

    sql = """
    WITH nomes_socios AS (
        SELECT DISTINCT nome
        FROM UNNEST(@lista_nomes) AS nome
    ),
    socios_vinc AS (
        SELECT
            s.ano,
            s.mes,
            s.data,
            s.cnpj_basico,
            s.nome AS nome_socio,
            s.documento,
            s.qualificacao,
            s.data_entrada_sociedade
        FROM `basedosdados.br_me_cnpj.socios` AS s
        JOIN nomes_socios n
            ON s.nome = n.nome
    ),
    empresas_completa AS (
        SELECT
            e.ano,
            e.mes,
            e.data,
            e.cnpj_basico,
            e.razao_social,
            e.natureza_juridica,
            nj.descricao AS natureza_juridica_descricao,
            e.capital_social
        FROM `basedosdados.br_me_cnpj.empresas` AS e
        LEFT JOIN `basedosdados.br_bd_diretorios_brasil.natureza_juridica` nj
            ON e.natureza_juridica = nj.id_natureza_juridica
    )
    SELECT
        sv.nome_socio,
        sv.documento,
        sv.qualificacao,
        sv.data_entrada_sociedade,
        sv.cnpj_basico,
        ec.razao_social,
        ec.natureza_juridica_descricao,
        ec.capital_social,
        sv.ano,
        sv.mes,
        sv.data
    FROM socios_vinc sv
    LEFT JOIN empresas_completa ec
        ON sv.cnpj_basico = ec.cnpj_basico
    WHERE sv.cnpj_basico != @cnpj_foco
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("lista_nomes", "STRING", nomes_socios),
            bigquery.ScalarQueryParameter("cnpj_foco", "STRING", cnpj_basico_foco),
        ]
    )

    df = client.query(sql, job_config=job_config).to_dataframe()
    log(f"Empresas vinculadas consultadas: {len(df)} linhas.")
    return df


# ============================================================
# GRAFO (NETWORKX)
# ============================================================

def construir_grafo(
    df_empresa_foco: pd.DataFrame,
    df_socios_qsa: pd.DataFrame,
    df_empresas_vinc: pd.DataFrame,
    cnpj_basico_foco: str,
) -> nx.Graph:
    """
    Cria grafo:
    - nó foco (empresa raiz)
    - nós sócios (PF/PJ) ligados ao foco
    - nós empresas vinculadas ligadas aos sócios
    """
    G = nx.Graph()

    # nó foco
    if df_empresa_foco.empty:
        raise ValueError("Empresa foco não encontrada no CNPJ básico informado.")

    row_foco = df_empresa_foco.iloc[0]
    foco_id = "FOCO"
    G.add_node(
        foco_id,
        nivel="foco",
        tipo="PJ",
        label=row_foco["razao_social"],
        nome=row_foco["razao_social"],
        cnpj_basico=cnpj_basico_foco,
        eh_empresa_foco=True,
    )

    # sócios do QSA atual (nível 2)
    socios_qsa = df_socios_qsa.copy()
    socios_qsa = socios_qsa.dropna(subset=["nome"])
    socios_qsa = socios_qsa.drop_duplicates(subset=["nome"]).reset_index(drop=True)

    map_nome_to_node = {}

    for i, r in socios_qsa.iterrows():
        nome = r["nome"]
        tipo_str = str(r["tipo"] or "")
        tipo_node = "PF" if "Física" in tipo_str else "PJ"
        node_id = f"SOC_{i}"

        G.add_node(
            node_id,
            nivel="socio",
            tipo=tipo_node,
            label=nome,
            nome=nome,
            documento=r.get("documento", None),
            cnpj_basico=None,
            eh_empresa_foco=False,
        )
        # aresta sócio -> foco
        G.add_edge(
            node_id,
            foco_id,
            tipo_relacao="sociedade_foco",
            qualificacao=r.get("qualificacao", None),
            data_entrada=str(r.get("data_entrada_sociedade", "")),
        )

        map_nome_to_node[nome] = node_id

    # empresas vinculadas (nível 3)
    if not df_empresas_vinc.empty:
        # info de cada empresa única
        emp_unique = (
            df_empresas_vinc.dropna(subset=["cnpj_basico"])
            .drop_duplicates(subset=["cnpj_basico"])
            .reset_index(drop=True)
        )

        for _, r in emp_unique.iterrows():
            cnpj_b = str(r["cnpj_basico"])
            emp_id = f"EMP_{cnpj_b}"

            G.add_node(
                emp_id,
                nivel="empresa_vinc",
                tipo="PJ",
                label=r["razao_social"],
                nome=r["razao_social"],
                cnpj_basico=cnpj_b,
                eh_empresa_foco=False,
            )

        # arestas sócio -> empresa
        for _, r in df_empresas_vinc.iterrows():
            nome_socio = r["nome_socio"]
            socio_id = map_nome_to_node.get(nome_socio)
            cnpj_b = r["cnpj_basico"]
            if socio_id is None or pd.isna(cnpj_b):
                continue
            emp_id = f"EMP_{cnpj_b}"

            if emp_id not in G.nodes:
                continue

            G.add_edge(
                socio_id,
                emp_id,
                tipo_relacao="sociedade",
                qualificacao=r.get("qualificacao", None),
                data_entrada=str(r.get("data_entrada_sociedade", "")),
            )

    # degree
    degree_dict = dict(G.degree())
    nx.set_node_attributes(G, degree_dict, "degree")

    # convenience: marca label_visible pra foco + sócios
    for n, data in G.nodes(data=True):
        nivel = data.get("nivel")
        label_vis = (n == foco_id) or (nivel == "socio")
        data["label_visible"] = "true" if label_vis else "false"

    log(f"Grafo construído: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas.")
    return G


# ============================================================
# LAYOUT E ELEMENTOS CYTOSCAPE
# ============================================================

SCALE = 700.0
R_MIN_SOCIOS = 650.0
R_MAX_SOCIOS = 1100.0

def compute_positions(G: nx.Graph, foco_id: str = "FOCO", seed: int = 42, k: float = 2.0):
    log(f"Calculando layout (spring) com seed={seed}, k={k}...")
    pos = nx.spring_layout(
        G,
        k=k,
        iterations=400,
        seed=seed,
        weight=None,
    )

    # recentra no foco
    cx, cy = pos[foco_id]
    for n in pos:
        x, y = pos[n]
        pos[n] = ((x - cx) * SCALE, (y - cy) * SCALE)

    socios_ids = [n for n, d in G.nodes(data=True) if d.get("nivel") == "socio"]
    R_inner = 0.8 * R_MIN_SOCIOS  # empresas ficam dentro disso

    # foco no centro
    pos[foco_id] = (0.0, 0.0)

    # empresas vinculadas ficam dentro do círculo interno
    for n, data in G.nodes(data=True):
        if n == foco_id:
            continue
        if data.get("nivel") == "empresa_vinc":
            x, y = pos[n]
            r = math.hypot(x, y)
            if r > 0 and r > R_inner:
                fator = R_inner / r
                pos[n] = (x * fator, y * fator)

    # sócios: respeita ângulo, mas clampa raio entre min e max
    for sid in socios_ids:
        x, y = pos[sid]
        r = math.hypot(x, y)
        if r == 0:
            r = (R_MIN_SOCIOS + R_MAX_SOCIOS) / 2.0
            theta = 2 * math.pi * (hash(sid) % 360) / 360.0
            x, y = r * math.cos(theta), r * math.sin(theta)
        else:
            r_clamped = max(R_MIN_SOCIOS, min(R_MAX_SOCIOS, r))
            fator = r_clamped / r
            x, y = x * fator, y * fator
        pos[sid] = (x, y)

    log("Layout calculado.")
    return pos


def build_cytoscape_elements(G: nx.Graph, pos, selected_ids):
    selected_set = set(selected_ids or [])
    neighbor_ids = set()
    for sid in selected_set:
        neighbor_ids |= set(G.neighbors(sid))

    elements = []

    # nós
    for nid, data in G.nodes(data=True):
        x, y = pos[nid]
        if not selected_set:
            highlight = "none"
        else:
            if nid in selected_set:
                highlight = "self"
            elif nid in neighbor_ids:
                highlight = "neighbor"
            else:
                highlight = "dim"

        node_entry = {
            "data": {
                "id": nid,
                "label": data.get("label", ""),
                "tipo": data.get("tipo", ""),
                "nivel": data.get("nivel", ""),
                "eh_empresa_foco": str(bool(data.get("eh_empresa_foco", False))).lower(),
                "degree": int(data.get("degree", 0)),
                "highlight": highlight,
            },
            "position": {"x": float(x), "y": float(y)},
        }
        elements.append(node_entry)

    # arestas
    for u, v, data in G.edges(data=True):
        if not selected_set:
            highlight = "none"
        else:
            if u in selected_set or v in selected_set:
                highlight = "connected"
            else:
                highlight = "dim"

        edge_entry = {
            "data": {
                "id": f"{u}__{v}",
                "source": u,
                "target": v,
                "tipo_relacao": data.get("tipo_relacao", ""),
                "qualificacao": data.get("qualificacao", ""),
                "data_entrada": str(data.get("data_entrada", "")),
                "highlight": highlight,
            }
        }
        elements.append(edge_entry)

    return elements


def get_stylesheet():
    return [
        # base
        {
            "selector": "node",
            "style": {
                "width": "mapData(degree, 0, 20, 12, 40)",
                "height": "mapData(degree, 0, 20, 12, 40)",
                "border-width": 1,
                "border-color": "#555",
                "opacity": 1.0,
            },
        },
        # PF azul
        {
            "selector": "node[tipo = 'PF']",
            "style": {
                "background-color": "#1f77b4",
            },
        },
        # PJ branco
        {
            "selector": "node[tipo = 'PJ']",
            "style": {
                "background-color": "#ffffff",
            },
        },
        # foco verde
        {
            "selector": "node[eh_empresa_foco = 'true']",
            "style": {
                "background-color": "#2ca02c",
                "border-width": 3,
            },
        },
        # labels fixos
        {
            "selector": "node[label_visible = 'true']",
            "style": {
                "label": "data(label)",
                "font-size": "10px",
                "text-valign": "center",
                "text-halign": "center",
                "color": "#000000",
            },
        },
        # dim
        {
            "selector": "node[highlight = 'dim']",
            "style": {
                "opacity": 0.15,
            },
        },
        # nó selecionado
        {
            "selector": "node[highlight = 'self']",
            "style": {
                "opacity": 1.0,
                "border-width": 4,
                "border-color": "#000000",
                "label": "data(label)",
                "font-size": "11px",
            },
        },
        # vizinho
        {
            "selector": "node[highlight = 'neighbor']",
            "style": {
                "opacity": 0.95,
                "border-width": 2,
                "border-color": "#333333",
                "label": "data(label)",
                "font-size": "10px",
            },
        },
        # arestas base
        {
            "selector": "edge",
            "style": {
                "line-color": "#bbbbbb",
                "width": 1,
                "curve-style": "bezier",
                "opacity": 0.6,
            },
        },
        # arestas dim
        {
            "selector": "edge[highlight = 'dim']",
            "style": {
                "opacity": 0.1,
            },
        },
        # arestas conectadas
        {
            "selector": "edge[highlight = 'connected']",
            "style": {
                "line-color": "#555555",
                "width": 2,
                "opacity": 0.95,
            },
        },
    ]


# ============================================================
# UI PRINCIPAL
# ============================================================

col_input, col_info = st.columns([2, 1])

with col_input:
    cnpj_input = st.text_input(
        "CNPJ da empresa foco",
        value="54.651.716/0001-88",
        help="Pode informar com ou sem separadores.",
    )
    run_btn = st.button("🚀 Gerar análise", type="primary")

with col_info:
    st.markdown("### Sobre")
    st.markdown(
        "- Base: `basedosdados.br_me_cnpj.*`\n"
        "- Nível 1: empresa foco\n"
        "- Nível 2: sócios do QSA atual\n"
        "- Nível 3: empresas em que esses sócios aparecem no QSA"
    )

progress = st.empty()


def set_progress(pct: int, msg: str):
    progress.progress(pct / 100.0, text=msg)
    log(msg)


# ------------------------------------------------------------
# EXECUÇÃO
# ------------------------------------------------------------
if run_btn:
    try:
        st.session_state["logs"] = []
        render_logs()

        set_progress(5, "Normalizando CNPJ...")
        cnpj_14 = normalizar_cnpj(cnpj_input)
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

        st.session_state["selected_nodes"] = []  # limpa seleção ao trocar CNPJ

        # ----------------------------------------------------
        # VISUALIZAÇÃO DO GRAFO
        # ----------------------------------------------------
        set_progress(90, "Calculando layout e renderizando grafo...")

        seed = st.session_state["layout_seed"]
        pos = compute_positions(G, foco_id="FOCO", seed=seed, k=2.0)

        selected_ids = st.session_state["selected_nodes"]
        elements = build_cytoscape_elements(G, pos, selected_ids)
        stylesheet = get_stylesheet()

        # Botões de layout e limpeza
        col_b1, col_b2 = st.columns([1, 1])
        with col_b1:
            if st.button("🔁 Recalcular layout (shake)", use_container_width=True):
                st.session_state["layout_seed"] = int(time.time()) % 100000
                st.experimental_rerun()

        with col_b2:
            if st.button("🧹 Limpar seleção", use_container_width=True):
                st.session_state["selected_nodes"] = []
                st.experimental_rerun()

        # Grafo
        selection = cytoscape(
            elements=elements,
            stylesheet=stylesheet,
            layout={"name": "preset", "fit": True, "padding": 140},
            width="100%",
            height="900px",
            selection_type="additive",
            key="grafo-cnpj",
        )

        new_selected_ids = selection.get("nodes", [])

        # se seleção mudou, atualiza state e reroda pra atualizar highlight
        if set(new_selected_ids) != set(st.session_state["selected_nodes"]):
            st.session_state["selected_nodes"] = new_selected_ids
            st.experimental_rerun()

        set_progress(100, "Pronto! Grafo gerado com sucesso.")
        time.sleep(0.5)
        progress.empty()

        # ----------------------------------------------------
        # TABELAS DINÂMICAS ABAIXO DO GRAFO
        # ----------------------------------------------------
        st.markdown("### 📊 Detalhes dos nós selecionados")

        selected_ids = st.session_state["selected_nodes"]
        if not selected_ids:
            st.info("Selecione um ou mais nós no grafo para ver detalhes aqui embaixo.")
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

                # foco
                if nivel == "foco":
                    st.markdown("**Empresa foco**")
                    st.dataframe(df_empresa_foco)

                    # QSA atual
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

                # sócio
                elif nivel == "socio" and nome is not None:
                    df_socio_foco = (
                        df_socios_qsa[df_socios_qsa["nome"] == nome]
                        .sort_values("data_entrada_sociedade", ascending=False)
                    )
                    if not df_socio_foco.empty:
                        st.markdown("**Registro do sócio no QSA da empresa foco (mais recente primeiro)**")
                        st.dataframe(df_socio_foco)

                    df_emp_socio = (
                        df_empresas_vinc[df_empresas_vinc["nome_socio"] == nome]
                        .sort_values(["data", "razao_social"], ascending=[False, True])
                    )
                    if not df_emp_socio.empty:
                        st.markdown("**Empresas em que este sócio aparece no QSA**")
                        st.dataframe(df_emp_socio)

                # empresa vinculada
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

                        df_qsa_emp = df_emp[
                            ["nome_socio", "documento", "qualificacao", "data_entrada_sociedade"]
                        ].sort_values("nome_socio")
                        st.markdown("**QSA (amostra a partir dos sócios do grupo)**")
                        st.dataframe(df_qsa_emp)

    except Exception as e:
        st.error(f"Erro durante a execução: {e}")
        log(f"ERRO: {e}")
        progress.empty()

