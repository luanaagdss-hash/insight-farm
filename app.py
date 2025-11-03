# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time

# === Helpers para IA: tenta suportar OpenAI (novo e antigo), Hugging Face, e fallback template ===

def send_prompt_openai(prompt):
    """
    Tenta usar a API OpenAI:
    - se openai.OpenAI (nova lib) estiver presente, usa esse cliente
    - senão tenta usar openai.ChatCompletion (antiga 0.28)
    Retorna (report, error_flag)
    """
    try:
        import openai
    except Exception as e:
        return None, f"openai library not available: {e}"

    # tenta nova interface (openai>=1.0)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.2,
        )
        report = response.choices[0].message.content
        return report, None
    except Exception:
        # tenta interface antiga (0.28)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.2,
            )
            report = response["choices"][0]["message"]["content"]
            return report, None
        except Exception as e:
            return None, str(e)

def send_prompt_hf(prompt):
    """
    Usa Hugging Face Inference API se HF_TOKEN estiver presente.
    Retorna (report, error_flag)
    """
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        return None, "HF_TOKEN not set"

    model = "mistralai/Mistral-7B-Instruct-v0.1"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 400}}
       if r.status_code != 200:
    st.error(f"Erro HuggingFace: {r.status_code} - {r.text[:500]}")
    st.stop()

    try:
        r = requests.post(f"https://api-inference.huggingface.co/models/{model}", headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        # dependendo do modelo / endpoint a chave pode variar
        if isinstance(data, list):
            text = data[0].get("generated_text") or data[0].get("text") or str(data[0])
        else:
            text = data.get("generated_text") or str(data)
        return text, None
    except Exception as e:
        return None, str(e)

def generate_template_report(cultura, regiao, custo_variavel, custo_fixo, producao_esperada, preco_mercado, elasticidade, concorrencia, clima, ponto_equilibrio_unidades):
    # Template determinístico (fallback) — 4 parágrafos
    report = f"""
(1) Interpretação microeconômica:
Cultura: {cultura} — Região: {regiao}.
Com custo variável por unidade de R$ {custo_variavel:.2f} e custo fixo mensal estimado em R$ {custo_fixo:.2f}, a produção esperada é de {producao_esperada} toneladas ao preço médio de R$ {preco_mercado:.2f}. A margem unitária implícita (preço - custo variável) é um indicador-chave para a decisão de plantio. O ponto de equilíbrio estimado é de {ponto_equilibrio_unidades:.0f} unidades/toneladas.

(2) Riscos e suposições:
Este relatório assume elasticidade-preço constante aproximada de {elasticidade}. Riscos principais: variação climática ({clima}), flutuação de preços de mercado, e reação da concorrência (≈ {concorrencia} produtores). Estratégias de mitigação devem considerar contratos futuros, seguros agrícolas e diversificação de culturas.

(3) Recomendação prática:
Com base nos dados, recomenda-se testar ajuste de área de plantio e políticas de venda que protejam margem. Considere um teste piloto (A/B) de preço/contrato: por exemplo, venda antecipada de parte da produção a preço fixo e venda spot do restante. Métrica de sucesso: lucro líquido por hectare e % de cobertura de custos fixos.

(4) Métricas para acompanhar:
Monitorar mensalmente: lucro líquido por hectare, ponto de equilíbrio, custo marginal, receita média por tonelada, elasticidade observada e índice de competitividade regional.
"""
    return report

# === Interface Streamlit ===
st.set_page_config(page_title="InsightFarm — Estratégia Agrícola IA", layout="wide")
st.title("InsightFarm — Estratégia de Produção Agrícola com IA")

st.markdown("Preencha dados básicos da cultura e gere um relatório com recomendações microeconômicas.")

with st.form("inputs"):
    col1, col2 = st.columns(2)
    with col1:
        cultura = st.text_input("Cultura analisada", value="milho")
        regiao = st.text_input("Região produtora", value="Centro-Oeste")
        custo_variavel = st.number_input("Custo variável por unidade (R$)", min_value=0.0, value=2500.0, step=10.0)
        producao_esperada = st.number_input("Produção esperada (toneladas)", min_value=0.0, value=120.0, step=1.0)
    with col2:
        custo_fixo = st.number_input("Custo fixo total estimado (R$)", min_value=0.0, value=80000.0, step=100.0)
        preco_mercado = st.number_input("Preço médio de mercado (R$/ton)", min_value=0.0, value=1800.0, step=1.0)
        elasticidade = st.number_input("Elasticidade-preço estimada (ex: -1.3)", value=-1.3, step=0.1)
        concorrencia = st.number_input("Concorrência regional (nº produtores)", min_value=0, value=50, step=1)
    clima = st.text_input("Expectativa de clima / safra", value="chuvas irregulares previstas")
    submitted = st.form_submit_button("Gerar relatório")

if submitted:
    # cálculos simples de apoio
    margem_unitaria = preco_mercado - custo_variavel
    faturamento = preco_mercado * producao_esperada
    lucro = faturamento - (custo_fixo + custo_variavel * producao_esperada)
    # ponto de equilíbrio (unidades) simplificado
    ponto_equilibrio_unidades = custo_fixo / max(margem_unitaria, 1e-6)

    st.subheader("Métricas básicas")
    st.write(f"Margem unitária (R$/ton): R$ {margem_unitaria:.2f}")
    st.write(f"Faturamento esperado: R$ {faturamento:,.2f}")
    st.write(f"Lucro esperado: R$ {lucro:,.2f}")
    st.write(f"Ponto de equilíbrio (ton): {ponto_equilibrio_unidades:,.0f}")

    # gráfico (exemplo: lucro vs preço)
    precos = np.linspace(max(0.5, custo_variavel*0.8), preco_mercado*1.6, 25)
    lucros = []
    P0, Q0 = preco_mercado, producao_esperada
    for p in precos:
        q = Q0 * (p / P0) ** elasticidade
        profit = (p - custo_variavel) * q - custo_fixo
        lucros.append(profit)

    idx_best = int(np.argmax(lucros))
    preco_otimo = float(precos[idx_best])
    lucro_otimo = float(lucros[idx_best])

    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(precos, lucros)
    ax.scatter([preco_otimo], [lucro_otimo], color="red")
    ax.set_xlabel("Preço (R$/ton)")
    ax.set_ylabel("Lucro estimado (R$)")
    st.pyplot(fig)

    st.markdown(f"**Preço ótimo sugerido:** R$ {preco_otimo:.2f} — *Lucro estimado: R$ {lucro_otimo:,.2f}*")

    # monta o prompt (para enviar à IA)
    prompt = f"""
Você é um economista agrícola especializado em microeconomia aplicada ao agronegócio.

Dados:
- Cultura: {cultura}
- Região: {regiao}
- Custo variável por unidade: R$ {custo_variavel:.2f}
- Custo fixo total estimado: R$ {custo_fixo:.2f}
- Produção esperada: {producao_esperada} toneladas
- Preço médio de mercado: R$ {preco_mercado:.2f}
- Elasticidade-preço estimada: {elasticidade}
- Concorrência regional: {concorrencia}
- Expectativa de clima: {clima}

Resultados da simulação:
- Preço ótimo sugerido: R$ {preco_otimo:.2f}
- Lucro estimado (para esse preço): R$ {lucro_otimo:,.2f}
- Ponto de equilíbrio (ton): {ponto_equilibrio_unidades:.0f}

Por favor, produza um relatório técnico com 4 seções:
1) Análise microeconômica da situação atual (oferta/demanda, custo marginal, ponto de equilíbrio).
2) Cálculo e interpretação do preço ótimo e volume ideal de produção.
3) Riscos e estratégias de mitigação (clima, preço, concorrência).
4) Recomendações práticas e métricas de acompanhamento (lucro por hectare, CAC agrícola se aplicável, LTV do cliente cooperado, ticket, margem).

Seja objetivo e apresente recomendações práticas, citando as suposições.
"""

    # 1) tenta OpenAI (se disponível)
    st.subheader("Relatório gerado pela IA")
    report, err = send_prompt_openai(prompt)
    if report:
        st.write(report)
    else:
        # tenta Hugging Face
        hf_report, hf_err = send_prompt_hf(prompt)
        if hf_report:
            st.write(hf_report)
        else:
            # fallback: template determinístico
            fallback = generate_template_report(cultura, regiao, custo_variavel, custo_fixo, producao_esperada, preco_mercado, elasticidade, concorrencia, clima, ponto_equilibrio_unidades)
            st.info("Nenhuma API generativa disponível ou ocorreu erro; exibindo relatório gerado por template determinístico.")
            st.write(fallback)
            # mostrar erros técnicos para debug (apenas para você, não exibir em demo)
            st.write("---")
            st.write("Debug errors (OpenAI / HuggingFace):")
            st.write(err)
            st.write(hf_err)

    # botão para download
    try:
        text_to_download = report if report else (hf_report if hf_report else fallback)
    except Exception:
        text_to_download = fallback
    st.download_button("Baixar relatório (.txt)", text_to_download, file_name="insightfarm_report.txt", mime="text/plain")

st.markdown("---")
st.caption("InsightFarm — Protótipo de estratégia agrícola com geração de relatório. (Use Secrets para OPENAI_API_KEY ou HF_TOKEN)")
