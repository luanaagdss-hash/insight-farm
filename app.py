import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# -----------------------
# Helpers para IA
# -----------------------

def send_prompt_openai(prompt):
    try:
        import openai
    except Exception as e:
        return None, f"openai library not available: {e}"

    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.2,
        )
        return response.choices[0].message.content, None
    except Exception:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.2,
            )
            return response["choices"][0]["message"]["content"], None
        except Exception as e:
            return None, str(e)

def send_prompt_hf(prompt):
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        return None, "HF_TOKEN not set"

    model = "mistralai/Mistral-7B-Instruct-v0.1"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 500}}

    try:
        r = requests.post(
            f"https://router.huggingface.co/hf-inference/models/{model}",
            headers=headers,
            json=payload,
            timeout=30
        )
    except Exception as e:
        return None, f"Request failed: {e}"

    if r.status_code != 200:
        return None, f"HuggingFace error {r.status_code}: {r.text[:300]}"

    try:
        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            text = data[0].get("generated_text") or data[0].get("text") or str(data[0])
        else:
            text = str(data)
        return text, None
    except Exception as e:
        return None, f"JSON decode error: {e}"

def generate_template_report(cultura, regiao, custo_variavel, custo_fixo,
                             producao_esperada, preco_mercado, elasticidade,
                             concorrencia, clima, ponto_equilibrio_unidades):
    """
    Template determinístico (fallback) com formatação segura.
    Retorna um dicionário com 'text' (parágrafos) e 'formulas' (lista de latex strings).
    """
    # Garanta formatação numérica correta
    cv = f"{custo_variavel:,.2f}"
    cf = f"{custo_fixo:,.2f}"
    pm = f"{preco_mercado:,.2f}"
    pe = f"{ponto_equilibrio_unidades:,.0f}"
    el = f"{elasticidade:.2f}"

    text_lines = []
    text_lines.append("(1) Interpretação microeconômica:")
    text_lines.append(f"Cultura: {cultura} — Região: {regiao}.")
    text_lines.append(f"Com custo variável por unidade de R$ {cv} e custo fixo total estimado em R$ {cf},")
    text_lines.append(f"a produção esperada é de {producao_esperada} toneladas ao preço médio de R$ {pm}.")
    text_lines.append(f"A margem unitária (preço - custo variável) e o ponto de equilíbrio orientam a decisão de plantio.")
    text_lines.append(f"O ponto de equilíbrio estimado é de aproximadamente {pe} toneladas.\n")

    text_lines.append("(2) Riscos e suposições:")
    text_lines.append(f"Este relatório assume elasticidade-preço constante aproximada de {el}.")
    text_lines.append(f"Riscos principais: variação climática ({clima}), flutuações de preço e custos, e reação da concorrência (≈ {concorrencia} produtores).")
    text_lines.append("Mitigações: contratos futuros, seguros agrícolas e diversificação.\n")

    text_lines.append("(3) Recomendação prática:")
    text_lines.append("Recomenda-se testar políticas de venda antecipada (parcial) e realizar um experimento A/B em preço ou mix de canais para avaliar elasticidade real.")
    text_lines.append("Métrica de sucesso: aumento do lucro líquido por hectare sem queda substancial no volume.\n")

    text_lines.append("(4) Métricas para acompanhar:")
    text_lines.append("Acompanhar mensalmente: lucro líquido por hectare, ponto de equilíbrio, custo marginal, receita média por tonelada, elasticidade observada e índice de competitividade regional.")

    # Fórmulas em LaTeX (strings) — renderizaremos com st.latex no frontend
    formulas = []
    # margem unitária
    formulas.append(r"\text{Margem unitária} = \text{Preço} - \text{Custo Variável}")
    # custo médio total (exemplo)
    formulas.append(r"\text{CMT} = \frac{\text{Custo Fixo Total} + \text{Custo Variável Total}}{\text{Produção Esperada}}")
    # ponto de equilíbrio
    formulas.append(r"\text{PE (ton)} = \frac{\text{Custo Fixo Total}}{\text{Preço} - \text{Custo Variável}}")

    return {
        "text": "\n\n".join(text_lines),
        "formulas": formulas,
        "values": {
            "custo_variavel": cv,
            "custo_fixo": cf,
            "preco_mercado": pm,
            "ponto_equilibrio": pe,
            "elasticidade": el
        }
    }   return f"""
(1) Interpretação microeconômica:
Cultura: {cultura} — Região: {regiao}.
Com custo variável por unidade de R$ {custo_variavel:.2f} e custo fixo total estimado em R$ {custo_fixo:.2f}, a produção esperada é de {producao_esperada} toneladas ao preço médio de R$ {preco_mercado:.2f}. A margem unitária (preço - custo variável) e o ponto de equilíbrio orientam a decisão de plantio. O ponto de equilíbrio estimado é de aproximadamente {ponto_equilibrio_unidades:,.0f} toneladas.

(2) Riscos e suposições:
Este relatório assume elasticidade-preço constante aproximada de {elasticidade:.2f}. Riscos principais incluem variação climática ({clima}), flutuações de preço e custos, além de reação da concorrência (≈ {concorrencia} produtores). Mitigações: contratos futuros, seguros agrícolas e diversificação.

(3) Recomendação prática:
Recomenda-se testar políticas de venda antecipada (parcial) e realizar um experimento A/B em preço ou mix de canais para avaliar elasticidade real. Métrica de sucesso: aumento do lucro líquido por hectare sem queda substancial no volume.

(4) Métricas para acompanhar:
Acompanhar mensalmente: lucro líquido por hectare, ponto de equilíbrio, custo marginal, receita média por tonelada, elasticidade observada e índice de competitividade regional.
"""

# -----------------------
# Interface Streamlit
# -----------------------

st.set_page_config(page_title="InsightFarm — Estratégia Agrícola IA", layout="wide")
st.title("🌾 InsightFarm — Estratégia de Produção Agrícola com IA")
st.markdown("Preencha os dados abaixo e gere um relatório com recomendações microeconômicas detalhadas.")

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
    margem_unitaria = preco_mercado - custo_variavel
    faturamento = preco_mercado * producao_esperada
    lucro = faturamento - (custo_fixo + custo_variavel * producao_esperada)
    ponto_equilibrio_unidades = custo_fixo / max(margem_unitaria, 1e-6)

    st.subheader("📊 Métricas básicas")
    st.write(f"**Margem unitária (R$/ton):** R$ {margem_unitaria:.2f}")
    st.write(f"**Faturamento esperado:** R$ {faturamento:,.2f}")
    st.write(f"**Lucro esperado:** R$ {lucro:,.2f}")
    st.write(f"**Ponto de equilíbrio (ton):** {ponto_equilibrio_unidades:,.0f}")

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

    st.markdown(f"**💰 Preço ótimo sugerido:** R$ {preco_otimo:.2f} — *Lucro estimado: R$ {lucro_otimo:,.2f}*")

    # ---- PROMPT LLM COMPLETO ----
    prompt = f"""
Você é um economista agrícola sênior com forte domínio de microeconomia aplicada, precificação, experimentos A/B e elaboração de relatórios técnicos executáveis.

Dados (use os valores fornecidos):
- Cultura: {cultura}
- Região: {regiao}
- Custo variável por unidade (R$): {custo_variavel}
- Custo fixo total estimado (R$): {custo_fixo}
- Produção esperada (ton): {producao_esperada}
- Preço médio de mercado (R$/ton): {preco_mercado}
- Elasticidade-preço estimada: {elasticidade}
- Concorrência regional (nº produtores): {concorrencia}
- Expectativa climática: {clima}
- Resultado da simulação (preço ótimo: R$ {preco_otimo:.2f}, lucro estimado: R$ {lucro_otimo:,.2f}, ponto de equilíbrio: {ponto_equilibrio_unidades:.0f} tons)

Objetivo: gere um RELATÓRIO TÉCNICO COMPLETO com seções numeradas e cabeçalhos claros. Use linguagem técnica, mas com recomendações práticas e executáveis. Inclua fórmulas, tabelas resumidas em texto e um plano de ação.

Estrutura requerida:
A) Resumo executivo (3–4 frases) com recomendação principal.
B) Análise microeconômica detalhada:
   - Interprete elasticidade, margem unitária, custo marginal e custo médio.
   - Calcule e explique o ponto de equilíbrio (unidades/ton) e sensibilidade ao preço.
C) Cenários (Pessimista / Base / Otimista):
   - Defina variações plausíveis (% de preço e % de volume).
   - Para cada cenário, apresente: preço, quantidade, faturamento, custo total e lucro.
   - Apresente uma tabela resumida (texto/tabular).
D) Sensibilidade por preço:
   - Mostre lucros para -10%, -5%, 0%, +5%, +10% no preço.
   - Identifique preço reserva (preço mínimo que cobre custo variável) e confirme preço que maximiza lucro.
E) Design de teste A/B para precificação:
   - Hipótese nula e alternativa.
   - Tamanho de amostra sugerido (estimativa prática), duração, métricas primárias e secundárias.
   - Regra de decisão para adotar o novo preço.
F) Riscos e mitigação operacional (clima, mercado, logística) com ações concretas.
G) KPIs e fórmulas: liste e defina (ex.: CAC, LTV, margem bruta, margem líquida, ticket médio).
H) Plano de ação (6 passos) para 8 semanas, com responsáveis e entregáveis.
I) Conclusão (2 frases).

Exija que o relatório explique claramente todas as suposições numéricas usadas e apresente resultados em R$ com duas casas decimais. Seja objetivo e formatado (A, B, C...). Não invente dados adicionais — use somente os valores fornecidos e calcule a partir deles.
"""

    st.subheader("📑 Relatório gerado pela IA / Fallback")

    report, err = send_prompt_openai(prompt)
    if report:
        st.write(report)
        text_to_download = report
    else:
        hf_report, hf_err = send_prompt_hf(prompt)
        if hf_report:
            st.write(hf_report)
            text_to_download = hf_report
        else:
            fallback = generate_template_report(
                cultura, regiao, custo_variavel, custo_fixo,
                producao_esperada, preco_mercado, elasticidade,
                concorrencia, clima, ponto_equilibrio_unidades
            )
            st.info("Nenhuma API generativa disponível; exibindo relatório determinístico.")
            st.write(fallback)
            text_to_download = fallback
            st.write("---")
            st.write("Debug errors (OpenAI / HuggingFace):")
            st.write(err)
            st.write(hf_err)

    st.download_button("Baixar relatório (.txt)", text_to_download, file_name="insightfarm_report.txt", mime="text/plain")

st.markdown("---")
st.caption("InsightFarm — Protótipo de estratégia agrícola com geração de relatório. (Use Secrets para OPENAI_API_KEY ou HF_TOKEN)")
