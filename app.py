import gc

import streamlit as st
import torch
from scripts.predict import carregar_modelo, gerar

st.set_page_config(page_title='CodeFix AI', page_icon='🧠')
st.title('CodeFix AI')
st.caption('Depuração de código Python com LLM local usando GPU CUDA')

experimento = st.selectbox('Modelo', ['Original (sem fine-tuning)', 'Experimento E1', 'Experimento E2', 'Experimento E3'])
paths = {'Original (sem fine-tuning)': None, 'Experimento E1': 'models/e1/adapter',
         'Experimento E2': 'models/e2/adapter', 'Experimento E3': 'models/e3/adapter'}
texto = st.text_area('Cole o código Python:', value='print(10 / 0)', height=180)

def iniciar(adapter):
    # Mantém somente um modelo na VRAM por sessão.
    if st.session_state.get('adapter_atual') != str(adapter):
        st.session_state.pop('modelo_carregado', None)
        st.session_state.pop('adapter_atual', None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if 'modelo_carregado' not in st.session_state:
        st.session_state['modelo_carregado'] = carregar_modelo(adapter)
        st.session_state['adapter_atual'] = str(adapter)
    return st.session_state['modelo_carregado']

if st.button('Analisar erro', type='primary'):
    if not texto.strip():
        st.warning('Informe um trecho de código.')
    else:
        try:
            tokenizer, model = iniciar(paths[experimento])
            with st.spinner('Gerando resposta usando a GPU...'):
                resposta = gerar(texto, tokenizer, model)
            st.subheader('Análise')
            st.code(resposta, language='text')
        except Exception as exc:
            st.error(f'Falha na execução: {exc}')

st.info('O código informado não é executado. O modelo apenas analisa texto.')
