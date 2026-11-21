import os
import streamlit as st
from workflow import run_workflow

st.set_page_config(page_title='Multi-Agent Feedback Lab', page_icon=':robot_face:')
st.title(':robot_face: Multi-Agent Customer Feedback Lab')
st.caption('Advanced AI classroom | Streamlit + LangGraph + Groq-hosted open-weight model')
st.markdown('**Workflow:** Message → Gatekeeper agent → Sentiment agent → Positive or Negative response agent')

try:
    if 'GROQ_API_KEY' in st.secrets:
        os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']
    if 'GROQ_MODEL' in st.secrets:
        os.environ['GROQ_MODEL'] = st.secrets['GROQ_MODEL']
except FileNotFoundError:
    pass

examples = [
    'I love your service! The staff were helpful.',
    'I hate your services. The delivery was late.',
    'What is the capital of France?',
]
example = st.selectbox('Example message (optional)', ['Write my own'] + examples)
feedback = st.text_area('Customer message', value='' if example == 'Write my own' else example, height=120)
if st.button('Run multi-agent workflow', type='primary'):
    if not feedback.strip():
        st.warning('Enter a message first.')
    elif not os.getenv('GROQ_API_KEY'):
        st.error('GROQ_API_KEY is missing. Add it in Streamlit Community Cloud → App settings → Secrets.')
    else:
        with st.spinner('Running agents...'):
            try:
                result = run_workflow(feedback.strip())
            except Exception as exc:
                st.error(f'Workflow failed: {exc}')
            else:
                st.success('Workflow complete')
                left, right = st.columns(2)
                left.metric('Gatekeeper agent', result['scope'])
                right.metric('Sentiment agent', result['sentiment'])
                if result['scope'] == 'Other':
                    st.write('**Selected path:** Off-topic standard reply (no response agent used)')
                    st.write('**Final response:**')
                    st.warning(result['response'])
                else:
                    st.write('**Selected response agent:**', result['sentiment'])
                    st.write('**Final response:**')
                    st.info(result['response'])
                with st.expander('Workflow state (for students)'):
                    st.json(result)
st.divider()
st.caption('Feedback uses three LLM requests per run; off-topic messages use one. '
           'Free Groq API usage is rate-limited; never commit API keys to GitHub.')
