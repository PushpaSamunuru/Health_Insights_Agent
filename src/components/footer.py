import streamlit as st

def show_footer(in_sidebar: bool = False):
    margin = "0" if in_sidebar else "2rem"

    html = f"""
    <div style="
        text-align: center;
        padding: 0.9rem;
        margin-top: {margin};
        background: linear-gradient(to right,
            rgba(25,118,210,0.03),
            rgba(100,181,246,0.05),
            rgba(25,118,210,0.03)
        );
        border-top: 1px solid rgba(100,181,246,0.15);
        box-shadow: 0 -2px 8px rgba(100,181,246,0.05);
    ">
        <div style="
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 8px;
            background: rgba(100,181,246,0.08);
            color: #1976D2;
            font-weight: 600;
            font-size: 0.8rem;
        ">
            🩺 Health Insights Agent
        </div>

        <div style="
            margin-top: 6px;
            color: #546E7A;
            font-size: 0.75rem;
        ">
            AI-powered report summaries • Educational use only
        </div>

        <div style="
            margin-top: 4px;
            color: #90A4AE;
            font-size: 0.7rem;
        ">
            Not a medical diagnosis. Please consult a doctor.
        </div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)
