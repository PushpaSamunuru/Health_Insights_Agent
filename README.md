# 🩺 Health Insights Agent (HIA)

An AI-powered web application that analyzes blood reports and presents
human-friendly health insights using Large Language Models.

#DEMO Screenshots
<img width="1920" height="1080" alt="Screenshot 2025-12-18 082734" src="https://github.com/user-attachments/assets/edd97de7-7aa9-4144-96db-108d79e3bf63" />
<img width="1920" height="1080" alt="Screenshot 2025-12-18 082838" src="https://github.com/user-attachments/assets/8afa2d03-66ee-474f-94e6-dc1eca147f79" />


## 🌟 Features
- Upload blood report PDFs
- Extract and analyze lab values
- AI-generated health summary in simple language
- Secure login & session tracking
- Clean, responsive UI built with Streamlit

## 🛠️ Tech Stack
- Streamlit
- Python
- Groq LLMs
- Supabase
- PDFPlumber
- FAISS + Embeddings

## 🚀 Installation
```bash
git clone https://github.com/PushpaSamunuru/Health_Insights_Agent.git
cd hia
pip install -r requirements.txt
streamlit run src/main.py
