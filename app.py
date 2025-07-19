import os
os.environ["STREAMLIT_SERVER_ENABLE_WATCHDOG"] = "false"
import time
import streamlit as st
from urllib.parse import urlparse, parse_qs
from streamlit_extras.stylable_container import stylable_container
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from groq import Groq
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


st.set_page_config(
    page_title="🎬 YouTube AI Tutor Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .summary-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .question-card {
        background: rgba(255,255,255,0.9);
        border-left: 4px solid #6366f1;
        border-radius: 0 10px 10px 0;
        padding: 15px;
        margin: 10px 0;
    }
    .progress-bar {
        height: 5px;
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .hover-card:hover {
        transform: translateY(-3px);
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

def animated_progress():
    progress_bar = st.empty()
    for percent_complete in range(100):
        progress_bar.markdown(f"""
        <div class="progress-bar" style="width: {percent_complete}%;"></div>
        """, unsafe_allow_html=True)
        time.sleep(0.02)
    progress_bar.empty()

def extract_video_id(url):
    try:
        if "youtu.be" in url:
            return url.split("/")[-1].split("?")[0]
        elif "youtube.com" in url:
            query = parse_qs(urlparse(url).query)
            return query.get("v", [""])[0]
    except Exception:
        return ""
    return ""

def save_transcript_to_file(text, video_id):
    """Saves transcript with video-specific filename"""
    filename = f"transcript_{video_id}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    return filename

def get_youtube_transcript(url):
    try:
        video_id = extract_video_id(url)
        
        if not st.session_state.get("force_refresh", False):
            if "cached_transcripts" in st.session_state:
                if video_id in st.session_state.cached_transcripts:
                    st.toast("💾 Using cached transcript", icon="✅")
                    return st.session_state.cached_transcripts[video_id], video_id
        
        with st.spinner("🌐 Fetching fresh transcript..."):
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            text = " ".join([segment['text'] for segment in transcript])
            
            if "cached_transcripts" not in st.session_state:
                st.session_state.cached_transcripts = {}
            st.session_state.cached_transcripts[video_id] = text
            
            save_transcript_to_file(text, video_id)
            return text, video_id
            
    except Exception as e:
        st.error(f"❌ Error: {type(e).__name__}: {str(e)}")
        return None, None

def get_compressed_summary(transcript):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Extract 5-7 key points with emoji headings in markdown format"
            },
            {"role": "user", "content": f"Transcript:\n{transcript[:10000]}"},
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.2
    )
    return response.choices[0].message.content

def ask_groq(question, context):
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Answer concisely in 2-3 sentences"},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}"},
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.7
    )
    return response.choices[0].message.content

with st.sidebar:
    st.title("⚙️ Settings")
    with st.expander("🔧 Advanced Options"):
        model_choice = st.radio(
            "AI Model",
            [ "meta-llama/llama-4-scout-17b-16e-instruct (Recommended)" ,"Llama3-70B "],
            index=0
        )
        summary_length = st.slider("Summary Length", 3, 10, 5)

st.title("🎬 AI POWERED YOUTUBE TRANSCRIPTOR ")
st.caption("Transform videos into interactive learning experiences")

with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        video_url = st.text_input(
            "📺 Paste YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed"
        )
    with col2:
        force_refresh = st.checkbox(
            "Force refresh", 
            help="Always fetch fresh transcript",
            key="force_refresh"
        )

    if st.button("✨ Process Video", use_container_width=True):
        if video_url:
            with st.status("🚀 Processing video...", expanded=True) as status:
                current_video_id = extract_video_id(video_url)
                
                if (force_refresh or 
                    ("last_video_id" in st.session_state and 
                     st.session_state.last_video_id != current_video_id)):
                    
                    if "cached_transcripts" in st.session_state:
                        st.session_state.cached_transcripts.pop(current_video_id, None)
                    st.session_state.pop("summary", None)
                    st.session_state.pop("vectorstore", None)
                    st.toast("♻️ Cleared previous video data", icon="⚠️")
                
                transcript_text, video_id = get_youtube_transcript(video_url)
                
                if transcript_text:

                    st.session_state.last_video_id = current_video_id
                    
                    st.write("Generating smart summary...")
                    animated_progress()
                    st.session_state.summary = get_compressed_summary(transcript_text)
                    
                    st.write("Building knowledge base...")
                    loader = TextLoader(f"transcript_{video_id}.txt")
                    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    docs = splitter.split_documents(loader.load())
                    st.session_state.vectorstore = FAISS.from_documents(
                        docs, 
                        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    )
                    status.update(label="✅ Processing complete!", state="complete")
                else:
                    status.update(label="❌ Failed to get transcript", state="error")

if "summary" in st.session_state:
    with st.container():
        st.subheader("📌 Smart Summary")
        with stylable_container(
            key="summary-card",
            css_styles="""
            {
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            """
        ):
            st.markdown(st.session_state.summary)
        
        st.subheader("💡 Suggested Questions")
        questions = [
            "What are the main concepts covered?",
            "Can you explain the key takeaways?",
            "What practical examples were given?"
        ]
        cols = st.columns(3)
        for i, q in enumerate(questions):
            with cols[i]:
                if st.button(q, use_container_width=True):
                    st.session_state.user_question = q

if "vectorstore" in st.session_state:
    st.divider()
    
    user_question = st.text_input(
        "🤔 Ask anything about the video",
        value=st.session_state.get("user_question", ""),
        key="question_input"
    )
    
    if user_question:
        with st.spinner("💭 Generating answer..."):
            docs = st.session_state.vectorstore.similarity_search(user_question, k=2)
            answer = ask_groq(user_question, "\n".join([doc.page_content for doc in docs]))
            
            with stylable_container(
                key="answer-card",
                css_styles="""
                {
                    background: rgba(236, 253, 245, 0.8);
                    border-radius: 10px;
                    padding: 20px;
                    margin-top: 15px;
                }
                """
            ):
                st.markdown(f"**🎯 Answer:** {answer}")
                
                
                col1, col2, col3 = st.columns([1,1,8])
                with col1:
                    if st.button("👍", help="Good answer"):
                        st.toast("Thanks for your feedback!")
                with col2:
                    if st.button("👎", help="Needs improvement"):
                        st.toast("We'll improve this!")
                
                with st.expander("📜 View relevant transcript"):
                    st.write(docs[0].page_content)


st.divider()
st.caption("✨ Powered by LLM(groq) & Streamlit | Made with ❤️ for YouTube learners")